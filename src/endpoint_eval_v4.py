"""Multi-fire endpoint eval: counts ALL fires, not just first.

A conversation VAD should fire at EVERY turn boundary, not just one.
For concat (utt_A + gap + utt_B), an ideal head fires twice:
  - once after utt_A speech ends (during the gap)
  - once after utt_B speech ends (post-pause)

This eval:
  - Solo: as before (first-fire latency)
  - Concat: per-utt pair, count distinct fire events (refractory K frames),
    report distribution of fire positions, and check whether the model
    found BOTH expected boundaries
  - AMI conv. doubles: same as concat but on real AMI audio

A fire event = first frame after refractory that crosses τ for K_consec
frames. We slide forward, after each fire skip ahead refractory_frames.

Usage:
    python -m src.endpoint_eval_v4 \\
        --data data/endpoint_v2/data.pt \\
        --ckpt checkpoints/endpoint_head_v2.pt
"""

from __future__ import annotations
import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch

from src.endpoint_head import EndpointHead

logger = logging.getLogger(__name__)
FRAME_S = 0.08
SR = 16000


def k_consec_runs(p: torch.Tensor, tau: float, k: int,
                  refractory: int = 25) -> list[int]:
    """Find all start indices where p > tau holds for k consecutive frames.

    After each found event, skip ahead by refractory frames (default 2s).
    """
    above = (p > tau).float()
    if k <= 1:
        events = []
        i = 0
        while i < above.shape[0]:
            if above[i] > 0.5:
                events.append(i)
                i += refractory
            else:
                i += 1
        return events
    # Compute run-of-trues at each position
    runs = torch.zeros_like(above)
    for i in range(above.shape[0]):
        runs[i] = above[i] * (runs[i - 1] + 1 if i > 0 else 1)
    events = []
    i = 0
    while i < runs.shape[0]:
        if runs[i] >= k:
            events.append(i)
            i += refractory
        else:
            i += 1
    return events


def parse_gap_s(ref: str) -> float | None:
    m = re.search(r"<gap_([\d.]+)s>", ref)
    return float(m.group(1)) if m else None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/endpoint_v2/data.pt")
    p.add_argument("--ckpt", default="checkpoints/endpoint_head_v2.pt")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--refractory-s", type=float, default=1.5)
    p.add_argument("--out", default="research/54-endpoint-multi-fire.json")
    args = p.parse_args()

    state = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    head = EndpointHead(d_audio=1024, hidden=args.hidden).cuda().bfloat16()
    head.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    head.eval()

    raw = torch.load(args.data, weights_only=False)
    n = len(raw["aut_frames"])
    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=rng).tolist()
    split = int(0.85 * n)
    eval_data = {k: [raw[k][i] for i in perm[split:]] for k in raw}

    refr = int(args.refractory_s / FRAME_S)
    logger.info("τ=%.2f k=%d refractory=%d frames (%.1f s)",
                 args.tau, args.k, refr, args.refractory_s)

    by_kind_stats = defaultdict(lambda: {
        "n_total": 0, "n_at_least_one": 0,
        "n_events_per_utt": [],
        "events_within_target_of_last_we": 0,
        "concat_found_both": 0,
        "concat_found_only_post_b": 0,
        "concat_found_only_gap": 0,
        "concat_found_neither": 0,
    })

    with torch.no_grad():
        for i in range(len(eval_data["aut_frames"])):
            x = eval_data["aut_frames"][i].cuda().bfloat16().unsqueeze(0)
            kind = eval_data["kind"][i]
            last_we = eval_data["last_word_end_frame"][i]
            _, end_log = head(x)
            p = torch.sigmoid(end_log).squeeze(0).squeeze(-1)
            events = k_consec_runs(p, args.tau, args.k, refractory=refr)
            stats = by_kind_stats[kind]
            stats["n_total"] += 1
            stats["n_events_per_utt"].append(len(events))
            if events:
                stats["n_at_least_one"] += 1
                for e in events:
                    if abs(e - last_we) <= 8:  # within ±640ms of last_we
                        stats["events_within_target_of_last_we"] += 1
                        break

            if kind == "concat":
                # Estimate gap region. ref has <gap_X.Xs> token.
                ref = eval_data["ref"][i]
                gap_s = parse_gap_s(ref) or 1.0
                audio_end = eval_data["audio_len_frames"][i]
                # Heuristic: gap region is roughly [30%, 50%] of audio
                gap_lo = int(audio_end * 0.30)
                gap_hi = int(audio_end * 0.55)
                post_b_lo = max(0, last_we - 4)
                post_b_hi = audio_end + 8
                found_in_gap = any(gap_lo <= e <= gap_hi for e in events)
                found_post_b = any(post_b_lo <= e <= post_b_hi for e in events)
                if found_in_gap and found_post_b:
                    stats["concat_found_both"] += 1
                elif found_post_b:
                    stats["concat_found_only_post_b"] += 1
                elif found_in_gap:
                    stats["concat_found_only_gap"] += 1
                else:
                    stats["concat_found_neither"] += 1

    summary = {}
    for kind, stats in by_kind_stats.items():
        n = stats["n_total"]
        events_per = stats["n_events_per_utt"]
        rec = {
            "n_total": n,
            "avg_events_per_utt": sum(events_per) / max(1, n),
            "min_events": min(events_per) if events_per else 0,
            "max_events": max(events_per) if events_per else 0,
            "fire_rate": stats["n_at_least_one"] / max(1, n),
            "hit_last_we_rate": stats["events_within_target_of_last_we"] / max(1, n),
        }
        if kind == "concat":
            rec["found_both_rate"] = stats["concat_found_both"] / max(1, n)
            rec["found_only_post_b_rate"] = stats["concat_found_only_post_b"] / max(1, n)
            rec["found_only_gap_rate"] = stats["concat_found_only_gap"] / max(1, n)
            rec["found_neither_rate"] = stats["concat_found_neither"] / max(1, n)
        summary[kind] = rec
        logger.info("[%s] n=%d  avg_events=%.2f  fire_rate=%.2f  hit_last_we=%.2f  %s",
                     kind, n, rec["avg_events_per_utt"], rec["fire_rate"],
                     rec["hit_last_we_rate"],
                     "concat: " + " ".join(f"{k.replace('found_','').replace('_rate','')}={v:.2f}"
                                            for k,v in rec.items() if k.startswith("found_")) if kind == "concat" else "")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "tau": args.tau, "k": args.k, "refractory_s": args.refractory_s,
        "summary": summary,
    }, indent=2))
    logger.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
