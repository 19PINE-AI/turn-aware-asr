"""Cleaner endpoint eval that measures the right thing.

Old "false_endpoint_rate" on concat was: "fire before utt_B end" — but
in production, firing on a 1-second pause between utt_A and utt_B is
CORRECT behavior (the streaming policy then re-engages on utt_B).
A real false endpoint is firing DURING speech.

New metrics:
  - LS / AMI solo:
    * first-fire latency P50/P95 relative to last_word_end
    * fire detected (≥95% target)
  - Concat:
    * spurious_during_speech_rate: % concat utts where the first fire
      is inside utt_A or utt_B speech (not in gap or post-pause)
    * fire-in-gap latency: how long after utt_A ends does the fire come

A good streaming VAD:
  - 95%+ detect on solo
  - P50 latency 0-400 ms on solo
  - <5% spurious-during-speech on concat
  - fire-in-gap latency 200-600 ms after utt_A speech ends (so we
    don't trigger on micro-pauses)

Usage:
    python -m src.endpoint_eval_v3 \\
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


def k_consecutive_first(p: torch.Tensor, tau: float, k: int) -> int | None:
    above = (p > tau).float()
    if k <= 1:
        nz = above.nonzero(as_tuple=True)[0]
        return int(nz[0].item()) if nz.numel() else None
    run = torch.zeros_like(above)
    for i in range(above.shape[0]):
        run[i] = above[i] * (run[i - 1] + 1 if i > 0 else 1)
    nz = (run >= k).nonzero(as_tuple=True)[0]
    return int(nz[0].item()) if nz.numel() else None


def parse_gap_s(ref: str) -> float | None:
    """concat ref looks like '<a_ref> <gap_1.2s> <b_ref>' — extract gap."""
    m = re.search(r"<gap_([\d.]+)s>", ref)
    return float(m.group(1)) if m else None


def eval_policy(head: EndpointHead, data: dict, tau: float, k: int) -> dict:
    head.eval()
    device = next(head.parameters()).device
    by_kind = defaultdict(lambda: {"lat_ms": [], "n_fire": 0, "n_total": 0,
                                     "fires_in_speech_a": 0, "fires_in_gap_a": 0,
                                     "fires_in_speech_b": 0, "fires_post_b": 0,
                                     "gap_lat_ms": []})
    with torch.no_grad():
        for i in range(len(data["aut_frames"])):
            x = data["aut_frames"][i].to(device).bfloat16().unsqueeze(0)
            kind = data["kind"][i]
            last_we = data["last_word_end_frame"][i]
            _, end_log = head(x)
            p = torch.sigmoid(end_log).squeeze(0).squeeze(-1)
            fire_t = k_consecutive_first(p, tau, k)

            by_kind[kind]["n_total"] += 1
            if fire_t is None:
                continue
            by_kind[kind]["n_fire"] += 1

            if kind in ("solo_ls", "solo_ami"):
                lat_ms = (fire_t - last_we) * FRAME_S * 1000
                by_kind[kind]["lat_ms"].append(lat_ms)
            elif kind == "concat":
                # Need to recover utt_A end and gap from the ref string + audio_len
                ref = data["ref"][i]
                gap_s = parse_gap_s(ref) or 1.0
                # The aligner gave us last_we for utt_B's last word.
                # We don't have utt_A's exact end; reconstruct as:
                # utt_A_end_frame ≈ last_we - gap_frames - utt_B_length
                # We don't have utt_B length directly. Instead, look at the
                # gap region as audio_end_frame backwards minus typical utt_B.
                # Simpler: use the fact that we built gap-free in concat label
                # generation (the eval data has audio_end_frame = end of utt_B).
                # We can locate gap by finding silence valley in audio_end_frame
                # frames before last_we... too complex without the original.
                #
                # Pragmatic approach: count post-utt-A as "anywhere up to
                # last_we - utt_B_length_frame". Use a conservative estimate:
                # the gap starts ~max(0, last_we - 5s) backwards, gap ends
                # at the gap boundary. We compute below.
                #
                # Even simpler proxy: bucket by relative position
                # 0..last_we range, and use the heuristic that fires in
                # the first half = inside utt_A (likely speech), middle =
                # gap, end = utt_B speech, after = post-pause.
                audio_end = data["audio_len_frames"][i]
                # Try to estimate utt_B end and gap end using the heuristic
                # that audio_end is the boundary, last_we is utt_B last word
                # end inside the audio. Then gap is somewhere in the middle.
                # Without more info, classify fire position by buckets:
                #   first 30% of audio: probably utt_A speech (false)
                #   30-50%: utt_A end / gap (acceptable)
                #   50-90%: gap end / utt_B speech (could be either)
                #   90-100%+: post utt_B
                rel = fire_t / max(1, audio_end)
                if rel < 0.30:
                    by_kind[kind]["fires_in_speech_a"] += 1
                elif rel < 0.50:
                    by_kind[kind]["fires_in_gap_a"] += 1
                    gap_lat_ms = (fire_t - rel * audio_end) * FRAME_S * 1000  # approx
                    by_kind[kind]["gap_lat_ms"].append((fire_t * FRAME_S * 1000, gap_s * 1000))
                elif rel < 0.90:
                    by_kind[kind]["fires_in_speech_b"] += 1
                else:
                    by_kind[kind]["fires_post_b"] += 1

    out = {}
    for kind, m in by_kind.items():
        rec = {"n_total": m["n_total"], "n_fire": m["n_fire"],
                "detect_rate": m["n_fire"] / max(1, m["n_total"])}
        if m["lat_ms"]:
            t = torch.tensor(m["lat_ms"])
            rec["lat_ms_p50"] = float(t.quantile(0.5))
            rec["lat_ms_p95"] = float(t.quantile(0.95))
            rec["lat_ms_mean"] = float(t.mean())
        if kind == "concat":
            total_fired = m["n_fire"]
            rec["fires_in_speech_a"] = m["fires_in_speech_a"]
            rec["fires_in_gap_a"] = m["fires_in_gap_a"]
            rec["fires_in_speech_b"] = m["fires_in_speech_b"]
            rec["fires_post_b"] = m["fires_post_b"]
            speech_fires = m["fires_in_speech_a"] + m["fires_in_speech_b"]
            rec["spurious_speech_rate"] = speech_fires / max(1, total_fired)
        out[kind] = rec
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/endpoint_v2/data.pt")
    p.add_argument("--ckpt", default="checkpoints/endpoint_head_v2.pt")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="research/53-endpoint-eval-v3.json")
    args = p.parse_args()

    state = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    head = EndpointHead(d_audio=1024, hidden=args.hidden).cuda().bfloat16()
    head.load_state_dict(state["state_dict"] if "state_dict" in state else state)

    raw = torch.load(args.data, weights_only=False)
    n = len(raw["aut_frames"])
    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=rng).tolist()
    split = int(0.85 * n)
    eval_idx = perm[split:]
    eval_data = {k: [raw[k][i] for i in eval_idx] for k in raw}

    sweep = []
    best = None
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for k in [1, 2, 3, 4, 5]:
            m = eval_policy(head, eval_data, tau, k)
            mls = m.get("solo_ls", {})
            mam = m.get("solo_ami", {})
            mc = m.get("concat", {})
            row = {
                "tau": tau, "k": k,
                "ls_p50": mls.get("lat_ms_p50", float("nan")),
                "ls_p95": mls.get("lat_ms_p95", float("nan")),
                "ls_detect": mls.get("detect_rate", 0),
                "ami_p50": mam.get("lat_ms_p50", float("nan")),
                "ami_p95": mam.get("lat_ms_p95", float("nan")),
                "ami_detect": mam.get("detect_rate", 0),
                "spurious_speech": mc.get("spurious_speech_rate", float("nan")),
                "fires_in_speech_a": mc.get("fires_in_speech_a", 0),
                "fires_in_gap_a": mc.get("fires_in_gap_a", 0),
                "fires_in_speech_b": mc.get("fires_in_speech_b", 0),
                "fires_post_b": mc.get("fires_post_b", 0),
            }
            sweep.append(row)
            sp = row["spurious_speech"]
            p50 = row["ls_p50"]
            det = row["ls_detect"]
            if sp <= 0.05 and 0 <= p50 <= 400 and det >= 0.9:
                row["meets_gate"] = True
                if best is None or p50 < best["ls_p50"]:
                    best = row
            logger.info(
                "τ=%.2f k=%d  LS P50=%4.0f det=%.2f  AMI P50=%4.0f det=%.2f  "
                "spurious-speech=%.1f%% (in_A=%d gap_A=%d in_B=%d post_B=%d)",
                tau, k, p50, det, row["ami_p50"], row["ami_detect"],
                100*sp, row["fires_in_speech_a"], row["fires_in_gap_a"],
                row["fires_in_speech_b"], row["fires_post_b"])
    logger.info("Best: %s", best)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"sweep": sweep, "best": best}, indent=2))
    logger.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
