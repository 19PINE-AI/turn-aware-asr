"""Train EndpointHead at scale on the v2 data with threshold sweep.

Per-frame BCE on eager + end. Loss-weighted: end label is sparse,
so we upweight positive class.

Eval splits metrics by kind (solo_ls / solo_ami / concat):
  - solo_ls / solo_ami: latency P50 / P95 of first-fire vs last_word_end
  - concat: false-endpoint rate (fire in the internal-gap region)

Threshold sweep finds τ that hits the gate:
  ≤ 5 % false-endpoint on concat AND P50 latency ∈ [0, +400 ms]

Usage:
    python -m src.train_endpoint_v2 --data data/endpoint_v2/data.pt \\
        --steps 3000 --bsz 8 --hidden 256 \\
        --ckpt checkpoints/endpoint_head_v2.pt \\
        --out research/51-endpoint-head-v2-results.json
"""

from __future__ import annotations
import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.endpoint_head import EndpointHead

logger = logging.getLogger(__name__)

FRAME_S = 0.08


def _run_eval(model: EndpointHead, data: dict, threshold: float,
              gap_buffer_frames: int = 1) -> dict:
    """Evaluate at a single threshold. Returns by-kind metrics.

    A "fire" is the first frame where p_end > threshold.
    For solo: latency_ms = (first_fire - last_word_end) * FRAME_S * 1000
    For concat: false-endpoint if first fire occurs before
                last_word_end_frame - 2 frames (i.e. inside any prior gap).
    """
    model.eval()
    device = next(model.parameters()).device

    by_kind = defaultdict(lambda: {"lat_ms": [], "n_fire": 0, "n_total": 0,
                                     "n_false_endpoint": 0})
    with torch.no_grad():
        for i in range(len(data["aut_frames"])):
            x = data["aut_frames"][i].to(device).bfloat16().unsqueeze(0)
            kind = data["kind"][i]
            last_we = data["last_word_end_frame"][i]
            audio_end = data["audio_len_frames"][i]

            _, end_log = model(x)
            p = torch.sigmoid(end_log).squeeze(0).squeeze(-1)  # (T,)
            fires = (p > threshold).nonzero(as_tuple=True)[0]

            by_kind[kind]["n_total"] += 1
            if fires.numel() == 0:
                continue
            first = fires[0].item()
            by_kind[kind]["n_fire"] += 1

            if kind in ("solo_ls", "solo_ami"):
                lat_ms = (first - last_we) * FRAME_S * 1000
                by_kind[kind]["lat_ms"].append(lat_ms)
            elif kind == "concat":
                # False endpoint if fired before the actual last_word_end
                # (i.e. during utt_A or the gap).
                if first < last_we - gap_buffer_frames:
                    by_kind[kind]["n_false_endpoint"] += 1
                else:
                    # Real endpoint after last word
                    lat_ms = (first - last_we) * FRAME_S * 1000
                    by_kind[kind]["lat_ms"].append(lat_ms)
    model.train()
    out = {}
    for kind, m in by_kind.items():
        rec = {
            "n_total": m["n_total"],
            "n_fire": m["n_fire"],
            "detect_rate": m["n_fire"] / max(1, m["n_total"]),
        }
        if m["lat_ms"]:
            t = torch.tensor(m["lat_ms"])
            rec["lat_ms_p50"] = float(t.quantile(0.5))
            rec["lat_ms_p95"] = float(t.quantile(0.95))
            rec["lat_ms_mean"] = float(t.mean())
        if kind == "concat":
            rec["false_endpoint_rate"] = m["n_false_endpoint"] / max(1, m["n_total"])
            rec["n_false_endpoint"] = m["n_false_endpoint"]
        out[kind] = rec
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/endpoint_v2/data.pt")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--bsz", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--end-pos-weight", type=float, default=1.5,
                    help="positive-class weight for end loss (end labels are dense post-pause)")
    p.add_argument("--ckpt", default="checkpoints/endpoint_head_v2.pt")
    p.add_argument("--out", default="research/51-endpoint-head-v2-results.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=500)
    args = p.parse_args()

    logger.info("Loading %s…", args.data)
    raw = torch.load(args.data, weights_only=False)
    n = len(raw["aut_frames"])
    kinds = raw["kind"]
    logger.info("Loaded %d (solo_ls=%d, solo_ami=%d, concat=%d)",
                 n,
                 sum(1 for k in kinds if k == "solo_ls"),
                 sum(1 for k in kinds if k == "solo_ami"),
                 sum(1 for k in kinds if k == "concat"))

    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=rng).tolist()
    split = int(0.85 * n)
    train_idx, eval_idx = perm[:split], perm[split:]
    train_data = {k: [raw[k][i] for i in train_idx] for k in raw}
    eval_data = {k: [raw[k][i] for i in eval_idx] for k in raw}
    logger.info("Train %d / eval %d", len(train_idx), len(eval_idx))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = EndpointHead(d_audio=1024, hidden=args.hidden).to(device).bfloat16()
    n_params = sum(p.numel() for p in head.parameters()) / 1e6
    logger.info("EndpointHead params: %.3f M", n_params)

    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)

    n_train = len(train_idx)
    indices = list(range(n_train))
    cursor = 0
    rng2 = torch.Generator().manual_seed(args.seed)

    pos_weight_end = torch.tensor(args.end_pos_weight, dtype=torch.float32, device=device)

    t0 = time.perf_counter()
    for step in range(args.steps):
        if cursor + args.bsz > len(indices):
            perm2 = torch.randperm(len(indices), generator=rng2).tolist()
            indices = [indices[j] for j in perm2]
            cursor = 0
        batch_idx = indices[cursor : cursor + args.bsz]
        cursor += args.bsz

        max_T = max(train_data["aut_frames"][i].shape[0] for i in batch_idx)
        B = len(batch_idx)
        X = torch.zeros(B, max_T, 1024, dtype=torch.bfloat16, device=device)
        end_t = torch.zeros(B, max_T, dtype=torch.float32, device=device)
        eager_t = torch.zeros(B, max_T, dtype=torch.float32, device=device)
        mask = torch.zeros(B, max_T, dtype=torch.bool, device=device)
        for j, i in enumerate(batch_idx):
            x = train_data["aut_frames"][i].to(device).bfloat16()
            T = x.shape[0]
            X[j, :T] = x
            end_t[j, :T] = train_data["end_targets"][i].to(device)
            eager_t[j, :T] = train_data["eager_targets"][i].to(device)
            mask[j, :T] = True

        eager_log, end_log = head(X)
        l_eager = F.binary_cross_entropy_with_logits(
            eager_log.squeeze(-1), eager_t, reduction="none"
        )
        l_end = F.binary_cross_entropy_with_logits(
            end_log.squeeze(-1), end_t,
            pos_weight=pos_weight_end, reduction="none"
        )
        loss = ((l_eager + l_end) * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optim.step()
        optim.zero_grad()

        if step % 100 == 0 or step == args.steps - 1:
            logger.info("step %4d  loss %.4f  (%.0fs)", step, loss.item(),
                         time.perf_counter() - t0)

        if args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
            mid = _run_eval(head, eval_data, threshold=0.5)
            for kind, m in mid.items():
                logger.info("  EVAL[%s] %s", kind, m)

    # Final eval and threshold sweep
    logger.info("Final threshold sweep on eval set")
    sweep = {}
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        sweep[tau] = _run_eval(head, eval_data, threshold=tau)
        rec_concat = sweep[tau].get("concat", {})
        rec_ls = sweep[tau].get("solo_ls", {})
        rec_ami = sweep[tau].get("solo_ami", {})
        logger.info("τ=%.2f  concat FE=%.1f%%  solo_ls P50=%.0fms  solo_ami P50=%.0fms",
                     tau,
                     100 * rec_concat.get("false_endpoint_rate", 0.0),
                     rec_ls.get("lat_ms_p50", float("nan")),
                     rec_ami.get("lat_ms_p50", float("nan")))

    # Pick the best τ that meets the gate (≤5% FE AND P50 in [0, +400])
    best_tau = None
    for tau, rec in sorted(sweep.items()):
        fe = rec.get("concat", {}).get("false_endpoint_rate", 1.0)
        p50_ls = rec.get("solo_ls", {}).get("lat_ms_p50", float("inf"))
        if fe <= 0.05 and 0 <= p50_ls <= 400:
            best_tau = tau
            break
    logger.info("Gate-passing threshold: %s", best_tau)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n_params_M": n_params,
        "train_steps": args.steps,
        "lr": args.lr,
        "end_pos_weight": args.end_pos_weight,
        "n_train": len(train_idx),
        "n_eval": len(eval_idx),
        "sweep": {str(k): v for k, v in sweep.items()},
        "gate_passing_tau": best_tau,
    }, indent=2))

    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "hidden": args.hidden}, args.ckpt)
    logger.info("Saved head to %s, results to %s", args.ckpt, args.out)


if __name__ == "__main__":
    main()
