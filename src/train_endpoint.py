"""Train the 0-delay BCE endpoint head on AuT hidden states.

Loads the prepared data from `eval/endpoint_data_prep.py`. The head has
~0.3 M params (vs Qwen3-ASR's 938 M); trains in minutes.

Metrics tracked:
  - per-frame BCE loss (eager + end)
  - frame-level precision/recall on the "end" target
  - endpoint latency: P50/P95 of (first-fire-time - true_end_time) per utt
  - false-endpoint rate on synthetic concat pairs (fire during internal gap)

This is the project's actual architectural contribution: Qwen3-ASR ships
no VAD; this head adds one.

Usage:
  python -m src.train_endpoint --data data/endpoint/data.pt --steps 500
"""

from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.endpoint_head import EndpointHead

logger = logging.getLogger(__name__)

FRAME_S = 0.08  # 80 ms per frame at 12.5 Hz


def evaluate(model: EndpointHead, data: dict, threshold: float = 0.5) -> dict:
    model.eval()
    device = next(model.parameters()).device
    latencies = []
    false_endpoints = 0  # fires during internal gap (concat only)
    n_concat = 0
    frame_tp = frame_fp = frame_fn = frame_tn = 0

    with torch.no_grad():
        for i in range(len(data["aut_frames"])):
            x = data["aut_frames"][i].to(device).bfloat16().unsqueeze(0)
            T = x.shape[1]
            T_audio = data["audio_len_frames"][i]
            end_targ = data["end_targets"][i].to(device)
            kind = data["kind"][i]

            eager_log, end_log = model(x)
            p_end = torch.sigmoid(end_log).squeeze(0).squeeze(-1)  # (T,)

            # First fire = first frame where p_end > threshold
            fires = (p_end > threshold).nonzero(as_tuple=True)[0]
            true_end_frame = T_audio - 1  # 0-indexed
            true_end_s = true_end_frame * FRAME_S

            if kind == "concat":
                n_concat += 1
                # The true endpoint is only at T_audio - 1. Any fire well before
                # is a false endpoint (during the internal gap or in utt_A).
                # We'll count fires at frames < T_audio - 5 (>=0.4s early) as false.
                if fires.numel() > 0:
                    first_fire = fires[0].item()
                    if first_fire < T_audio - 5:
                        false_endpoints += 1

            if kind == "solo":
                if fires.numel() > 0:
                    first_fire = fires[0].item()
                    latency_s = first_fire * FRAME_S - true_end_s
                    latencies.append(latency_s * 1000)  # ms

            # Frame-level metrics
            pred = (p_end > threshold).float()
            tp = ((pred == 1) & (end_targ == 1)).sum().item()
            fp = ((pred == 1) & (end_targ == 0)).sum().item()
            fn = ((pred == 0) & (end_targ == 1)).sum().item()
            tn = ((pred == 0) & (end_targ == 0)).sum().item()
            frame_tp += tp; frame_fp += fp; frame_fn += fn; frame_tn += tn

    model.train()
    out = {
        "frame_precision": frame_tp / max(1, frame_tp + frame_fp),
        "frame_recall": frame_tp / max(1, frame_tp + frame_fn),
        "n_solo": len(latencies),
    }
    if latencies:
        lat = torch.tensor(latencies)
        out["latency_ms_p50"] = float(lat.quantile(0.5))
        out["latency_ms_p95"] = float(lat.quantile(0.95))
        out["latency_ms_mean"] = float(lat.mean())
    if n_concat > 0:
        out["false_endpoint_rate"] = false_endpoints / n_concat
        out["n_concat"] = n_concat
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/endpoint/data.pt")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--bsz", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--out", default="research/17-endpoint-results.json")
    p.add_argument("--ckpt", default="checkpoints/endpoint_head.pt")
    args = p.parse_args()

    logger.info("Loading data %s…", args.data)
    raw = torch.load(args.data, weights_only=False)
    n = len(raw["aut_frames"])
    logger.info("Loaded %d examples (solo=%d, concat=%d)",
                 n,
                 sum(1 for k in raw["kind"] if k == "solo"),
                 sum(1 for k in raw["kind"] if k == "concat"))

    # 80/20 split
    rng = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=rng).tolist()
    split = int(0.8 * n)
    train_idx, eval_idx = perm[:split], perm[split:]
    train_data = {k: [raw[k][i] for i in train_idx] for k in raw}
    eval_data = {k: [raw[k][i] for i in eval_idx] for k in raw}
    logger.info("Train %d / eval %d", len(train_idx), len(eval_idx))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = EndpointHead(d_audio=1024, hidden=args.hidden).to(device).bfloat16()
    n_params = sum(p.numel() for p in head.parameters()) / 1e6
    logger.info("EndpointHead params: %.2f M", n_params)

    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)

    # Simple per-utt training. Pad to max-len batch.
    indices = train_idx[:]
    cursor = 0
    rng2 = torch.Generator().manual_seed(0)

    for step in range(args.steps):
        if cursor + args.bsz > len(indices):
            indices = [indices[j] for j in torch.randperm(len(indices), generator=rng2).tolist()]
            cursor = 0
        batch_idx = indices[cursor : cursor + args.bsz]
        cursor += args.bsz

        max_T = max(raw["aut_frames"][i].shape[0] for i in batch_idx)
        B = len(batch_idx)
        X = torch.zeros(B, max_T, 1024, dtype=torch.bfloat16, device=device)
        end_t = torch.zeros(B, max_T, dtype=torch.float32, device=device)
        eager_t = torch.zeros(B, max_T, dtype=torch.float32, device=device)
        mask = torch.zeros(B, max_T, dtype=torch.bool, device=device)
        for j, i in enumerate(batch_idx):
            x = raw["aut_frames"][i].to(device).bfloat16()
            T = x.shape[0]
            X[j, :T] = x
            end_t[j, :T] = raw["end_targets"][i].to(device)
            eager_t[j, :T] = raw["eager_targets"][i].to(device)
            mask[j, :T] = True

        eager_log, end_log = head(X)
        loss = EndpointHead.loss(eager_log, end_log, eager_t, end_t, mask=mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optim.step()
        optim.zero_grad()

        if step % 50 == 0 or step == args.steps - 1:
            logger.info("step %4d  loss %.4f", step, loss.item())

    # Final eval on held-out
    logger.info("Evaluating on held-out…")
    metrics = evaluate(head, eval_data)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %-32s %.3f", k, v)
        else:
            logger.info("  %-32s %s", k, v)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n_params_M": n_params,
        "train_steps": args.steps,
        "lr": args.lr,
        "n_train": len(train_idx),
        "n_eval": len(eval_idx),
        "metrics": metrics,
    }, indent=2))

    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), args.ckpt)
    logger.info("Saved head to %s, results to %s", args.ckpt, args.out)


if __name__ == "__main__":
    main()
