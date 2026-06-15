"""Re-evaluate trained EndpointHead with a K-consecutive-frames trigger.

Original policy: fire on first frame where p_end > τ. This fires too
fast on internal gaps (1s of silence saturates p_end).

K-consecutive policy: fire on first frame where p_end > τ has held
for K consecutive frames. This implicitly waits K * 80 ms before
committing — distinguishes "natural pause within sentence" from
"end of speech" by requiring sustained silence.

We sweep (τ, K) jointly and find any point hitting:
  - false_endpoint_rate ≤ 5 % on concat
  - solo_ls latency P50 ∈ [0, +400 ms]

Usage:
    python -m src.endpoint_eval_policy \\
        --data data/endpoint_v2/data.pt \\
        --ckpt checkpoints/endpoint_head_v2.pt
"""

from __future__ import annotations
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch

from src.endpoint_head import EndpointHead

logger = logging.getLogger(__name__)
FRAME_S = 0.08


def k_consecutive_first(p: torch.Tensor, tau: float, k: int) -> int | None:
    """First index t such that p[t-k+1 : t+1] all > tau. Returns None if never."""
    above = (p > tau).float()
    if k <= 1:
        nz = above.nonzero(as_tuple=True)[0]
        return int(nz[0].item()) if nz.numel() else None
    # Cumulative count of consecutive trues ending at each idx
    run = torch.zeros_like(above)
    for i in range(above.shape[0]):
        run[i] = above[i] * (run[i - 1] + 1 if i > 0 else 1)
    nz = (run >= k).nonzero(as_tuple=True)[0]
    return int(nz[0].item()) if nz.numel() else None


def eval_policy(head: EndpointHead, data: dict, tau: float, k: int,
                 gap_buffer_frames: int = 1) -> dict:
    head.eval()
    device = next(head.parameters()).device
    by_kind = defaultdict(lambda: {"lat_ms": [], "n_fire": 0, "n_total": 0,
                                     "n_false_endpoint": 0})
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
                if fire_t < last_we - gap_buffer_frames:
                    by_kind[kind]["n_false_endpoint"] += 1
                else:
                    lat_ms = (fire_t - last_we) * FRAME_S * 1000
                    by_kind[kind]["lat_ms"].append(lat_ms)

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
            rec["false_endpoint_rate"] = m["n_false_endpoint"] / max(1, m["n_total"])
            rec["n_false_endpoint"] = m["n_false_endpoint"]
        out[kind] = rec
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/endpoint_v2/data.pt")
    p.add_argument("--ckpt", default="checkpoints/endpoint_head_v2.pt")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="research/52-endpoint-policy-sweep.json")
    args = p.parse_args()

    logger.info("Loading %s…", args.ckpt)
    state = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    head = EndpointHead(d_audio=1024, hidden=args.hidden).cuda().bfloat16()
    head.load_state_dict(state["state_dict"] if "state_dict" in state else state)

    logger.info("Loading data…")
    raw = torch.load(args.data, weights_only=False)
    n = len(raw["aut_frames"])
    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=rng).tolist()
    split = int(0.85 * n)
    eval_idx = perm[split:]
    eval_data = {k: [raw[k][i] for i in eval_idx] for k in raw}
    logger.info("eval %d", len(eval_idx))

    grid = []
    best = None
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for k in [1, 2, 3, 4, 5, 6, 8, 10]:
            m = eval_policy(head, eval_data, tau, k)
            mc = m.get("concat", {})
            mls = m.get("solo_ls", {})
            mam = m.get("solo_ami", {})
            row = {
                "tau": tau, "k": k,
                "fe_rate": mc.get("false_endpoint_rate", float("nan")),
                "ls_p50": mls.get("lat_ms_p50", float("nan")),
                "ls_p95": mls.get("lat_ms_p95", float("nan")),
                "ls_detect": mls.get("detect_rate", 0),
                "ami_p50": mam.get("lat_ms_p50", float("nan")),
                "ami_p95": mam.get("lat_ms_p95", float("nan")),
                "ami_detect": mam.get("detect_rate", 0),
            }
            grid.append(row)
            fe = row["fe_rate"]
            p50 = row["ls_p50"]
            if fe <= 0.05 and 0 <= p50 <= 400 and row["ls_detect"] >= 0.9:
                row["meets_gate"] = True
                if best is None or row["ls_p50"] < best["ls_p50"]:
                    best = row
            logger.info("τ=%.2f k=%-2d  FE=%5.1f%%  LS P50=%5.0fms P95=%5.0fms det=%.2f  AMI P50=%5.0fms",
                         tau, k, 100*fe, p50, row["ls_p95"], row["ls_detect"], row["ami_p50"])

    logger.info("Best gate-passing point: %s", best)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"grid": grid, "best": best}, indent=2))
    logger.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
