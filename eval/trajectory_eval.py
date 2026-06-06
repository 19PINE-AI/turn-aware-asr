"""Evaluate a sequence of semantic-endpoint checkpoints to trace accuracy
trajectory across training steps.

For each checkpoint, runs the standard 90-utterance balanced eval and
collects the key metrics. Saves a single JSON table that plots
nicely.

Usage:
    python -m eval.trajectory_eval \\
        --ckpt-dir checkpoints/semantic_endpoint_v3_long \\
        --steps 3000 6000 9000 12000 15000 18000
"""

from __future__ import annotations
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_eval(ckpt: str, data: str, max_examples: int, out: str) -> dict:
    cmd = [
        sys.executable, "-u", "-m", "eval.semantic_endpoint_eval",
        "--checkpoint", ckpt,
        "--data", data,
        "--max-examples", str(max_examples),
        "--out", out,
        "--seed", "0",
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Eval failed for %s:\n%s", ckpt, result.stderr[-2000:])
        return {}
    return json.loads(Path(out).read_text())["summary"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="checkpoints/semantic_endpoint_v3_long")
    p.add_argument("--steps", nargs="+", type=int,
                    default=[3000, 6000, 9000, 12000, 15000, 18000])
    p.add_argument("--data", default="data/semantic_endpoint_v2_simple/data.pt")
    p.add_argument("--max-examples", type=int, default=90)
    p.add_argument("--out", default="research/23-trajectory-eval.json")
    args = p.parse_args()

    rows = []
    ckpt_dir = Path(args.ckpt_dir)
    for step in args.steps:
        ckpt_path = ckpt_dir / f"step{step}.pt"
        if not ckpt_path.exists():
            logger.warning("Checkpoint %s missing — skipping", ckpt_path)
            continue
        out_path = f"/tmp/trajectory_step{step}.json"
        summary = run_eval(str(ckpt_path), args.data, args.max_examples, out_path)
        if not summary:
            continue
        rows.append({"step": step, **summary})
        # Print intermediate progress so the user sees the trajectory as it builds
        logger.info("step %5d  single=%.3f  double=%.3f  disfl-correct=%.3f  "
                     "disfl-overfire=%.3f  wer-s=%.2f  wer-d=%.2f  wer-f=%.2f",
                     step,
                     summary.get("end_recall_single", 0),
                     summary.get("end_recall_double", 0),
                     summary.get("end_correct_disfluency", 0),
                     summary.get("end_oversfire_disfluency", 0),
                     summary.get("wer_single", 0),
                     summary.get("wer_double", 0),
                     summary.get("wer_disfluency", 0))

    # Save the full trajectory
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rows": rows}, indent=2))

    # Pretty trajectory table
    print()
    print("=" * 88)
    print(f"{'step':>6} {'single':>8} {'double':>8} {'disfl-✓':>8} {'disfl-X':>8} "
          f"{'WER-s':>7} {'WER-d':>7} {'WER-f':>7}")
    print("=" * 88)
    for r in rows:
        print(f"{r['step']:>6d} "
              f"{r.get('end_recall_single', 0)*100:>7.1f}% "
              f"{r.get('end_recall_double', 0)*100:>7.1f}% "
              f"{r.get('end_correct_disfluency', 0)*100:>7.1f}% "
              f"{r.get('end_oversfire_disfluency', 0)*100:>7.1f}% "
              f"{r.get('wer_single', 0):>6.2f}% "
              f"{r.get('wer_double', 0):>6.2f}% "
              f"{r.get('wer_disfluency', 0):>6.2f}%")
    print("=" * 88)
    logger.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
