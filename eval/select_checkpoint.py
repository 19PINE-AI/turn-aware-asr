"""Evaluate many endpoint snapshots on the dictation/spelled probes with a
single base-model load, to select a unified checkpoint (research/76).

For each snap_step*.pt (or an explicit --checkpoints list) it swaps only the
trainable LoRA/marker weights and runs both probes, reporting the axes that
matter for the unified goal:
  digit premature/seq, digit final recall, spelled name/email +ctx exact,
  wrong-profile intrusion.
Then, offline, run the replay regression only on the shortlisted winners.

Usage:
    .venv/bin/python eval/select_checkpoint.py \
        --glob 'checkpoints/semantic_endpoint_v12_es/snap_step*.pt' \
        --out research/76-v12-probe-sweep.json
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.streaming_replay_eval import StreamDecoder  # noqa: E402
import eval.dictation_probe_eval as dpe               # noqa: E402

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="checkpoints/semantic_endpoint_v12_es/snap_step*.pt")
    p.add_argument("--checkpoints", nargs="*", default=None,
                   help="Explicit checkpoint paths (overrides --glob).")
    p.add_argument("--probes", default="data/probes")
    p.add_argument("--out", required=True)
    p.add_argument("--chunk-s", type=float, default=0.5)
    p.add_argument("--gate-rms", type=float, default=1e-3)
    args = p.parse_args()

    paths = args.checkpoints or sorted(
        glob.glob(args.glob), key=lambda s: int(Path(s).stem.split("step")[-1]))
    if not paths:
        raise SystemExit(f"no checkpoints match {args.glob}")
    logger.info("Evaluating %d checkpoints", len(paths))

    # Base model loaded ONCE; per-checkpoint we only load the trainable dict.
    from src.train_semantic_endpoint import (
        load_base_model, extend_tokenizer_and_model,
        apply_lora, freeze_except_lora_and_new_rows,
    )
    model, tokenizer, processor = load_base_model()
    model = model.cuda().bfloat16()
    new_ids, _, _ = extend_tokenizer_and_model(model, tokenizer, processor)
    # infer LoRA rank from the first checkpoint (supports r16 and r32 pools)
    _probe = torch.load(paths[0], weights_only=False, map_location="cpu")["trainable"]
    _rank = next(v.shape[0] for k, v in _probe.items() if "lora_A" in k)
    logger.info("LoRA rank inferred from checkpoint: %d", _rank)
    apply_lora(model.thinker, rank=_rank, alpha=2 * _rank)
    freeze_except_lora_and_new_rows(model, new_ids)

    digit_items = json.loads(Path(args.probes + "/digit_probe.json").read_text())
    spell_items = json.loads(Path(args.probes + "/spelled_probe.json").read_text())

    rows = []
    for path in paths:
        step = int(Path(path).stem.split("step")[-1])
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        model.load_state_dict(ckpt["trainable"], strict=False)
        model.eval()

        dec = StreamDecoder(model, processor, tokenizer, committed=True)
        import numpy as np
        per = [dpe.run_digit_item(dec, it, args.chunk_s, args.gate_rms, 0)
               for it in digit_items]
        lat = [r["final_latency_s"] for r in per if r["final_latency_s"] is not None]
        digit = {
            "premature_per_seq": round(float(np.mean([r["n_premature"] for r in per])), 3),
            "final_recall": round(float(np.mean([r["final_recall"] for r in per])), 3),
            "digit_acc": round(float(np.mean([r["digit_acc"] for r in per])), 3),
            "final_lat_p50": round(float(np.median(lat)), 3) if lat else None,
        }
        spell = dpe.run_probe_b(model, tokenizer, processor, spell_items)["summary"]
        row = {"step": step, "path": path, "digit": digit,
               "name_profile": spell["name_profile"], "email_profile": spell["email_profile"],
               "name_none": spell["name_none"], "email_none": spell["email_none"],
               "intrusion": spell["intrusion_rate"]}
        rows.append(row)
        logger.info("step %5d  prem=%.2f rec=%.2f | name+=%.2f email+=%.2f | intrusion=%.3f",
                    step, digit["premature_per_seq"], digit["final_recall"],
                    spell["name_profile"], spell["email_profile"], spell["intrusion_rate"])
        Path(args.out).write_text(json.dumps(rows, indent=1))

    # Rank by a simple unified objective: pass gates then minimize a cost.
    def ok(r):
        return (r["digit"]["premature_per_seq"] <= 0.6 and
                r["email_profile"] >= 0.85 and r["intrusion"] <= 0.06)
    winners = [r for r in rows if ok(r)]
    logger.info("=== %d checkpoints pass the unified gate (prem<=0.6, email+>=0.85, intrusion<=0.06) ===",
                len(winners))
    for r in winners:
        logger.info("  step %5d  prem=%.2f email+=%.2f intrusion=%.3f",
                    r["step"], r["digit"]["premature_per_seq"],
                    r["email_profile"], r["intrusion"])
    Path(args.out).write_text(json.dumps(
        {"rows": rows, "gate": "prem<=0.6,email+>=0.85,intrusion<=0.06",
         "winners": [r["step"] for r in winners]}, indent=1))
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
