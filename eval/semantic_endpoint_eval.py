"""Evaluate the semantic-endpoint fine-tuned model.

For each held-out example, run inference (greedy decode) and compute:
  - Did the model emit <END_SPEECH>? (recall)
  - For DOUBLE schema: did it emit TWO <END_SPEECH>?
  - For DISFLUENCY schema: did it emit only ONE <END_SPEECH> at the actual end?
                          (false-endpoint = model emitted >1 <END_SPEECH>)
  - WER on the transcript (compared against ref without the markers)

This validates the project's core claim: the LM can predict semantic
endpoints inline with transcription, without regressing WER.

Usage:
    python -m eval.semantic_endpoint_eval \\
        --checkpoint checkpoints/semantic_endpoint/step1000.pt \\
        --data data/semantic_endpoint/data.pt \\
        --max-examples 100
"""

from __future__ import annotations
import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from eval.metrics import wer
from src.train_semantic_endpoint import (
    EAGER_TOK, END_TOK,
    load_base_model, extend_tokenizer_and_model,
)

logger = logging.getLogger(__name__)


def strip_markers(text: str) -> str:
    return text.replace(EAGER_TOK, "").replace(END_TOK, "").strip()


@torch.no_grad()
def transcribe_with_markers(model, processor, tokenizer, audio_np, max_new=256) -> str:
    """Generate a transcription using the official transcribe API path."""
    device = next(model.parameters()).device
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio"}]},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], audio=[audio_np], return_tensors="pt", padding=True)
    inputs = {
        k: v.to(device).bfloat16() if torch.is_floating_point(v) else v.to(device)
        for k, v in inputs.items()
    }
    out = model.generate(**inputs, max_new_tokens=max_new)
    gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    # Strip the "language English<asr_text>" prefix
    text = re.sub(r"^language\s+\S+\s*<asr_text>", "", text)
    text = text.replace("<|im_end|>", "").strip()
    return text


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/semantic_endpoint/data.pt")
    p.add_argument("--max-examples", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="research/19-semantic-endpoint-eval.json")
    args = p.parse_args()

    logger.info("Loading data %s…", args.data)
    examples = torch.load(args.data, weights_only=False)

    # Use a deterministic held-out split (last 20% by index)
    n = len(examples)
    split = int(0.8 * n)
    eval_examples = examples[split:][: args.max_examples]
    logger.info("Eval %d examples", len(eval_examples))

    logger.info("Loading model + checkpoint…")
    model, tokenizer, processor = load_base_model()
    model = model.cuda().bfloat16()
    new_ids, eager_id, end_id = extend_tokenizer_and_model(model, tokenizer, processor)

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    state = ckpt["trainable"]
    # Load LoRA + embed/lm_head params
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected[:3])
    logger.info("Loaded %s (step %d, %d tensors)",
                 args.checkpoint, ckpt.get("step", -1), len(state))

    # Eval loop
    per = []
    for i, e in enumerate(tqdm(eval_examples, desc="eval")):
        hyp = transcribe_with_markers(model, processor, tokenizer, e["audio"])
        ref = e["text"]
        ref_stripped = strip_markers(ref)
        hyp_stripped = strip_markers(hyp)

        # Marker counts
        n_end_ref = ref.count(END_TOK)
        n_end_hyp = hyp.count(END_TOK)
        n_eager_ref = ref.count(EAGER_TOK)
        n_eager_hyp = hyp.count(EAGER_TOK)

        r = {
            "schema": e["schema"],
            "ref": ref,
            "hyp": hyp,
            "n_end_ref": n_end_ref,
            "n_end_hyp": n_end_hyp,
            "n_eager_ref": n_eager_ref,
            "n_eager_hyp": n_eager_hyp,
            "wer_stripped": wer(hyp_stripped, ref_stripped) * 100,
        }
        per.append(r)
        if i < 3:
            logger.info("REF[%d] (%s): %s", i, e["schema"], ref[:120])
            logger.info("HYP[%d]: %s", i, hyp[:120])

    # Aggregate
    def by_schema(schema, field, op="mean"):
        vals = [r[field] for r in per if r["schema"] == schema]
        if not vals:
            return None
        if op == "mean":
            return float(np.mean(vals))
        if op == "sum":
            return float(np.sum(vals))

    summary = {
        "n_examples": len(per),
        "n_single": sum(1 for r in per if r["schema"] == "single"),
        "n_double": sum(1 for r in per if r["schema"] == "double"),
        "n_disfluency": sum(1 for r in per if r["schema"] == "disfluency"),

        # End-marker emission accuracy
        "end_recall_single": (
            sum(1 for r in per if r["schema"] == "single" and r["n_end_hyp"] >= 1) /
            max(1, sum(1 for r in per if r["schema"] == "single"))
        ),
        "end_recall_double": (
            sum(1 for r in per if r["schema"] == "double" and r["n_end_hyp"] >= 2) /
            max(1, sum(1 for r in per if r["schema"] == "double"))
        ),
        "end_correct_disfluency": (
            sum(1 for r in per if r["schema"] == "disfluency" and r["n_end_hyp"] == 1) /
            max(1, sum(1 for r in per if r["schema"] == "disfluency"))
        ),
        "end_oversfire_disfluency": (
            sum(1 for r in per if r["schema"] == "disfluency" and r["n_end_hyp"] > 1) /
            max(1, sum(1 for r in per if r["schema"] == "disfluency"))
        ),

        # WER (transcript correctness, ignoring markers)
        "wer_single": by_schema("single", "wer_stripped"),
        "wer_double": by_schema("double", "wer_stripped"),
        "wer_disfluency": by_schema("disfluency", "wer_stripped"),
    }

    logger.info("== SUMMARY ==")
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info("  %-32s %.3f", k, v)
        else:
            logger.info("  %-32s %s", k, v)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "summary": summary,
        "per_example": per,
    }, indent=2))
    logger.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
