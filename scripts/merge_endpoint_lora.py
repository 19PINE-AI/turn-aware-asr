"""Merge an endpoint-LoRA checkpoint into a standard Qwen3-ASR HF checkpoint.

The two marker tokens (<EAGER_END_SPEECH>, <END_SPEECH>) occupy RESERVED
vocab slots (151705/151706 < 151936), so the merged model is a plain
Qwen3-ASR checkpoint — servable by vLLM (Qwen3ASRForConditionalGeneration)
with no adapter plumbing; markers decode as ordinary tokens.
(research/60-metronome-integration-scope.md, gate E1.)

Steps:
  1. Load base Qwen3-ASR-0.6B via qwen_asr (same as training).
  2. Re-init the two reserved rows, apply LoRA, load ckpt['trainable']
     (exactly the eval-time loading path, so parity is by construction).
  3. peft merge_and_unload() -> LoRA deltas folded into q/k/v/o_proj.
  4. Save model + processor + tokenizer (with the two added tokens) to
     --out-dir as a standard HF directory.

Usage:
    python -m scripts.merge_endpoint_lora \
        --checkpoint checkpoints/semantic_endpoint_v5_es/best.pt \
        --out-dir checkpoints/merged/qwen3-asr-0.6b-endpoint-v5
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    from src.train_semantic_endpoint import (
        load_base_model, extend_tokenizer_and_model,
        apply_lora, freeze_except_lora_and_new_rows,
    )
    model, tokenizer, processor = load_base_model()
    model = model.bfloat16()
    new_ids, eager_id, end_id = extend_tokenizer_and_model(model, tokenizer, processor)
    apply_lora(model.thinker, rank=16, alpha=32)
    freeze_except_lora_and_new_rows(model, new_ids)

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["trainable"], strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    logger.info("Loaded %s (step %s): %d trainable tensors",
                 args.checkpoint, ckpt.get("step"), len(ckpt["trainable"]))
    assert ckpt.get("eager_id", eager_id) == eager_id
    assert ckpt.get("end_id", end_id) == end_id

    # apply_lora injects peft LoraLayer modules in place (no PeftModel
    # wrapper survives — see train_semantic_endpoint.apply_lora). Merge each
    # layer's delta into its base weight, then swap the clean base layer back.
    from peft.tuners.lora import LoraLayer
    lora_layers = [(n, m) for n, m in model.thinker.model.named_modules()
                   if isinstance(m, LoraLayer)]
    assert lora_layers, "no LoRA layers found — wrong checkpoint or load path"
    for name, module in lora_layers:
        module.merge(safe_merge=True)
        parent = model.thinker.model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], module.base_layer)
    logger.info("Merged %d LoRA layers into base weights", len(lora_layers))

    # Sanity: marker rows are non-degenerate after the merge
    emb = model.thinker.model.embed_tokens.weight
    for tid, name in [(eager_id, "EAGER"), (end_id, "END")]:
        norm = emb[tid].float().norm().item()
        logger.info("embed row %d (%s) L2 = %.4f", tid, name, norm)
        assert norm > 1e-3, "marker embedding row looks uninitialized"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Qwen3-ASR ships a generation_config that pairs sampling params
    # (temperature/top_p/top_k) with do_sample=False; transformers' strict
    # validation rejects that on save_pretrained. Null the sampling fields so
    # the greedy (do_sample=False) config is valid; fall back to dropping the
    # generation_config entirely if that isn't enough (vLLM sets sampling per
    # request anyway; eos/pad come from the model config + tokenizer).
    def _sanitize_gc(m):
        gc = getattr(m, "generation_config", None)
        if gc is None:
            return
        for f in ("temperature", "top_p", "top_k", "typical_p", "top_a"):
            if getattr(gc, f, None) is not None:
                setattr(gc, f, None)
        gc.do_sample = False
    for mod in (model, getattr(model, "thinker", None)):
        if mod is not None:
            _sanitize_gc(mod)
    try:
        model.save_pretrained(out, safe_serialization=True)
    except ValueError as e:
        if "GenerationConfig" not in str(e):
            raise
        logger.warning("generation_config still invalid (%s); dropping it for save", e)
        model.generation_config = None
        if getattr(model, "thinker", None) is not None:
            model.thinker.generation_config = None
        model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    processor.save_pretrained(out)
    logger.info("Saved merged checkpoint to %s", out)
    logger.info("Parity check: transcribe a marker-bearing clip via "
                "transformers from this dir and compare against the LoRA path.")


if __name__ == "__main__":
    main()
