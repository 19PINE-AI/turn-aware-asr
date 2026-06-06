"""Measure streaming-mode end-of-turn latency on the fine-tuned model.

Protocol (transformers backend, since qwen-asr's streaming is vLLM-only):
  - Feed audio in CHUNK_S = 0.5 s windows from t=0.
  - After audio truly ends (at AUDIO_END_S), continue feeding silence
    chunks for up to MAX_TAIL_S more seconds.
  - At each chunk boundary, re-run model.generate on the accumulated
    audio (i.e. simulate the "re-feed everything" protocol of qwen-asr).
  - Record the first chunk boundary at which <END_SPEECH> appears in
    the decoded output.

Latency = (chunk_end_time - audio_true_end) — i.e. how long after the
user stops speaking does our system signal "they're done?"

Note: this does NOT use a real streaming KV cache. Each chunk does a
fresh forward + generate, so we measure UPPER-BOUND wall clock latency
that the user could plausibly tolerate. A real KV-cached streaming
implementation would only need to encode the *new* chunk per step.

Usage:
    python -m eval.streaming_latency \\
        --checkpoint checkpoints/semantic_endpoint_v3_long/step6000.pt \\
        --data data/semantic_endpoint_v2_simple/data.pt \\
        --n-examples 30
"""

from __future__ import annotations
import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

END_TOK = "<END_SPEECH>"
EAGER_TOK = "<EAGER_END_SPEECH>"
SR = 16000


def _generate(model, processor, tokenizer, audio_so_far: np.ndarray,
              max_new: int = 96) -> str:
    """Greedy-generate over `audio_so_far` and return decoded text."""
    device = next(model.parameters()).device
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio"}]},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], audio=[audio_so_far], return_tensors="pt", padding=True)
    inputs = {
        k: v.to(device).bfloat16() if torch.is_floating_point(v) else v.to(device)
        for k, v in inputs.items()
    }
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new)
    gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    text = re.sub(r"^language\s+\S+\s*<asr_text>", "", text)
    text = text.replace("<|im_end|>", "").strip()
    return text


def measure_latency_on_example(model, processor, tokenizer, audio: np.ndarray,
                                 audio_end_s: float,
                                 chunk_s: float = 0.5,
                                 max_tail_s: float = 4.0) -> dict:
    """Return latency info for one example.

    audio: the full audio to feed (just the utterance; we pad silence here)
    audio_end_s: t at which the user effectively stops speaking
    """
    n_audio = len(audio)
    sr = SR
    pad = np.zeros(int(max_tail_s * sr), dtype=np.float32)
    full = np.concatenate([audio, pad])

    chunk_samples = int(chunk_s * sr)
    n_chunks_total = len(full) // chunk_samples

    first_end_chunk_idx: int | None = None
    first_end_t: float | None = None
    last_text = ""
    inference_times_ms: list[float] = []

    for k in range(1, n_chunks_total + 1):
        cur_end_samples = k * chunk_samples
        cur_end_s = cur_end_samples / sr
        audio_so_far = full[:cur_end_samples].copy()

        t0 = time.perf_counter()
        text = _generate(model, processor, tokenizer, audio_so_far)
        dt = (time.perf_counter() - t0) * 1000
        inference_times_ms.append(dt)
        last_text = text

        if END_TOK in text:
            first_end_chunk_idx = k
            first_end_t = cur_end_s
            break

    result = {
        "audio_end_s": audio_end_s,
        "first_end_chunk_idx": first_end_chunk_idx,
        "first_end_t": first_end_t,
        "latency_s": (first_end_t - audio_end_s) if first_end_t is not None else None,
        "n_chunks_run": len(inference_times_ms),
        "median_chunk_ms": float(np.median(inference_times_ms)) if inference_times_ms else None,
        "final_text": last_text,
    }
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/semantic_endpoint_v2_simple/data.pt")
    p.add_argument("--n-examples", type=int, default=30)
    p.add_argument("--chunk-s", type=float, default=0.5)
    p.add_argument("--max-tail-s", type=float, default=4.0)
    p.add_argument("--schemas", nargs="+", default=["single", "disfluency"],
                    help="Which schemas to measure on (default skips double — it has multiple endpoints)")
    p.add_argument("--out", default="research/27-streaming-latency.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logger.info("Loading model + checkpoint…")
    from src.train_semantic_endpoint import (
        load_base_model, extend_tokenizer_and_model,
        apply_lora, freeze_except_lora_and_new_rows,
    )
    model, tokenizer, processor = load_base_model()
    model = model.cuda().bfloat16()
    new_ids, eager_id, end_id = extend_tokenizer_and_model(model, tokenizer, processor)
    apply_lora(model.thinker, rank=16, alpha=32)
    freeze_except_lora_and_new_rows(model, new_ids)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["trainable"], strict=False)
    model.eval()
    logger.info("Loaded step %s", ckpt.get("step", -1))

    logger.info("Loading data %s…", args.data)
    examples = torch.load(args.data, weights_only=False)

    import random as _r
    rng = _r.Random(args.seed)
    # Sample N examples across the allowed schemas
    pool = [e for e in examples if e["schema"] in args.schemas]
    rng.shuffle(pool)
    examples = pool[: args.n_examples]

    results: list[dict] = []
    for i, e in enumerate(tqdm(examples, desc="streaming-lat")):
        audio = np.asarray(e["audio"], dtype=np.float32)
        audio_end_s = len(audio) / SR
        r = measure_latency_on_example(
            model, processor, tokenizer, audio, audio_end_s,
            chunk_s=args.chunk_s, max_tail_s=args.max_tail_s,
        )
        r["schema"] = e["schema"]
        results.append(r)
        if i < 3:
            logger.info("[%s] audio_end=%.2fs  detected_at=%s  latency=%s",
                         e["schema"], audio_end_s,
                         f'{r["first_end_t"]:.2f}s' if r["first_end_t"] is not None else 'NEVER',
                         f'{r["latency_s"]:.2f}s' if r["latency_s"] is not None else 'NEVER')

    # Aggregate
    detected = [r for r in results if r["latency_s"] is not None]
    summary = {
        "n_examples": len(results),
        "n_detected": len(detected),
        "detection_rate": len(detected) / max(1, len(results)),
        "chunk_s": args.chunk_s,
    }
    if detected:
        lat = sorted([r["latency_s"] for r in detected])
        chunks = [r["median_chunk_ms"] for r in detected]
        summary.update({
            "latency_s_p50": lat[len(lat) // 2],
            "latency_s_p95": lat[int(0.95 * (len(lat) - 1))],
            "latency_s_mean": float(np.mean(lat)),
            "latency_s_min": lat[0],
            "latency_s_max": lat[-1],
            "median_chunk_inference_ms": float(np.median(chunks)),
        })

    logger.info("== STREAMING LATENCY SUMMARY ==")
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info("  %-36s %.3f", k, v)
        else:
            logger.info("  %-36s %s", k, v)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "summary": summary,
        "per_example": results,
    }, indent=2))
    logger.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
