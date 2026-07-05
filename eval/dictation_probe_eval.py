"""Run the dictation probes (research/74-75) against an endpoint checkpoint.

Probe A (digit dictation, streaming): replays each 10-digit sequence through
the same committed-prefix StreamDecoder + energy gate + fire logic as the
replay benchmark. Scores:
  - premature fires: fires at t < last_speech_end + 0.25 (inter-group pauses)
  - final recall: first fire in (last_speech_end, last_speech_end + 1.75]
  - final latency: that fire's time - last_speech_end
  - digit accuracy of the concatenated flushed transcript
Analytic silence-timeout baselines (X = 0.5 / 1.0 s) are computed from the
known pause layout: a timeout fires in every pause >= X and at last_end + X.

Probe B (spelled names/emails, offline single-shot): decodes each item under
three system prompts — none / profile CTX / distractor CTX — and scores
entity exact-match after spoken-form normalization ("K, O, W" -> kow,
"dot"->".", "at"->"@").

Usage:
    .venv/bin/python eval/dictation_probe_eval.py \
        --checkpoint checkpoints/semantic_endpoint_v9_es/best.pt \
        --out research/74-dictation-probe-v9.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.streaming_replay_eval import StreamDecoder, clean, END_TOK, EAGER_TOK, SR  # noqa: E402

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ loading

def load_model(checkpoint: str):
    from src.train_semantic_endpoint import (
        load_base_model, extend_tokenizer_and_model,
        apply_lora, freeze_except_lora_and_new_rows,
    )
    model, tokenizer, processor = load_base_model()
    model = model.cuda().bfloat16()
    new_ids, _, _ = extend_tokenizer_and_model(model, tokenizer, processor)
    apply_lora(model.thinker, rank=16, alpha=32)
    freeze_except_lora_and_new_rows(model, new_ids)
    ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["trainable"], strict=False)
    model.eval()
    logger.info("Loaded %s (step %s)", checkpoint, ckpt.get("step"))
    return model, tokenizer, processor


# ------------------------------------------------------------------ Probe A

def run_digit_item(dec: StreamDecoder, item: dict, chunk_s: float,
                   gate_rms: float, confirm_chunks: int = 0) -> dict:
    audio, sr = sf.read(item["wav"], dtype="float32")
    assert sr == SR
    n_chunks = int(np.ceil(len(audio) / (chunk_s * SR)))
    fires: list[float] = []
    flushed: list[str] = []
    prev_marker_count = 0
    pending_fire = None
    dec.reset_segment()
    for k in range(n_chunks):
        chunk = audio[int(k * chunk_s * SR): int((k + 1) * chunk_s * SR)]
        silent = len(chunk) == 0 or float(np.sqrt(np.mean(chunk ** 2))) < gate_rms
        if pending_fire is not None:
            if not silent:
                pending_fire = None
            elif k - pending_fire >= confirm_chunks:
                fires.append((k + 1) * chunk_s)
                pending_fire = None
        if dec.buffer is None and silent:
            continue                                   # energy gate
        text = dec.step(chunk)
        n_mark = text.count(END_TOK)
        if n_mark > prev_marker_count:
            after = text.rsplit(END_TOK, 1)[1].replace(EAGER_TOK, "").strip()
            if confirm_chunks == 0:
                fires.extend([(k + 1) * chunk_s] * (n_mark - prev_marker_count))
            elif not after:
                pending_fire = k
            if not after:
                flushed.append(text.replace(EAGER_TOK, ""))
                dec.reset_segment()
                prev_marker_count = 0
                continue
        prev_marker_count = n_mark
    if dec.raw:
        flushed.append(clean(dec.raw).replace(EAGER_TOK, ""))

    last_end = item["last_speech_end_s"]
    premature = [t for t in fires if t <= last_end + 0.25]
    final = [t for t in fires if last_end + 0.25 < t <= last_end + 1.75]
    hyp = " ".join(t.replace(END_TOK, " ") for t in flushed).lower()
    hyp_digits = re.findall(
        r"zero|one|two|three|four|five|six|seven|eight|nine", hyp)
    ref_digits = item["text"].split()
    # digit accuracy = 1 - WER restricted to digit vocab (simple edit distance)
    import jiwer
    dacc = 1.0 - min(1.0, jiwer.wer(" ".join(ref_digits), " ".join(hyp_digits))
                     if hyp_digits else 1.0)
    return {
        "id": item["id"], "n_fires": len(fires), "fires_s": fires,
        "n_premature": len(premature),
        "final_recall": int(bool(final)),
        "final_latency_s": (final[0] - last_end) if final else None,
        "digit_acc": round(dacc, 3),
        "hyp": hyp[:200],
    }


def timeout_baseline(items: list[dict], X: float) -> dict:
    prem, rec, lat = 0, 0, []
    for it in items:
        prem += sum(1 for s, e in it["pauses"] if (e - s) >= X)
        if X <= 1.75:
            rec += 1
            lat.append(X)
    n = len(items)
    return {"X": X, "premature_per_seq": round(prem / n, 2),
            "final_recall": round(rec / n, 3),
            "final_latency_s": X if X <= 1.75 else None}


# ------------------------------------------------------------------ Probe B

def norm_entity(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9@. ]", " ", s)
    s = re.sub(r"\bat\b", "@", s)
    s = re.sub(r"\bdot\b", ".", s)
    s = re.sub(r"\bdash\b", "-", s)
    s = re.sub(r"\s+", "", s)
    return s


def offline_decode(model, tokenizer, processor, audio: np.ndarray,
                   ctx: str, max_new: int = 160) -> str:
    msgs = [
        {"role": "system", "content": ctx},
        {"role": "user", "content": [{"type": "audio"}]},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
    inputs = processor(text=[prompt], audio=[audio], return_tensors="pt",
                       padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device).bfloat16() if torch.is_floating_point(v)
              else v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new)
    gen = tokenizer.decode(out.sequences[0, inputs["input_ids"].shape[1]:],
                           skip_special_tokens=False)
    return clean(gen).replace(END_TOK, " ").replace(EAGER_TOK, " ")


def run_probe_b(model, tokenizer, processor, items: list[dict]) -> dict:
    per = []
    for it in items:
        audio, sr = sf.read(it["wav"], dtype="float32")
        assert sr == SR
        row = {"id": it["id"], "kind": it["kind"], "style": it.get("style", "spelled"),
               "target": it["target"]}
        for cond, ctx in [("none", ""), ("profile", it["ctx"]),
                          ("distractor", it["ctx_distractor"])]:
            hyp = offline_decode(model, tokenizer, processor, audio, ctx)
            hit = int(norm_entity(it["target"]) in norm_entity(hyp))
            row[f"hyp_{cond}"] = hyp[:220]
            row[f"hit_{cond}"] = hit
        # distractor intrusion: distractor profile's email local-part appears
        m = re.search(r"email: ([^;]+);", it["ctx_distractor"])
        if m:
            d_ent = norm_entity(m.group(1).split("@")[0])
            row["intrusion_distractor"] = int(
                d_ent in norm_entity(row["hyp_distractor"]) and
                d_ent not in norm_entity(it["target"]))
        per.append(row)
        logger.info("B[%d/%d] %s hit none=%d profile=%d distractor=%d",
                    it["id"], len(items), it["kind"], row["hit_none"],
                    row["hit_profile"], row["hit_distractor"])

    def rate(kind, cond, style=None):
        rows = [r for r in per if r["kind"] == kind and
                (style is None or r["style"] == style)]
        return round(sum(r[f"hit_{cond}"] for r in rows) / max(1, len(rows)), 3)

    summary = {}
    for kind in ("name", "email"):
        for cond in ("none", "profile", "distractor"):
            summary[f"{kind}_{cond}"] = rate(kind, cond)
    for style in ("spelled", "natural"):
        for cond in ("none", "profile"):
            summary[f"email_{style}_{cond}"] = rate("email", cond, style)
    summary["intrusion_rate"] = round(
        sum(r.get("intrusion_distractor", 0) for r in per) / len(per), 3)
    return {"summary": summary, "per_item": per}


# ------------------------------------------------------------------ main

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--probes", default="data/probes")
    p.add_argument("--out", required=True)
    p.add_argument("--chunk-s", type=float, default=0.5)
    p.add_argument("--gate-rms", type=float, default=1e-3)
    p.add_argument("--confirm-silent-chunks", type=int, default=0)
    p.add_argument("--skip-a", action="store_true")
    p.add_argument("--skip-b", action="store_true")
    args = p.parse_args()

    model, tokenizer, processor = load_model(args.checkpoint)
    result = {"checkpoint": args.checkpoint,
              "confirm_silent_chunks": args.confirm_silent_chunks}

    if not args.skip_a:
        items = json.loads(Path(args.probes + "/digit_probe.json").read_text())
        dec = StreamDecoder(model, processor, tokenizer, committed=True)
        per = []
        t0 = time.time()
        for it in items:
            r = run_digit_item(dec, it, args.chunk_s, args.gate_rms,
                               args.confirm_silent_chunks)
            per.append(r)
            logger.info("A[%d/%d] premature=%d final=%d lat=%s acc=%.2f",
                        it["id"] + 1, len(items), r["n_premature"],
                        r["final_recall"], r["final_latency_s"], r["digit_acc"])
        lat = [r["final_latency_s"] for r in per if r["final_latency_s"] is not None]
        result["digit"] = {
            "summary": {
                "n_items": len(per),
                "premature_per_seq": round(np.mean([r["n_premature"] for r in per]), 3),
                "seqs_with_any_premature": round(
                    np.mean([r["n_premature"] > 0 for r in per]), 3),
                "final_recall": round(np.mean([r["final_recall"] for r in per]), 3),
                "final_latency_p50": round(float(np.median(lat)), 3) if lat else None,
                "digit_acc_mean": round(np.mean([r["digit_acc"] for r in per]), 3),
                "runtime_s": round(time.time() - t0, 1),
            },
            "timeout_baselines": [timeout_baseline(items, x) for x in (0.5, 1.0)],
            "per_item": per,
        }
        logger.info("Probe A summary: %s", result["digit"]["summary"])

    if not args.skip_b:
        items = json.loads(Path(args.probes + "/spelled_probe.json").read_text())
        result["spelled"] = run_probe_b(model, tokenizer, processor, items)
        logger.info("Probe B summary: %s", result["spelled"]["summary"])

    Path(args.out).write_text(json.dumps(result, indent=1))
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
