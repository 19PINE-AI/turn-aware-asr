"""Build a conversational turn-boundary eval set from the AMI Meeting Corpus.

AMI ships per-utterance segments with speaker_id and timing. For each meeting
we walk consecutive utterances; when the speaker changes, that's a real
turn boundary. We build three eval schemas:

  * single: solo utterance (one speaker, one utterance) — should emit
    one <END_SPEECH> at the end.
  * double: two consecutive utterances from DIFFERENT speakers, concatenated
    with their natural gap reconstructed as silence (gap = utt_B.begin -
    utt_A.end clipped to [0.2, 3.0] s). Should emit two <END_SPEECH>.
  * disfluency: two consecutive utterances from the SAME speaker, joined
    with their natural gap. Should emit ONE <END_SPEECH> (the speaker
    continues across the pause).

The "real" thing this measures over the LibriSpeech-synthetic eval:
- Spontaneous speech (filler words, "yeah", incomplete sentences)
- Real proper nouns and meeting jargon
- Natural inter-utterance pause durations
- Multi-speaker diversity (33 meetings in test)

Usage:
    python -m eval.ami_conversational_eval --max-per-schema 50 \\
        --checkpoint checkpoints/semantic_endpoint_v3_long/step6000.pt
"""

from __future__ import annotations
import argparse
import io
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from tqdm import tqdm

from eval.metrics import wer

logger = logging.getLogger(__name__)

END_TOK = "<END_SPEECH>"
EAGER_TOK = "<EAGER_END_SPEECH>"


def decode_audio(blob: dict) -> tuple[np.ndarray, int]:
    """AMI audio cell is {bytes, path, sampling_rate, array}. Use 'bytes' if present."""
    if "array" in blob and blob["array"] is not None:
        return np.asarray(blob["array"], dtype=np.float32), int(blob.get("sampling_rate", 16000))
    if "bytes" in blob:
        audio, sr = sf.read(io.BytesIO(blob["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        return audio.astype(np.float32), int(sr)
    raise ValueError(f"can't decode audio cell: keys={list(blob.keys())}")


def load_ami_meetings(parquet_dir: Path, max_meetings: int | None = None,
                      restrict_to: set[str] | None = None
                      ) -> dict[str, list[dict]]:
    """Group utterances by meeting_id, sorted by begin_time.

    If `restrict_to` is given, only those meeting_ids are loaded — used to
    enforce the train/eval meeting split when evaluating a model that was
    trained on AMI.
    """
    meetings: dict[str, list[dict]] = defaultdict(list)
    for pq_path in sorted(parquet_dir.glob("*.parquet")):
        logger.info("Reading %s…", pq_path.name)
        t = pq.read_table(pq_path)
        for row in t.to_pylist():
            if restrict_to is not None and row["meeting_id"] not in restrict_to:
                continue
            meetings[row["meeting_id"]].append(row)
    for mid in meetings:
        meetings[mid].sort(key=lambda r: r["begin_time"])
    if max_meetings is not None:
        keep = sorted(meetings.keys())[:max_meetings]
        meetings = {k: meetings[k] for k in keep}
    logger.info("Loaded %d meetings with %d total utts",
                 len(meetings), sum(len(v) for v in meetings.values()))
    return meetings


def build_examples(meetings: dict[str, list[dict]], rng: random.Random,
                    max_per_schema: int = 50, max_dur_s: float = 12.0) -> list[dict]:
    """Walk consecutive utterance pairs in each meeting; build examples."""
    singles, doubles, disfluencies = [], [], []
    sr = 16000

    for mid, utts in meetings.items():
        if len(utts) < 2:
            continue
        # Solo examples — pick at random
        if len(singles) < max_per_schema * 4:
            for utt in utts:
                try:
                    audio, asr = decode_audio(utt["audio"])
                except Exception:
                    continue
                if asr != sr or len(audio) == 0:
                    continue
                if len(audio) > int(max_dur_s * sr):
                    continue
                if not utt["text"].strip():
                    continue
                singles.append({
                    "audio": audio,
                    "text": utt["text"].strip(),
                    "schema": "single",
                    "meeting_id": mid,
                    "speaker_id": utt["speaker_id"],
                })
                if len(singles) >= max_per_schema * 4:
                    break

        # Pair examples
        for i in range(len(utts) - 1):
            a = utts[i]
            b = utts[i + 1]
            if not a["text"].strip() or not b["text"].strip():
                continue
            gap = float(b["begin_time"]) - float(a["end_time"])
            if not (0.05 < gap < 3.0):
                continue
            try:
                aa, asr_a = decode_audio(a["audio"])
                ab, asr_b = decode_audio(b["audio"])
            except Exception:
                continue
            if asr_a != sr or asr_b != sr:
                continue
            if len(aa) == 0 or len(ab) == 0:
                continue
            total_dur = (len(aa) + len(ab)) / sr + gap
            if total_dur > max_dur_s:
                continue
            gap_samples = int(gap * sr)
            cat = np.concatenate([aa, np.zeros(gap_samples, dtype=np.float32), ab])
            example = {
                "audio": cat,
                "text_a": a["text"].strip(),
                "text_b": b["text"].strip(),
                "speaker_a": a["speaker_id"],
                "speaker_b": b["speaker_id"],
                "gap_s": gap,
                "meeting_id": mid,
            }
            if a["speaker_id"] != b["speaker_id"]:
                example["schema"] = "double"
                doubles.append(example)
            else:
                example["schema"] = "disfluency"
                disfluencies.append(example)

    # Sample / shuffle
    rng.shuffle(singles)
    rng.shuffle(doubles)
    rng.shuffle(disfluencies)
    selected = (
        singles[:max_per_schema]
        + doubles[:max_per_schema]
        + disfluencies[:max_per_schema]
    )
    logger.info("Built %d examples (single=%d, double=%d, disfluency=%d)",
                 len(selected),
                 min(max_per_schema, len(singles)),
                 min(max_per_schema, len(doubles)),
                 min(max_per_schema, len(disfluencies)))
    return selected


def transcribe(model, processor, tokenizer, audio: np.ndarray, max_new: int = 160) -> str:
    import re
    device = next(model.parameters()).device
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio"}]},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], audio=[audio], return_tensors="pt", padding=True)
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


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ami-dir", default="data/ami/ihm")
    p.add_argument("--max-per-schema", type=int, default=50)
    p.add_argument("--max-meetings", type=int, default=None)
    p.add_argument("--max-dur-s", type=float, default=12.0)
    p.add_argument("--out", default="research/26-ami-conversational-eval.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--restrict-to-meetings",
                    help="Path to JSON file with 'eval_meeting_ids' list; only those meetings are used. "
                         "Required when the model was trained on AMI to avoid contamination.")
    args = p.parse_args()

    rng = random.Random(args.seed)

    restrict_to: set[str] | None = None
    if args.restrict_to_meetings:
        d = json.loads(Path(args.restrict_to_meetings).read_text())
        restrict_to = set(d.get("eval_meeting_ids", []))
        logger.info("Restricting to %d held-out meetings", len(restrict_to))

    logger.info("Loading AMI meetings…")
    meetings = load_ami_meetings(Path(args.ami_dir), max_meetings=args.max_meetings,
                                  restrict_to=restrict_to)

    logger.info("Building examples…")
    examples = build_examples(meetings, rng, args.max_per_schema, args.max_dur_s)

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
    logger.info("Loaded %s (step %d)", args.checkpoint, ckpt.get("step", -1))

    per: list[dict] = []
    for i, e in enumerate(tqdm(examples, desc="ami-eval")):
        hyp = transcribe(model, processor, tokenizer, e["audio"])
        n_end = hyp.count(END_TOK)
        n_eager = hyp.count(EAGER_TOK)
        r = {
            "schema": e["schema"],
            "meeting_id": e["meeting_id"],
            "hyp": hyp,
            "n_end_hyp": n_end,
            "n_eager_hyp": n_eager,
        }
        # Schema-specific bookkeeping
        if e["schema"] == "single":
            ref = e["text"]
            r["ref"] = ref
        else:
            r["text_a"] = e["text_a"]
            r["text_b"] = e["text_b"]
            r["speaker_a"] = e["speaker_a"]
            r["speaker_b"] = e["speaker_b"]
            r["gap_s"] = e["gap_s"]
        if i < 3:
            logger.info("REF[%d] (%s): %s",
                         i, e["schema"],
                         (e.get("text") or f"{e.get('text_a', '')} | {e.get('text_b', '')}")[:140])
            logger.info("HYP[%d]: %s", i, hyp[:140])
        per.append(r)

    # Aggregate
    def _n(schema): return sum(1 for r in per if r["schema"] == schema)
    summary = {
        "n_examples": len(per),
        "n_single": _n("single"),
        "n_double": _n("double"),
        "n_disfluency": _n("disfluency"),
        # Marker metrics
        "end_recall_single": (
            sum(1 for r in per if r["schema"] == "single" and r["n_end_hyp"] >= 1) / max(1, _n("single"))
        ),
        "end_recall_double": (
            sum(1 for r in per if r["schema"] == "double" and r["n_end_hyp"] >= 2) / max(1, _n("double"))
        ),
        "end_correct_disfluency": (
            sum(1 for r in per if r["schema"] == "disfluency" and r["n_end_hyp"] == 1) / max(1, _n("disfluency"))
        ),
        "end_overfire_disfluency": (
            sum(1 for r in per if r["schema"] == "disfluency" and r["n_end_hyp"] > 1) / max(1, _n("disfluency"))
        ),
    }
    logger.info("== AMI SUMMARY ==")
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info("  %-36s %.3f", k, v)
        else:
            logger.info("  %-36s %s", k, v)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Save without audio bytes (would blow up the JSON)
    Path(args.out).write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "summary": summary,
        "per_example": per,
    }, indent=2))
    logger.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
