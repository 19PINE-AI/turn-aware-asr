"""Build v3 training data: existing synthetic + truncated + AMI turn pairs.

Mix:
  - 2500 existing v2 examples (single/double/disfluency on LibriSpeech)
  - 2000 truncated examples (audio cut at 30-80%, proportional partial text,
        schema=`truncated`, NO end markers)
  - up to 1500 AMI examples from train-meeting half (real conversational)

AMI meetings are split deterministically by hash(meeting_id) % 3:
  train (used here): hash % 3 in {0, 1}
  eval (held out):   hash % 3 == 2

Output: data/semantic_endpoint_v3/data.pt
        data/semantic_endpoint_v3/meeting_split.json (train-vs-eval meeting ids)

Usage:
    python -m eval.build_v3_training_data \\
        --n-truncated 2000 --n-ami-per-schema 500
"""

from __future__ import annotations
import argparse
import hashlib
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

logger = logging.getLogger(__name__)

EAGER_TOK = "<EAGER_END_SPEECH>"
END_TOK = "<END_SPEECH>"
SR = 16000


def meeting_in_train(meeting_id: str) -> bool:
    h = int(hashlib.md5(meeting_id.encode()).hexdigest(), 16)
    return (h % 3) in (0, 1)


def decode_audio(blob: dict):
    if "array" in blob and blob["array"] is not None:
        return np.asarray(blob["array"], dtype=np.float32), int(blob.get("sampling_rate", 16000))
    if "bytes" in blob:
        audio, sr = sf.read(io.BytesIO(blob["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        return audio.astype(np.float32), int(sr)
    return None, None


def build_truncated(source_examples: list[dict], n: int, rng: random.Random,
                     min_frac: float = 0.30, max_frac: float = 0.80) -> list[dict]:
    """For each source single-utt example, cut audio + text at the same fraction.

    Truncated example schema: audio is truncated, text is partial without
    end markers. Teaches the LM: "if audio looks cut, emit a partial
    transcript and stop (NO end marker)."
    """
    rng.shuffle(source_examples)
    out: list[dict] = []
    skipped = 0
    for src in source_examples:
        if len(out) >= n:
            break
        if src.get("schema") != "single":
            continue
        # Strip end markers from the source text to get clean transcript
        text_full = src.get("text", "").replace(EAGER_TOK, "").replace(END_TOK, "").strip()
        words = text_full.split()
        if len(words) < 3:
            skipped += 1
            continue
        audio = np.asarray(src["audio"], dtype=np.float32)
        if len(audio) < int(2.5 * SR):
            skipped += 1
            continue
        frac = rng.uniform(min_frac, max_frac)
        cut_samples = int(len(audio) * frac)
        if cut_samples < int(1.0 * SR):
            skipped += 1
            continue
        cut_words = max(1, int(round(len(words) * frac)))
        out.append({
            "audio": audio[:cut_samples],
            "text": " ".join(words[:cut_words]),
            "schema": "truncated",
            "src_text_full": text_full,
            "trunc_frac": frac,
        })
    logger.info("Built %d truncated examples (skipped %d)", len(out), skipped)
    return out


def build_ami_train_examples(ami_dir: Path, train_meeting_ids: set[str],
                              n_per_schema: int, rng: random.Random,
                              max_dur_s: float = 12.0) -> list[dict]:
    """Build single/double/disfluency examples from AMI train meetings."""
    meetings: dict[str, list[dict]] = defaultdict(list)
    for pq_path in sorted(ami_dir.glob("*.parquet")):
        logger.info("Reading %s for AMI training set…", pq_path.name)
        t = pq.read_table(pq_path)
        for row in t.to_pylist():
            if row["meeting_id"] in train_meeting_ids:
                meetings[row["meeting_id"]].append(row)
    for mid in meetings:
        meetings[mid].sort(key=lambda r: r["begin_time"])

    singles, doubles, disfluencies = [], [], []
    for mid, utts in meetings.items():
        if len(utts) < 2:
            continue
        # Singles
        for utt in utts:
            if len(singles) >= n_per_schema * 3:
                break
            audio, asr = decode_audio(utt["audio"])
            if audio is None or asr != SR or len(audio) == 0 or len(audio) > int(max_dur_s * SR):
                continue
            if not utt["text"].strip():
                continue
            singles.append({
                "audio": audio,
                "text": f"{utt['text'].strip()} {EAGER_TOK}{END_TOK}",
                "schema": "single",
                "source": "ami",
                "meeting_id": mid,
            })

        # Pairs
        for i in range(len(utts) - 1):
            if len(doubles) >= n_per_schema * 3 and len(disfluencies) >= n_per_schema * 3:
                break
            a, b = utts[i], utts[i + 1]
            if not a["text"].strip() or not b["text"].strip():
                continue
            gap = float(b["begin_time"]) - float(a["end_time"])
            if not (0.05 < gap < 3.0):
                continue
            aa, asr_a = decode_audio(a["audio"])
            ab, asr_b = decode_audio(b["audio"])
            if aa is None or ab is None or asr_a != SR or asr_b != SR:
                continue
            total_dur = (len(aa) + len(ab)) / SR + gap
            if total_dur > max_dur_s or len(aa) == 0 or len(ab) == 0:
                continue
            gap_samples = int(gap * SR)
            cat = np.concatenate([aa, np.zeros(gap_samples, dtype=np.float32), ab])
            text_a = a["text"].strip()
            text_b = b["text"].strip()
            if a["speaker_id"] != b["speaker_id"]:
                doubles.append({
                    "audio": cat,
                    "text": f"{text_a} {EAGER_TOK}{END_TOK} {text_b} {EAGER_TOK}{END_TOK}",
                    "schema": "double",
                    "source": "ami",
                    "meeting_id": mid,
                    "gap_s": gap,
                })
            else:
                disfluencies.append({
                    "audio": cat,
                    "text": f"{text_a} {text_b} {EAGER_TOK}{END_TOK}",
                    "schema": "disfluency",
                    "source": "ami",
                    "meeting_id": mid,
                    "gap_s": gap,
                })

    rng.shuffle(singles)
    rng.shuffle(doubles)
    rng.shuffle(disfluencies)
    out = singles[:n_per_schema] + doubles[:n_per_schema] + disfluencies[:n_per_schema]
    logger.info("AMI train: %d single + %d double + %d disfluency = %d",
                 min(n_per_schema, len(singles)),
                 min(n_per_schema, len(doubles)),
                 min(n_per_schema, len(disfluencies)),
                 len(out))
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--base-data", default="data/semantic_endpoint_v2_simple/data.pt")
    p.add_argument("--ami-dir", default="data/ami/ihm")
    p.add_argument("--n-truncated", type=int, default=2000)
    p.add_argument("--n-ami-per-schema", type=int, default=500)
    p.add_argument("--out-data", default="data/semantic_endpoint_v3/data.pt")
    p.add_argument("--out-split", default="data/semantic_endpoint_v3/meeting_split.json")
    p.add_argument("--max-dur-s", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)

    logger.info("Loading existing v2 data %s…", args.base_data)
    base_examples = torch.load(args.base_data, weights_only=False)
    logger.info("v2 contributes %d examples", len(base_examples))

    # Add metadata for source tracking
    for e in base_examples:
        e.setdefault("source", "librispeech_synth")

    # Build truncated examples from the same source pool
    truncated = build_truncated(
        [e for e in base_examples if e.get("schema") == "single"],
        n=args.n_truncated,
        rng=rng,
    )

    # Determine AMI meeting split
    all_ami_meetings: set[str] = set()
    for pq_path in sorted(Path(args.ami_dir).glob("*.parquet")):
        col = pq.read_table(pq_path, columns=["meeting_id"]).column("meeting_id").to_pylist()
        all_ami_meetings.update(col)
    train_meetings = {m for m in all_ami_meetings if meeting_in_train(m)}
    eval_meetings = all_ami_meetings - train_meetings
    logger.info("AMI meetings: %d total → %d train, %d eval",
                 len(all_ami_meetings), len(train_meetings), len(eval_meetings))

    ami_examples = build_ami_train_examples(
        Path(args.ami_dir), train_meetings,
        n_per_schema=args.n_ami_per_schema,
        rng=rng, max_dur_s=args.max_dur_s,
    )

    combined = base_examples + truncated + ami_examples
    rng.shuffle(combined)
    logger.info("Combined: %d examples", len(combined))
    by_schema = defaultdict(int)
    by_source = defaultdict(int)
    for e in combined:
        by_schema[e["schema"]] += 1
        by_source[e.get("source", "librispeech_synth")] += 1
    logger.info("Schema counts: %s", dict(by_schema))
    logger.info("Source counts: %s", dict(by_source))

    Path(args.out_data).parent.mkdir(parents=True, exist_ok=True)
    torch.save(combined, args.out_data)
    Path(args.out_split).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_split).write_text(json.dumps({
        "train_meeting_ids": sorted(train_meetings),
        "eval_meeting_ids": sorted(eval_meetings),
    }, indent=2))
    logger.info("Saved %s and %s", args.out_data, args.out_split)


if __name__ == "__main__":
    main()
