"""Build v8 training data: disjoint no_fire pool.

v7 added 1000 no_fire examples that CLONED the same 500 AMI singles
already in training as fire-with-markers. Model defaulted to "don't
fire" on AMI-style audio (single recall 0 %).

v8 fix: draw a FRESH 1000 AMI utts from train meetings, NOT
overlapping with the 500 already in v6 as fire-with-markers. Same
schema (no_fire), but new audio.

If the audio is truly disjoint, the model should keep firing on the
500 known fire-with-markers audios while learning "AMI-style audio
without markers in training means no fire" generalizes more
carefully — relying on whatever subtle features distinguish the two
sets.

Composition:
  v6 base: 9994 examples (incl. 500 AMI singles as fire)
  + 1000 fresh AMI no_fire (different utts, train meetings)
  ≈ 11000 total

Usage:
    python -m eval.build_v8_training_data --n-no-fire 1000
"""

from __future__ import annotations
import argparse
import hashlib
import io
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

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


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--base-data", default="data/semantic_endpoint_v6/data.pt")
    p.add_argument("--ami-dir", default="data/ami/ihm")
    p.add_argument("--n-no-fire", type=int, default=1000)
    p.add_argument("--out", default="data/semantic_endpoint_v8/data.pt")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--max-dur-s", type=float, default=12.0)
    p.add_argument("--min-dur-s", type=float, default=0.3)
    args = p.parse_args()

    rng = random.Random(args.seed)
    logger.info("Loading v6 base %s…", args.base_data)
    base = torch.load(args.base_data, weights_only=False)

    # Build the existing AMI utt fingerprint set (meeting_id, text) to exclude.
    existing_keys = set()
    for e in base:
        if e.get("source") == "ami":
            txt = e["text"]
            for tok in (EAGER_TOK, END_TOK):
                txt = txt.replace(tok, "")
            existing_keys.add((e.get("meeting_id"), txt.strip()))
    logger.info("Existing AMI fingerprints in v6: %d", len(existing_keys))

    # Walk parquet, collect candidate utts from train meetings only.
    candidates: list[dict] = []
    for pq_path in sorted(Path(args.ami_dir).glob("*.parquet")):
        logger.info("Reading %s…", pq_path.name)
        t = pq.read_table(pq_path)
        for row in t.to_pylist():
            mid = row["meeting_id"]
            if not meeting_in_train(mid):
                continue
            text = (row["text"] or "").strip()
            if not text:
                continue
            key = (mid, text)
            if key in existing_keys:
                continue
            audio, sr = decode_audio(row["audio"])
            if audio is None or sr != SR or len(audio) == 0:
                continue
            dur = len(audio) / SR
            if dur < args.min_dur_s or dur > args.max_dur_s:
                continue
            candidates.append({
                "audio": audio,
                "text": text,
                "schema": "no_fire",
                "source": "ami",
                "meeting_id": mid,
                "src_schema": "single",
            })
            if len(candidates) >= args.n_no_fire * 3:
                break
        if len(candidates) >= args.n_no_fire * 3:
            break
    logger.info("Candidate pool: %d", len(candidates))

    rng.shuffle(candidates)
    extra = candidates[: args.n_no_fire]
    logger.info("Selected %d no_fire examples", len(extra))

    combined = base + extra
    rng.shuffle(combined)
    counts = defaultdict(int)
    sources = defaultdict(int)
    for e in combined:
        counts[e["schema"]] += 1
        sources[e.get("source", "librispeech_synth")] += 1
    logger.info("v8: %d examples", len(combined))
    logger.info("  schema: %s", dict(counts))
    logger.info("  source: %s", dict(sources))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(combined, args.out)
    logger.info("Saved %s", args.out)

    import shutil
    src_split = Path(args.base_data).parent / "meeting_split.json"
    if src_split.exists():
        dst_split = Path(args.out).parent / "meeting_split.json"
        shutil.copy(src_split, dst_split)
        logger.info("Copied meeting split → %s", dst_split)


if __name__ == "__main__":
    main()
