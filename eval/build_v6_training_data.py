"""Build v6 training data: bridge v3 (AMI conv.) and v5 (streaming).

v5 traded AMI conversation accuracy for streaming P50 going positive.
The reason: v5 learned "fire only after trailing silence", and AMI's
tight conversational style doesn't have trailing silence between
turns.

v6 fix: add AMI examples WITH synthetic trailing silence. Teach
the model:
  - AMI utts CAN have trailing silence at the end (fire then)
  - AMI utts also work in tight conversation (existing examples)

This breaks the "trailing silence required" prior without removing
the streaming benefit.

Composition:
  v5 base: 8494 examples
  + 1000 AMI-single + trailing silence
  + 500  AMI-double + trailing silence
  ≈ 10000 total, AMI presence ~30 %

Usage:
    python -m eval.build_v6_training_data \\
        --n-ami-single-trail 1000 --n-ami-double-trail 500
"""

from __future__ import annotations
import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

EAGER_TOK = "<EAGER_END_SPEECH>"
END_TOK = "<END_SPEECH>"
SR = 16000


def trail_example(src: dict, rng: random.Random,
                   min_silence_s: float = 0.3,
                   max_silence_s: float = 1.0) -> dict:
    silence_s = rng.uniform(min_silence_s, max_silence_s)
    audio = np.asarray(src["audio"], dtype=np.float32)
    pad = np.zeros(int(silence_s * SR), dtype=np.float32)
    new_audio = np.concatenate([audio, pad])
    return {
        "audio": new_audio,
        "text": src["text"],  # markers stay where they are
        "schema": "trailing_silence",
        "source": src.get("source", "ami"),
        "src_schema": src["schema"],
        "trail_silence_s": silence_s,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--base-data", default="data/semantic_endpoint_v5/data.pt")
    p.add_argument("--n-ami-single-trail", type=int, default=1000)
    p.add_argument("--n-ami-double-trail", type=int, default=500)
    p.add_argument("--out", default="data/semantic_endpoint_v6/data.pt")
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()

    rng = random.Random(args.seed)
    logger.info("Loading v5 base data %s…", args.base_data)
    base = torch.load(args.base_data, weights_only=False)
    by_schema_source = defaultdict(list)
    for e in base:
        key = (e["schema"], e.get("source", "librispeech_synth"))
        by_schema_source[key].append(e)
    logger.info("v5 source counts: %s",
                 {f"{k[0]}-{k[1]}": len(v) for k, v in by_schema_source.items()})

    extra: list[dict] = []

    # AMI single + trailing silence
    ami_singles = by_schema_source.get(("single", "ami"), [])
    if not ami_singles:
        logger.warning("No AMI single examples found in base data")
    rng.shuffle(ami_singles)
    for i in range(args.n_ami_single_trail):
        if not ami_singles: break
        src = ami_singles[i % len(ami_singles)]
        extra.append(trail_example(src, rng))
    logger.info("Added %d AMI single+trail examples", min(args.n_ami_single_trail, len(ami_singles) * 3 if ami_singles else 0))

    # AMI double + trailing silence
    ami_doubles = by_schema_source.get(("double", "ami"), [])
    if not ami_doubles:
        logger.warning("No AMI double examples found in base data")
    rng.shuffle(ami_doubles)
    for i in range(args.n_ami_double_trail):
        if not ami_doubles: break
        src = ami_doubles[i % len(ami_doubles)]
        extra.append(trail_example(src, rng))
    logger.info("Added %d AMI double+trail examples", min(args.n_ami_double_trail, len(ami_doubles) * 3 if ami_doubles else 0))

    combined = base + extra
    rng.shuffle(combined)
    counts = defaultdict(int)
    sources = defaultdict(int)
    for e in combined:
        counts[e["schema"]] += 1
        sources[e.get("source", "librispeech_synth")] += 1
    logger.info("v6: %d examples", len(combined))
    logger.info("  schema: %s", dict(counts))
    logger.info("  source: %s", dict(sources))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(combined, args.out)
    logger.info("Saved %s", args.out)

    # Copy meeting split forward
    import shutil
    src_split = Path(args.base_data).parent / "meeting_split.json"
    if src_split.exists():
        dst_split = Path(args.out).parent / "meeting_split.json"
        shutil.copy(src_split, dst_split)
        logger.info("Copied meeting split → %s", dst_split)


if __name__ == "__main__":
    main()
