"""Build v7 training data: add NO-FIRE examples for AMI-style audio.

v6 added AMI-with-trailing-silence to recover AMI conversation,
but streaming P50 regressed −5.17 s because the silence-cue prior
collapsed: the model treated silence as one valid trigger among
several rather than NECESSARY.

v7 adds a counter-signal: AMI single utterances with the markers
STRIPPED. Same audio, just transcript, no end token. This teaches:

  audio + silence + markers  → fire        (existing)
  audio + no silence + no markers → DON'T fire  (NEW)
  truncated audio + no markers → don't fire  (existing)

Making silence necessary instead of optional.

Composition:
  v6 base: 9994 examples
  + ~1000 AMI single no-fire (clone of existing AMI single, no markers)
  ≈ 11000 total

The cloned audio is the unmodified AMI utterance segment — minimal
trailing silence since AMI uses tight word-level segmentation.

Usage:
    python -m eval.build_v7_training_data --n-no-fire 1000
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


def strip_markers(text: str) -> str:
    return text.replace(EAGER_TOK, "").replace(END_TOK, "").strip()


def no_fire_example(src: dict) -> dict:
    return {
        "audio": np.asarray(src["audio"], dtype=np.float32),
        "text": strip_markers(src["text"]),
        "schema": "no_fire",
        "source": src.get("source", "ami"),
        "src_schema": src["schema"],
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--base-data", default="data/semantic_endpoint_v6/data.pt")
    p.add_argument("--n-no-fire", type=int, default=1000)
    p.add_argument("--out", default="data/semantic_endpoint_v7/data.pt")
    p.add_argument("--seed", type=int, default=4)
    args = p.parse_args()

    rng = random.Random(args.seed)
    logger.info("Loading v6 base %s…", args.base_data)
    base = torch.load(args.base_data, weights_only=False)

    # Source pool: AMI single examples from v6 base.
    # NOTE: only use single schema where the original audio ended naturally
    # (no synthetic trailing silence appended). We exclude trailing_silence
    # schema because those have been silence-padded.
    ami_singles = [e for e in base
                    if e["schema"] == "single" and e.get("source") == "ami"]
    logger.info("AMI single source pool: %d", len(ami_singles))
    rng.shuffle(ami_singles)

    extra: list[dict] = []
    for i in range(args.n_no_fire):
        if not ami_singles:
            break
        src = ami_singles[i % len(ami_singles)]
        extra.append(no_fire_example(src))
    logger.info("Generated %d no_fire examples", len(extra))

    combined = base + extra
    rng.shuffle(combined)
    counts = defaultdict(int)
    sources = defaultdict(int)
    for e in combined:
        counts[e["schema"]] += 1
        sources[e.get("source", "librispeech_synth")] += 1
    logger.info("v7: %d examples", len(combined))
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
