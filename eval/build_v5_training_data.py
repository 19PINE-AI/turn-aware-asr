"""Build v5 training data: add trailing-silence examples to v4.

v4 has 6494 examples with 38 % truncated. Streaming P50 is -7.6s
because the model can't distinguish chunk-1 of a long utterance from
a complete short utterance — both look like "speech that just ended."

v5 adds explicit trailing-silence training:
  - 1500 single+silence: take existing single example, append 0.3-1.0s
    silence to the audio, KEEP the existing markers in text.
  - 500 double+silence: take existing double example, append 0.3-1.0s
    silence at the end, KEEP both marker pairs.

Teaches the model that the firing CUE is silence-after-speech, not
end-of-audio. At inference time:
  - chunk N during ongoing speech: no trailing silence in buffer →
    don't fire (matches truncated training)
  - chunk N when user stopped: trailing silence appears → fire
    (matches trailing-silence training)

Usage:
    python -m eval.build_v5_training_data \\
        --n-single-trail 1500 --n-double-trail 500
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
    """Append silence to a complete example; text + markers unchanged."""
    silence_s = rng.uniform(min_silence_s, max_silence_s)
    audio = np.asarray(src["audio"], dtype=np.float32)
    pad = np.zeros(int(silence_s * SR), dtype=np.float32)
    new_audio = np.concatenate([audio, pad])
    return {
        "audio": new_audio,
        "text": src["text"],  # unchanged: markers stay at end
        "schema": "trailing_silence",
        "source": src.get("source", "librispeech_synth"),
        "src_schema": src["schema"],
        "trail_silence_s": silence_s,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--base-data", default="data/semantic_endpoint_v4/data.pt")
    p.add_argument("--n-single-trail", type=int, default=1500)
    p.add_argument("--n-double-trail", type=int, default=500)
    p.add_argument("--out", default="data/semantic_endpoint_v5/data.pt")
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args()

    rng = random.Random(args.seed)
    logger.info("Loading v4 base data %s…", args.base_data)
    base = torch.load(args.base_data, weights_only=False)
    by_schema = defaultdict(list)
    for e in base:
        by_schema[e["schema"]].append(e)
    logger.info("v4 source schema counts: %s",
                 {k: len(v) for k, v in by_schema.items()})

    extra: list[dict] = []

    # Single + trailing silence
    singles = list(by_schema["single"])
    rng.shuffle(singles)
    n = min(args.n_single_trail, len(singles))
    for i in range(args.n_single_trail):
        src = singles[i % len(singles)]
        extra.append(trail_example(src, rng))
    logger.info("Generated %d single+trail examples", args.n_single_trail)

    # Double + trailing silence
    doubles = list(by_schema["double"])
    rng.shuffle(doubles)
    for i in range(args.n_double_trail):
        src = doubles[i % len(doubles)]
        extra.append(trail_example(src, rng))
    logger.info("Generated %d double+trail examples", args.n_double_trail)

    combined = base + extra
    rng.shuffle(combined)
    counts = defaultdict(int)
    for e in combined:
        counts[e["schema"]] += 1
    logger.info("Final v5: %d examples, schema counts %s",
                 len(combined), dict(counts))

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
