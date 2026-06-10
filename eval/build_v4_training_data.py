"""Build v4 training data: heavier truncated mix to fix streaming.

v3 had 494 truncated examples (11% of mix) and got streaming P50 to
-9.6s. To push that toward zero we need ~30-40% truncated. This script
generates more aggressive truncation:

  - n_single_truncated: extra single-utt truncations (1500 default)
  - n_double_truncated: take double examples, cut at a point in utt_A's
    audio (before any marker), label as truncated (no markers). 500 default.

Combined with v3 base, target distribution:
  Original: 1000 S + 2000 D + 1000 F + 494 T = 4494
  v4: add 1500 ST + 500 DT → 6494, truncated = 2494 (38 %)

Usage:
    python -m eval.build_v4_training_data \\
        --n-single-trunc 1500 --n-double-trunc 500
"""

from __future__ import annotations
import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

EAGER_TOK = "<EAGER_END_SPEECH>"
END_TOK = "<END_SPEECH>"
SR = 16000


def truncate_single(src: dict, rng: random.Random,
                     min_frac: float = 0.30, max_frac: float = 0.80) -> dict | None:
    text_full = src.get("text", "").replace(EAGER_TOK, "").replace(END_TOK, "").strip()
    words = text_full.split()
    if len(words) < 3:
        return None
    audio = np.asarray(src["audio"], dtype=np.float32)
    if len(audio) < int(2.5 * SR):
        return None
    frac = rng.uniform(min_frac, max_frac)
    cut_samples = int(len(audio) * frac)
    if cut_samples < int(1.0 * SR):
        return None
    cut_words = max(1, int(round(len(words) * frac)))
    return {
        "audio": audio[:cut_samples],
        "text": " ".join(words[:cut_words]),
        "schema": "truncated",
        "source": src.get("source", "librispeech_synth"),
        "src_schema": "single",
        "trunc_frac": frac,
    }


def truncate_double_early(src: dict, rng: random.Random) -> dict | None:
    """Cut audio at a point inside utt_A (well before the boundary marker)."""
    audio = np.asarray(src["audio"], dtype=np.float32)
    # The source double has audio = utt_A + gap + utt_B.
    # We need to know where utt_A ends. If src has 'audio_end_s_a' / 'gap_start_s'
    # from earlier synth (eval.semantic_endpoint_data), use that. Otherwise
    # heuristically cap by half the total audio.
    gap_start_s = src.get("gap_start_s") or src.get("audio_end_s_a")
    if gap_start_s is not None:
        utt_a_end_samples = int(float(gap_start_s) * SR)
    else:
        # Heuristic: utt_A is likely the first ~45 % of audio
        utt_a_end_samples = int(len(audio) * 0.45)
    if utt_a_end_samples < int(1.5 * SR):
        return None
    # Cut at 50 - 95 % of utt_A's duration
    frac = rng.uniform(0.5, 0.95)
    cut_samples = int(utt_a_end_samples * frac)
    if cut_samples < int(1.0 * SR):
        return None

    # Get utt_A's text from the full target (split at the first <END_SPEECH>)
    full_text = src.get("text", "")
    a_text = full_text.split(END_TOK, 1)[0]
    a_text = a_text.replace(EAGER_TOK, "").strip()
    words = a_text.split()
    if len(words) < 3:
        return None
    cut_words = max(1, int(round(len(words) * frac)))
    return {
        "audio": audio[:cut_samples],
        "text": " ".join(words[:cut_words]),
        "schema": "truncated",
        "source": src.get("source", "librispeech_synth"),
        "src_schema": "double",
        "trunc_frac": frac,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--base-data", default="data/semantic_endpoint_v3/data.pt")
    p.add_argument("--n-single-trunc", type=int, default=1500)
    p.add_argument("--n-double-trunc", type=int, default=500)
    p.add_argument("--out", default="data/semantic_endpoint_v4/data.pt")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    rng = random.Random(args.seed)

    logger.info("Loading v3 base data %s…", args.base_data)
    base = torch.load(args.base_data, weights_only=False)
    by_schema = defaultdict(list)
    for e in base:
        by_schema[e["schema"]].append(e)
    logger.info("v3 source schema counts: %s",
                 {k: len(v) for k, v in by_schema.items()})

    extra_truncated = []

    # Single truncations
    singles = list(by_schema["single"])
    rng.shuffle(singles)
    logger.info("Generating %d single-truncated examples…", args.n_single_trunc)
    src_idx = 0
    while len(extra_truncated) < args.n_single_trunc and src_idx < len(singles) * 3:
        src = singles[src_idx % len(singles)]
        t = truncate_single(src, rng)
        if t is not None:
            extra_truncated.append(t)
        src_idx += 1
    n_after_single = len(extra_truncated)
    logger.info("  generated %d single-truncated", n_after_single)

    # Double truncations
    doubles = list(by_schema["double"])
    rng.shuffle(doubles)
    logger.info("Generating %d double-truncated examples…", args.n_double_trunc)
    src_idx = 0
    while len(extra_truncated) - n_after_single < args.n_double_trunc and src_idx < len(doubles) * 3:
        src = doubles[src_idx % len(doubles)]
        t = truncate_double_early(src, rng)
        if t is not None:
            extra_truncated.append(t)
        src_idx += 1
    logger.info("  generated %d double-truncated", len(extra_truncated) - n_after_single)

    # Combine
    combined = base + extra_truncated
    rng.shuffle(combined)
    final_counts = defaultdict(int)
    for e in combined:
        final_counts[e["schema"]] += 1
    pct_trunc = final_counts["truncated"] / max(1, len(combined)) * 100
    logger.info("Final v4: %d examples, schema counts %s (truncated = %.1f %%)",
                 len(combined), dict(final_counts), pct_trunc)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(combined, args.out)
    logger.info("Saved %s", args.out)

    # Copy the meeting split forward
    import shutil
    src_split = Path("data/semantic_endpoint_v3/meeting_split.json")
    dst_split = Path(args.out).parent / "meeting_split.json"
    if src_split.exists():
        shutil.copy(src_split, dst_split)
        logger.info("Copied meeting split → %s", dst_split)


if __name__ == "__main__":
    main()
