"""Validate the published Qwen3-ASR-0.6B WER using the official qwen-asr package.

Goal: reproduce the 2.11 % LibriSpeech test-clean WER from the paper. If we
match within ~0.5 pp, we have a working ASR baseline that we can build on
for endpoint detection, context-prefix biasing, and streaming-window
training.

Usage:
    python -m eval.run_qwen3asr_pkg --max-utterances 500
"""

from __future__ import annotations
import argparse
import json
import logging
import glob
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from eval.metrics import wer

logger = logging.getLogger(__name__)


def iter_librispeech_files(split_dir: str):
    """Yield (flac_path, transcript) ordered by filename to match test-clean prep."""
    flacs = sorted(glob.glob(f"{split_dir}/*/*/*.flac"))
    # Each chapter dir has a *.trans.txt with utt_id <text>
    txt_cache: dict[str, dict[str, str]] = {}
    for flac in flacs:
        utt_id = Path(flac).stem
        chap_dir = str(Path(flac).parent)
        if chap_dir not in txt_cache:
            trans = list(Path(chap_dir).glob("*.trans.txt"))
            if not trans:
                continue
            d = {}
            for line in open(trans[0]):
                line = line.strip()
                if not line:
                    continue
                u, _, t = line.partition(" ")
                d[u] = t
            txt_cache[chap_dir] = d
        if utt_id in txt_cache[chap_dir]:
            yield flac, txt_cache[chap_dir][utt_id]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", default="data/librispeech_raw/LibriSpeech/test-clean")
    p.add_argument("--max-utterances", type=int, default=200)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--out", default="eval/results/qwen3asr_pkg.json")
    args = p.parse_args()

    logger.info("Loading %s…", args.model)
    from qwen_asr import Qwen3ASRModel
    t0 = time.time()
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        cache_dir="data/qwen3-asr-0.6b-pkg",
        max_inference_batch_size=4,
        max_new_tokens=256,
    )
    logger.info("Loaded in %.1f s", time.time() - t0)

    pairs = list(iter_librispeech_files(args.split_dir))[: args.max_utterances]
    logger.info("Eval over %d utterances", len(pairs))

    refs, hyps = [], []
    for i, (flac, ref) in enumerate(tqdm(pairs, desc="transcribe")):
        audio, sr = sf.read(flac, dtype="float32")
        results = model.transcribe(audio=(np.array(audio, dtype=np.float32), sr))
        hyp = results[0].text if results else ""
        refs.append(ref)
        hyps.append(hyp)
        if i < 3:
            logger.info("REF[%d]: %s", i, ref[:90])
            logger.info("HYP[%d]: %s", i, hyp[:90])

    overall_wer = wer(" \n".join(hyps), " \n".join(refs))
    logger.info("WER: %.2f %% (over %d utt)", overall_wer * 100, len(refs))
    logger.info("Published paper baseline: 2.11 % on test-clean")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n_utterances": len(refs),
        "wer_final": overall_wer * 100,
        "model": args.model,
        "paper_baseline_wer": 2.11,
        "split": args.split_dir,
    }, indent=2))


if __name__ == "__main__":
    main()
