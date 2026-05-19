"""Build semantic-endpoint training data from LibriSpeech concatenation.

Each example: one of these schemas
  - SINGLE: solo utterance, transcript ends with <EAGER_END_SPEECH><END_SPEECH>
  - DOUBLE: utt_A + 1.0-2.0s silence + utt_B, transcript has end markers
    between and after both utterances
  - DISFLUENCY: utt_A's audio split with a 0.3-0.8s internal pause (mid-sentence),
    transcript has NO end markers in the middle — just at the very end

Output: data/semantic_endpoint/data.pt with
  - audio: list of np.float32 (variable length, 16 kHz)
  - text: list of str (the full target transcript including markers)
  - schema: list of {"single", "double", "disfluency"}
  - segments: list of dict (word/segment info for evaluation)

Usage:
    python -m eval.semantic_endpoint_data --max-pairs 500
"""

from __future__ import annotations
import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from eval.run_qwen3asr_pkg import iter_librispeech_files

logger = logging.getLogger(__name__)


END_TOK = "<END_SPEECH>"
EAGER_TOK = "<EAGER_END_SPEECH>"


def gen_single(audio: np.ndarray, text: str) -> dict:
    return {
        "audio": audio,
        "text": f"{text} {EAGER_TOK}{END_TOK}",
        "schema": "single",
        "audio_end_s": len(audio) / 16000,
    }


def gen_double(audio_a: np.ndarray, text_a: str,
               audio_b: np.ndarray, text_b: str,
               gap_s: float, rng: random.Random) -> dict:
    sr = 16000
    gap = np.zeros(int(gap_s * sr), dtype=np.float32)
    cat = np.concatenate([audio_a, gap, audio_b])
    # Markers after utt_A (within the silence) and after utt_B
    text = (
        f"{text_a} {EAGER_TOK}{END_TOK} "
        f"{text_b} {EAGER_TOK}{END_TOK}"
    )
    return {
        "audio": cat,
        "text": text,
        "schema": "double",
        "audio_end_s_a": len(audio_a) / sr,
        "gap_start_s": len(audio_a) / sr,
        "gap_end_s": (len(audio_a) + len(gap)) / sr,
        "audio_end_s": len(cat) / sr,
    }


def gen_disfluency(audio: np.ndarray, text: str,
                    pause_s: float, rng: random.Random) -> dict:
    """Insert a short internal pause without an end marker — the LM should
    not predict END_SPEECH during this pause because the speaker continues.
    """
    sr = 16000
    pause = np.zeros(int(pause_s * sr), dtype=np.float32)
    # Pick a split point at ~30-70% through the audio
    n_samples = len(audio)
    split = int(n_samples * rng.uniform(0.3, 0.7))
    cat = np.concatenate([audio[:split], pause, audio[split:]])
    # Transcript is UNCHANGED — endpoint markers only at the actual end
    return {
        "audio": cat,
        "text": f"{text} {EAGER_TOK}{END_TOK}",
        "schema": "disfluency",
        "audio_end_s": len(cat) / sr,
        "pause_start_s": split / sr,
        "pause_end_s": (split + len(pause)) / sr,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", default="data/librispeech_raw/LibriSpeech/train-clean-100")
    p.add_argument("--max-pairs", type=int, default=300,
                    help="Number of DOUBLE concat pairs to build")
    p.add_argument("--max-disfluency", type=int, default=200,
                    help="Number of DISFLUENCY examples to build")
    p.add_argument("--max-single", type=int, default=200,
                    help="Number of SINGLE examples to keep")
    p.add_argument("--out", default="data/semantic_endpoint/data.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)

    logger.info("Loading LibriSpeech…")
    pairs = list(iter_librispeech_files(args.split_dir))
    rng.shuffle(pairs)
    # Drop too-short utts (need ≥ 2 s) and too-long (> 25 s)
    candidates = []
    for flac, ref in pairs:
        try:
            info = sf.info(flac)
            dur = info.frames / info.samplerate
        except Exception:
            continue
        if 2.0 <= dur <= 25.0:
            candidates.append((flac, ref))
    logger.info("Loaded %d viable utterances", len(candidates))

    # Pre-load audio for first N (memory bound — load lazily for double-pairs)
    def load(idx):
        flac, ref = candidates[idx]
        audio, sr = sf.read(flac, dtype="float32")
        if sr != 16000:
            return None
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        return audio, ref

    examples = []
    used: set[int] = set()

    # Single
    logger.info("Building %d single examples", args.max_single)
    for i in tqdm(range(min(args.max_single, len(candidates))), desc="single"):
        if i in used: continue
        loaded = load(i)
        if loaded is None: continue
        audio, ref = loaded
        examples.append(gen_single(audio, ref))
        used.add(i)

    # Double
    logger.info("Building %d double examples", args.max_pairs)
    n_done = 0
    attempts = 0
    while n_done < args.max_pairs and attempts < args.max_pairs * 3:
        attempts += 1
        a = rng.randrange(len(candidates))
        b = rng.randrange(len(candidates))
        if a == b: continue
        la = load(a)
        lb = load(b)
        if la is None or lb is None: continue
        gap_s = rng.uniform(0.7, 2.0)
        examples.append(gen_double(la[0], la[1], lb[0], lb[1], gap_s, rng))
        n_done += 1

    # Disfluency
    logger.info("Building %d disfluency examples", args.max_disfluency)
    for i in tqdm(range(min(args.max_disfluency, len(candidates))), desc="disfluency"):
        idx = rng.randrange(len(candidates))
        loaded = load(idx)
        if loaded is None: continue
        pause_s = rng.uniform(0.3, 0.7)  # Short — under typical end-of-turn threshold
        examples.append(gen_disfluency(loaded[0], loaded[1], pause_s, rng))

    logger.info("Total %d examples (single=%d, double=%d, disfluency=%d)",
                 len(examples),
                 sum(1 for e in examples if e["schema"] == "single"),
                 sum(1 for e in examples if e["schema"] == "double"),
                 sum(1 for e in examples if e["schema"] == "disfluency"))

    # Persist as pt
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Use float32 numpy stored directly (more compact than pickled-Python lists)
    torch.save(examples, args.out)
    logger.info("Saved to %s", args.out)


if __name__ == "__main__":
    main()
