"""Generate v2 of semantic-endpoint training data.

Improvements over v1 (eval/semantic_endpoint_data.py):
  - 2500 examples (was 700): 500 single + 1500 double + 500 disfluency
  - Target transcripts are produced by running the BASE Qwen3-ASR on each
    audio first, then appending markers. This fixes the v1 WER regression
    (model was learning uppercase LibriSpeech style from references).
  - Logs progress aggressively so it's resumable

Usage:
  python -m eval.semantic_endpoint_data_v2 \\
      --n-single 500 --n-double 1500 --n-disfluency 500
"""

from __future__ import annotations
import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from eval.run_qwen3asr_pkg import iter_librispeech_files

logger = logging.getLogger(__name__)

END_TOK = "<END_SPEECH>"
EAGER_TOK = "<EAGER_END_SPEECH>"


def transcribe_base(asr_model, audio: np.ndarray, sr: int = 16000) -> str:
    """Use the base Qwen3-ASR model to transcribe audio, returning the
    natural cased+punctuated text the model itself prefers.
    """
    results = asr_model.transcribe(audio=(audio, sr))
    if not results:
        return ""
    return results[0].text or ""


def gen_single(audio: np.ndarray, base_text: str) -> dict:
    return {
        "audio": audio,
        "text": f"{base_text} {EAGER_TOK}{END_TOK}",
        "schema": "single",
        "audio_end_s": len(audio) / 16000,
    }


def gen_double(audio_a: np.ndarray, text_a: str,
               audio_b: np.ndarray, text_b: str,
               gap_s: float, rng: random.Random) -> dict:
    sr = 16000
    gap = np.zeros(int(gap_s * sr), dtype=np.float32)
    cat = np.concatenate([audio_a, gap, audio_b])
    text = f"{text_a} {EAGER_TOK}{END_TOK} {text_b} {EAGER_TOK}{END_TOK}"
    return {
        "audio": cat,
        "text": text,
        "schema": "double",
        "audio_end_s_a": len(audio_a) / sr,
        "gap_start_s": len(audio_a) / sr,
        "gap_end_s": (len(audio_a) + len(gap)) / sr,
        "audio_end_s": len(cat) / sr,
    }


def gen_disfluency(audio: np.ndarray, base_text: str,
                    pause_s: float, rng: random.Random) -> dict:
    sr = 16000
    pause = np.zeros(int(pause_s * sr), dtype=np.float32)
    n_samples = len(audio)
    split = int(n_samples * rng.uniform(0.3, 0.7))
    cat = np.concatenate([audio[:split], pause, audio[split:]])
    return {
        "audio": cat,
        "text": f"{base_text} {EAGER_TOK}{END_TOK}",
        "schema": "disfluency",
        "audio_end_s": len(cat) / sr,
        "pause_start_s": split / sr,
        "pause_end_s": (split + len(pause)) / sr,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", default="data/librispeech_raw/LibriSpeech/train-clean-100")
    p.add_argument("--n-single", type=int, default=500)
    p.add_argument("--n-double", type=int, default=1500)
    p.add_argument("--n-disfluency", type=int, default=500)
    p.add_argument("--out", default="data/semantic_endpoint_v2/data.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-dur-s", type=float, default=9.0,
                    help="Per-utt max duration; doubles can stack to ~20s")
    p.add_argument("--transcripts-cache", default="data/semantic_endpoint_v2/base_transcripts.json",
                    help="Cache base-model transcripts to avoid re-running")
    args = p.parse_args()

    rng = random.Random(args.seed)

    logger.info("Loading LibriSpeech…")
    pairs = list(iter_librispeech_files(args.split_dir))
    rng.shuffle(pairs)
    candidates = []
    for flac, ref in pairs:
        try:
            info = sf.info(flac)
            dur = info.frames / info.samplerate
        except Exception:
            continue
        if 2.5 <= dur <= args.max_dur_s:
            candidates.append((flac, ref))
    logger.info("Filtered to %d viable utterances (2.5 .. %.1f s)",
                 len(candidates), args.max_dur_s)

    # We need transcripts for: n_single + n_double*2 (each double needs both)
    # + n_disfluency. Take min unique-utts to cover.
    needed = max(
        args.n_single,
        args.n_double * 2,  # rough upper bound; we sample with replacement
        args.n_disfluency,
    )
    needed = min(needed, len(candidates))
    candidates = candidates[: needed + 100]  # buffer
    logger.info("Will transcribe %d unique utterances with base Qwen3-ASR…", len(candidates))

    # Load base model
    from qwen_asr import Qwen3ASRModel
    asr = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        cache_dir="data/qwen3-asr-0.6b-pkg",
        max_inference_batch_size=4,
        max_new_tokens=256,
    )

    # Transcribe candidates (with cache)
    cache_path = Path(args.transcripts_cache)
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        logger.info("Loaded %d cached transcripts", len(cache))

    new_count = 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for flac, _ in tqdm(candidates, desc="base-transcribe"):
        if flac in cache:
            continue
        try:
            audio, sr = sf.read(flac, dtype="float32")
            if sr != 16000:
                continue
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)
            text = transcribe_base(asr, audio.astype(np.float32), sr)
            cache[flac] = text
            new_count += 1
            if new_count % 50 == 0:
                cache_path.write_text(json.dumps(cache))
        except Exception as e:
            logger.warning("transcribe failed for %s: %s", flac, e)
            cache[flac] = ""
    cache_path.write_text(json.dumps(cache))
    logger.info("Transcribed %d new utterances; %d total cached", new_count, len(cache))

    # Now build examples
    def load(flac):
        audio, sr = sf.read(flac, dtype="float32")
        if sr != 16000 or audio.ndim > 1:
            audio = audio.mean(axis=-1) if audio.ndim > 1 else audio
        return audio.astype(np.float32)

    # Filter candidates to those we successfully transcribed
    usable = [(f, ref) for f, ref in candidates if cache.get(f, "").strip()]
    rng.shuffle(usable)
    logger.info("%d usable utts (with non-empty base transcript)", len(usable))

    examples: list[dict] = []

    # SINGLE
    for flac, _ in tqdm(usable[: args.n_single], desc="single"):
        text = cache[flac]
        audio = load(flac)
        examples.append(gen_single(audio, text))

    # DISFLUENCY
    pool_offset = args.n_single
    for flac, _ in tqdm(usable[pool_offset : pool_offset + args.n_disfluency], desc="disfluency"):
        text = cache[flac]
        audio = load(flac)
        pause_s = rng.uniform(0.3, 0.7)
        examples.append(gen_disfluency(audio, text, pause_s, rng))

    # DOUBLE: sample pairs from the pool
    logger.info("Building %d double examples…", args.n_double)
    pool = usable
    n_done = 0
    pbar = tqdm(total=args.n_double, desc="double")
    while n_done < args.n_double:
        a = rng.choice(pool)
        b = rng.choice(pool)
        if a[0] == b[0]:
            continue
        try:
            audio_a = load(a[0])
            audio_b = load(b[0])
        except Exception:
            continue
        gap_s = rng.uniform(0.7, 2.0)
        examples.append(gen_double(audio_a, cache[a[0]], audio_b, cache[b[0]], gap_s, rng))
        n_done += 1
        pbar.update(1)
    pbar.close()

    logger.info(
        "Total %d examples (single=%d, double=%d, disfluency=%d)",
        len(examples),
        sum(1 for e in examples if e["schema"] == "single"),
        sum(1 for e in examples if e["schema"] == "double"),
        sum(1 for e in examples if e["schema"] == "disfluency"),
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(examples, args.out)
    logger.info("Saved to %s", args.out)


if __name__ == "__main__":
    main()
