"""Download LibriSpeech splits and pre-compute mel features + token ids.

Outputs per-split shards under data/librispeech/<split>/ with:
  audio.pt    — list of waveform tensors (float32, 16 kHz)
  mel.pt      — list of mel tensors (128, T) precomputed
  text.pt     — list of str transcripts
  token_ids.pt — list of LongTensor token id sequences

Run:
  python -m scripts.prepare_librispeech --split test-clean
  python -m scripts.prepare_librispeech --split dev-clean
  python -m scripts.prepare_librispeech --split train-clean-100 --max-utterances 5000
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.features import log_mel

logger = logging.getLogger(__name__)


def prepare(split: str, out_dir: Path, max_utterances: int | None = None):
    logger.info("Loading split %s from openslr/librispeech_asr", split)
    ds = load_dataset(
        "openslr/librispeech_asr",
        split=split,
        streaming=False,
    )
    if max_utterances is not None and max_utterances < len(ds):
        ds = ds.select(range(max_utterances))
    logger.info("Got %d utterances", len(ds))

    out_dir.mkdir(parents=True, exist_ok=True)
    audios, mels, texts = [], [], []
    skipped = 0
    for ex in tqdm(ds, desc=f"prep:{split}"):
        audio = torch.tensor(ex["audio"]["array"], dtype=torch.float32)
        sr = ex["audio"]["sampling_rate"]
        if sr != 16000:
            # downsample
            import torchaudio
            audio = torchaudio.functional.resample(audio, sr, 16000)
        if audio.shape[0] > 16000 * 30:
            skipped += 1
            continue   # >30 s, skip (Qwen3-ASR pads to 30 s; longer needs chunking)
        try:
            mel = log_mel(audio)  # (T_frames, 128) float32
        except Exception as e:
            logger.warning("Skipping utt due to feature error: %s", e)
            skipped += 1
            continue
        audios.append(audio)
        mels.append(mel)
        texts.append(ex["text"])

    logger.info("Kept %d / skipped %d", len(audios), skipped)
    torch.save({"audio": audios, "mel": mels, "text": texts}, out_dir / "data.pt")
    logger.info("Saved %s", out_dir / "data.pt")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--out-root", default="data/librispeech")
    p.add_argument("--max-utterances", type=int, default=None)
    args = p.parse_args()
    out = Path(args.out_root) / args.split.replace(".", "-")
    prepare(args.split, out, args.max_utterances)


if __name__ == "__main__":
    main()
