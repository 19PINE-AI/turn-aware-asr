"""Walk an extracted LibriSpeech directory and pre-compute mel features.

Expects the openslr layout:
    data/librispeech_raw/LibriSpeech/<split>/<spk>/<chap>/<utt-id>.flac
                                                       /<spk>-<chap>.trans.txt

Outputs:
    data/librispeech/<split>/data.pt  — { 'audio': [Tensor], 'mel': [Tensor], 'text': [str] }

Usage:
    python -m scripts.prepare_librispeech --split test-clean
    python -m scripts.prepare_librispeech --split train-clean-100 --max-utterances 5000
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path

import torch
import soundfile as sf
import torchaudio
from tqdm import tqdm

from src.features import log_mel

logger = logging.getLogger(__name__)


def iter_split(split_dir: Path):
    """Yield (utt_id, flac_path, transcript) for all utterances in a split."""
    for spk_dir in sorted(split_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        for chap_dir in sorted(spk_dir.iterdir()):
            if not chap_dir.is_dir():
                continue
            trans_files = list(chap_dir.glob("*.trans.txt"))
            if not trans_files:
                continue
            transcripts = {}
            with open(trans_files[0]) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    utt_id, _, text = line.partition(" ")
                    transcripts[utt_id] = text
            for flac in sorted(chap_dir.glob("*.flac")):
                utt_id = flac.stem
                if utt_id in transcripts:
                    yield utt_id, flac, transcripts[utt_id]


def load_audio_16k(path: Path) -> torch.Tensor:
    audio, sr = sf.read(str(path), dtype="float32")
    audio = torch.from_numpy(audio)
    if audio.dim() == 2:
        audio = audio.mean(dim=-1)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    return audio


def prepare(split: str, raw_root: Path, out_dir: Path, max_utterances: int | None = None,
            skip_audio: bool = True):
    """Walk the LibriSpeech tree, compute mels, save to data.pt.

    skip_audio=True (default): only save mel + text. Cuts file size and
    RAM use ~2× since training only ever reads mel.
    """
    split_dir = raw_root / "LibriSpeech" / split
    if not split_dir.exists():
        raise FileNotFoundError(f"{split_dir} not found. Run scripts/download_librispeech.sh {split}")

    audios: list = []
    mels: list = []
    texts: list = []
    skipped = 0
    for i, (utt_id, flac, text) in enumerate(tqdm(list(iter_split(split_dir)), desc=f"prep:{split}")):
        if max_utterances is not None and i - skipped >= max_utterances:
            break
        try:
            audio = load_audio_16k(flac)
            if audio.shape[0] > 16000 * 30:
                skipped += 1
                continue
            mel = log_mel(audio)
        except Exception as e:
            logger.warning("Skipping %s: %s", utt_id, e)
            skipped += 1
            continue
        if not skip_audio:
            audios.append(audio)
        mels.append(mel.contiguous())
        texts.append(text)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.pt"
    payload = {"mel": mels, "text": texts}
    if not skip_audio:
        payload["audio"] = audios
    torch.save(payload, out_path)
    logger.info("Saved %d utterances to %s (skipped %d)", len(mels), out_path, skipped)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, help="e.g. test-clean")
    p.add_argument("--raw-root", default="data/librispeech_raw")
    p.add_argument("--out-root", default="data/librispeech")
    p.add_argument("--max-utterances", type=int, default=None)
    args = p.parse_args()
    out = Path(args.out_root) / args.split
    prepare(args.split, Path(args.raw_root), out, args.max_utterances)


if __name__ == "__main__":
    main()
