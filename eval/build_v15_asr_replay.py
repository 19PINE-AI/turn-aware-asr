"""v15 = v12 pool + plain-ASR replay examples, to attack the offline-WER
regression (narrow-schema drift; the 'standard mitigation' the paper names).

Plain-ASR example = LibriSpeech audio + its base-model transcript, NO marker,
NO trailing-silence tail, empty context. This is the same shape as
complete_nosil but sourced from a wider/longer LS draw, to broaden the
transcription distribution the fine-tune sees.

Risk: too much no-marker plain ASR could dilute the fire signal and hurt
endpoint recall; we add a moderate amount and let the checkpoint sweep find
the balance. Usage:
    .venv/bin/python eval/build_v15_asr_replay.py --n-asr 2500
"""
from __future__ import annotations
import argparse, json, logging, random
from pathlib import Path
import numpy as np, soundfile as sf, torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.build_v10_training_data import trim_edges, ex, SR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v12", default="data/semantic_endpoint_v12/data.pt")
    p.add_argument("--cache", default="data/semantic_endpoint_v9/ls_transcripts.json")
    p.add_argument("--out", default="data/semantic_endpoint_v15")
    p.add_argument("--n-asr", type=int, default=2500)
    p.add_argument("--min-dur", type=float, default=2.5)
    p.add_argument("--max-dur", type=float, default=14.0)  # longer than the 9s endpoint pool
    p.add_argument("--seed", type=int, default=15)
    args = p.parse_args()
    rng = random.Random(args.seed)

    base = torch.load(args.v12, weights_only=False)
    logger.info("v12 pool: %d examples", len(base))

    cache = json.loads(Path(args.cache).read_text())
    items = list(cache.items())
    rng.shuffle(items)
    added = []
    for flac, text in items:
        if len(added) >= args.n_asr:
            break
        if not Path(flac).exists() or not text.strip():
            continue
        try:
            audio, sr = sf.read(flac, dtype="float32")
        except Exception:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        dur = len(audio) / SR
        if not (args.min_dur <= dur <= args.max_dur):
            continue
        audio = trim_edges(audio, thresh=5e-3, keep_ms=60)   # no trailing tail -> no marker
        added.append(ex(audio, text.strip(), "asr_plain", "ls_replay"))
    logger.info("added %d plain-ASR examples", len(added))

    examples = list(base) + added
    rng.shuffle(examples)
    from collections import Counter
    logger.info("v15 schema counts: %s", dict(Counter(e["schema"] for e in examples)))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(examples, out / "data.pt")
    (out / "build_meta.json").write_text(json.dumps(
        {"base": args.v12, "n_base": len(base), "n_asr_plain": len(added),
         "schema_counts": dict(Counter(e["schema"] for e in examples))}, indent=1))
    logger.info("wrote %s (%d examples)", out / "data.pt", len(examples))


if __name__ == "__main__":
    main()
