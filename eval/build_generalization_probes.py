"""Dictation-generalization probes (paper Limitations / dictation-generalization).

Turns the "other enumerations are untrained" assertion into a measured zero-shot
transfer number. The released model was trained ONLY on 10-digit phone numbers
(3-3-4 groups). Here we test enumeration shapes it never saw:

  * card    — 16-digit card numbers, 4-4-4-4 groups (FSDD digits, held-out
              speakers george/lucas) — pure length/grouping transfer.
  * phone13 — 13-digit sequences, 3-3-3-4 groups — intermediate control.
  * address — TTS street addresses ("The address is seven four two Maple
              Avenue, apartment three B") — digits embedded in words, a
              genuinely different shape. Trailing silence trimmed, 2.0 s tail
              appended, last_speech_end computed from the trim.

Output matches data/probes/digit_probe.json exactly (same fields), so
`eval/dictation_probe_eval.py --probes <out> --skip-b` scores it unchanged.

Usage:
  .venv/bin/python eval/build_generalization_probes.py --shape card --out data/probes_card
  .venv/bin/python eval/build_generalization_probes.py --shape address --out data/probes_addr
"""
from __future__ import annotations
import argparse, asyncio, json, logging, random, subprocess
from pathlib import Path
import numpy as np, soundfile as sf
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.build_dictation_probes import (  # noqa: E402
    SR, DIGIT_WORDS, PROBE_FSDD_SPEAKERS, PROBE_VOICES,
    load_fsdd, build_digit_sequence, tts_save, mp3_to_wav)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SHAPES = {
    "card": {"n_digits": 16, "groups": (4, 4, 4, 4)},
    "phone13": {"n_digits": 13, "groups": (3, 3, 3, 4)},
}

STREETS = ["Maple Avenue", "Oak Street", "Cedar Lane", "Birch Boulevard",
           "Elm Court", "Willow Way", "Pine Ridge Road", "Ashford Terrace",
           "Kingsley Drive", "Harborview Crescent"]


def build_digit_shape(fsdd_dir: Path, out_dir: Path, shape: str, n_items: int, seed: int):
    cfg = SHAPES[shape]
    rng = random.Random(seed)
    pool = load_fsdd(fsdd_dir, PROBE_FSDD_SPEAKERS)
    wav_dir = out_dir / "digit_wavs"; wav_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for k in range(n_items):
        digits = [rng.randrange(10) for _ in range(cfg["n_digits"])]
        audio, meta = build_digit_sequence(pool, digits, rng, groups=cfg["groups"])
        path = wav_dir / f"{shape}_{k:03d}.wav"
        sf.write(path, audio, SR)
        meta["wav"] = str(path); meta["id"] = k; meta["shape"] = shape
        items.append(meta)
    (out_dir / "digit_probe.json").write_text(json.dumps(items, indent=1))
    logger.info("%s: %d sequences (%d digits, groups %s) -> %s",
                shape, len(items), cfg["n_digits"], cfg["groups"], out_dir / "digit_probe.json")


def trim_tail(audio: np.ndarray, thresh: float = 3e-3, win_ms: int = 30) -> float:
    """Return end-of-speech time (s): last window whose RMS exceeds thresh."""
    w = int(win_ms / 1000 * SR)
    last = len(audio)
    for start in range(len(audio) - w, 0, -w):
        if float(np.sqrt(np.mean(audio[start:start + w] ** 2))) >= thresh:
            last = start + w; break
    return last / SR


def build_address(out_dir: Path, n_items: int, seed: int):
    rng = random.Random(seed)
    wav_dir = out_dir / "digit_wavs"; wav_dir.mkdir(parents=True, exist_ok=True)
    specs = []
    for k in range(n_items):
        hnum = [rng.randrange(10) for _ in range(rng.choice([3, 4]))]
        unit = rng.randrange(10)
        letter = rng.choice("ABCD")
        street = rng.choice(STREETS)
        digit_words = " ".join(DIGIT_WORDS[d] for d in hnum)
        utt = (f"The delivery address is {digit_words} {street}, "
               f"apartment {DIGIT_WORDS[unit]} {letter}.")
        specs.append({"id": k, "shape": "address", "utt": utt,
                      "voice": PROBE_VOICES[k % len(PROBE_VOICES)],
                      "digits": hnum + [unit],
                      "text": " ".join(DIGIT_WORDS[d] for d in hnum + [unit])})

    async def synth():
        sem = asyncio.Semaphore(4)
        async def one(sp):
            mp3 = wav_dir / f"addr_{sp['id']:03d}.mp3"
            wav = wav_dir / f"addr_{sp['id']:03d}.wav"
            async with sem:
                await tts_save(sp["utt"], sp["voice"], mp3)
            mp3_to_wav(mp3, wav); mp3.unlink(missing_ok=True)
            speech_audio, _ = sf.read(wav, dtype="float32")
            if speech_audio.ndim > 1:
                speech_audio = speech_audio.mean(axis=1)
            last_end = trim_tail(speech_audio)
            tail = np.zeros(int(2.0 * SR), dtype=np.float32)
            full = np.concatenate([speech_audio[:int(last_end * SR)], tail])
            sf.write(wav, full, SR)
            sp["wav"] = str(wav)
            sp["last_speech_end_s"] = round(last_end, 3)
            sp["duration_s"] = round(len(full) / SR, 3)
            sp["pauses"] = []
        await asyncio.gather(*[one(sp) for sp in specs])

    asyncio.run(synth())
    (out_dir / "digit_probe.json").write_text(json.dumps(specs, indent=1))
    logger.info("address: %d utterances -> %s", len(specs), out_dir / "digit_probe.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", required=True, choices=["card", "phone13", "address"])
    p.add_argument("--fsdd", default="data/fsdd/recordings")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=93)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.shape == "address":
        build_address(out, args.n, args.seed)
    else:
        build_digit_shape(Path(args.fsdd), out, args.shape, args.n, args.seed)


if __name__ == "__main__":
    main()
