"""Build the dictation probe sets (research/74: does the model hold through
dictation pauses, and does a profile context fix spelled names/emails?).

Probe A — digit dictation (endpointing):
  50 ten-digit US-style phone numbers (3-3-4 groups) assembled from FSDD
  recordings of held-out speakers (george, lucas), 8 kHz -> 16 kHz.
  Within-group gaps 0.05-0.25 s; between-group pauses 0.6-1.2 s (the case a
  silence timeout cannot survive); trailing silence 2.0 s.
  Metadata records every pause window and the end of the last digit, so
  premature vs final fires are scored exactly.

Probe B — spelled names + emails (recognition, +/- context):
  40 uncommon names ("My name is X, spelled K, O, ...") and 40 emails (half
  spelled letter-by-letter, half spoken naturally), synthesized with edge-tts
  (en-US-Jenny/Guy — voices held out from any training synthesis).
  Each item carries a profile CTX ("User profile — name: ...; email: ...")
  and a distractor CTX (another item's profile).

Usage (from repo root):
    .venv/bin/python eval/build_dictation_probes.py --out data/probes
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)
SR = 16000

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
               "eight", "nine"]
PROBE_FSDD_SPEAKERS = ["george", "lucas"]          # held out from training data
TRAIN_FSDD_SPEAKERS = ["jackson", "nicolas", "theo", "yweweler"]

PROBE_VOICES = ["en-US-JennyNeural", "en-US-GuyNeural"]   # held out from training

# 40 probe identities: uncommon / ambiguously-spelled surnames. The TRAINING
# list (build_v10_training_data.py) is disjoint from this one.
PROBE_PEOPLE = [
    ("Anna", "Kowalski"), ("Marcus", "Szymanski"), ("Linh", "Nguyen"),
    ("Siobhan", "Gallagher"), ("Erik", "Bjornstad"), ("Priya", "Venkatesan"),
    ("Declan", "O'Shaughnessy"), ("Mireille", "Beauchamp"), ("Tomasz", "Wojciechowski"),
    ("Aoife", "Ni Bhriain"), ("Henrik", "Kjaergaard"), ("Ximena", "Izquierdo"),
    ("Bartholomew", "Featherstonhaugh"), ("Saoirse", "Caulfield"), ("Piotr", "Blaszczyk"),
    ("Ingrid", "Thorvaldsen"), ("Rhys", "Llewellyn"), ("Katarzyna", "Zielinska"),
    ("Eoin", "Mac Giolla"), ("Marguerite", "Duchesne"), ("Sven", "Oskarsson"),
    ("Niamh", "Whelan"), ("Casimir", "Przybylski"), ("Gwendolyn", "Postlethwaite"),
    ("Lars", "Sondergaard"), ("Roisin", "Kavanagh"), ("Zbigniew", "Grzegorczyk"),
    ("Maeve", "Fitzwilliam"), ("Anders", "Vestergaard"), ("Orlaith", "Brennan"),
    ("Wojtek", "Szczepanski"), ("Isolde", "Rutherford"), ("Bjorn", "Haraldsen"),
    ("Clodagh", "Meehan"), ("Stanislaw", "Wisniewski"), ("Imogen", "Winterbourne"),
    ("Gunnar", "Solheim"), ("Aisling", "Donohue"), ("Krzysztof", "Jablonski"),
    ("Beatrix", "Van Der Meulen"),
]
DOMAINS = ["pine.ai", "gmail.com", "outlook.com", "meridianhealth.org", "acmecorp.io"]


def domain_spoken(domain: str) -> str:
    """pine.ai -> 'pine dot A I'; gmail.com -> 'gmail dot com'."""
    host, tld = domain.rsplit(".", 1)
    tld_sp = " ".join(tld.upper()) if len(tld) <= 2 else tld
    host_sp = host.replace("-", " dash ")
    return f"{host_sp} dot {tld_sp}"


def email_of(first: str, last: str, domain: str) -> str:
    def norm(s: str) -> str:
        return "".join(c for c in s.lower() if c.isalpha())
    return f"{norm(first)}.{norm(last)}@{domain}"


def profile_ctx(first: str, last: str, email: str, rng: random.Random) -> str:
    phone = " ".join(rng.choice("0123456789") for _ in range(10))
    return (f"User profile — name: {first} {last}; email: {email}; "
            f"phone: {phone}; customer since {rng.randint(2019, 2025)}.")


def spelled(word: str) -> str:
    letters = [c.upper() for c in word if c.isalpha()]
    return ", ".join(letters)


# ------------------------------------------------------------------ Probe A

def load_fsdd(fsdd_dir: Path, speakers: list[str]) -> dict[int, list[np.ndarray]]:
    import librosa
    pool: dict[int, list[np.ndarray]] = {d: [] for d in range(10)}
    for wav in sorted(fsdd_dir.glob("*.wav")):
        digit_s, spk, _ = wav.stem.split("_")
        if spk not in speakers:
            continue
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        # normalize peak to a sane level
        peak = np.abs(audio).max() or 1.0
        pool[int(digit_s)].append((audio * (0.25 / peak)).astype(np.float32))
    return pool


def build_digit_sequence(pool, digits: list[int], rng: random.Random,
                         groups=(3, 3, 4), pause_range=(0.6, 1.2),
                         gap_range=(0.05, 0.25), tail_s=2.0):
    """Concatenate FSDD digits into a grouped sequence; return (audio, meta)."""
    parts: list[np.ndarray] = []
    t = 0.0
    digit_end_times: list[float] = []
    pauses: list[tuple[float, float]] = []       # (start, end) of group pauses
    i = 0
    for gi, glen in enumerate(groups):
        for j in range(glen):
            clip = rng.choice(pool[digits[i]])
            parts.append(clip)
            t += len(clip) / SR
            digit_end_times.append(t)
            i += 1
            if j < glen - 1:
                gap = rng.uniform(*gap_range)
                parts.append(np.zeros(int(gap * SR), dtype=np.float32))
                t += gap
        if gi < len(groups) - 1:
            pause = rng.uniform(*pause_range)
            pauses.append((t, t + pause))
            parts.append(np.zeros(int(pause * SR), dtype=np.float32))
            t += pause
    last_speech_end = t
    parts.append(np.zeros(int(tail_s * SR), dtype=np.float32))
    audio = np.concatenate(parts)
    meta = {
        "digits": digits,
        "text": " ".join(DIGIT_WORDS[d] for d in digits),
        "groups": list(groups),
        "pauses": pauses,
        "last_speech_end_s": round(last_speech_end, 3),
        "duration_s": round(len(audio) / SR, 3),
    }
    return audio, meta


def build_probe_a(fsdd_dir: Path, out_dir: Path, n_items: int, seed: int):
    rng = random.Random(seed)
    pool = load_fsdd(fsdd_dir, PROBE_FSDD_SPEAKERS)
    logger.info("FSDD probe pool: %s clips",
                {d: len(v) for d, v in pool.items()})
    wav_dir = out_dir / "digit_wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for k in range(n_items):
        digits = [rng.randrange(10) for _ in range(10)]
        audio, meta = build_digit_sequence(pool, digits, rng)
        path = wav_dir / f"digit_{k:03d}.wav"
        sf.write(path, audio, SR)
        meta["wav"] = str(path)
        meta["id"] = k
        items.append(meta)
    (out_dir / "digit_probe.json").write_text(json.dumps(items, indent=1))
    logger.info("Probe A: %d sequences -> %s", len(items), out_dir / "digit_probe.json")


# ------------------------------------------------------------------ Probe B

async def tts_save(text: str, voice: str, mp3_path: Path):
    import edge_tts
    for attempt in range(4):
        try:
            await edge_tts.Communicate(text, voice=voice).save(str(mp3_path))
            return
        except Exception as e:                                    # noqa: BLE001
            logger.warning("edge-tts retry %d for %s: %s", attempt, mp3_path.name, e)
            await asyncio.sleep(2 + 3 * attempt)
    raise RuntimeError(f"edge-tts failed for {mp3_path}")


def mp3_to_wav(mp3_path: Path, wav_path: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "quiet", "-i", str(mp3_path),
         "-ar", str(SR), "-ac", "1", str(wav_path)], check=True)


def build_probe_b(out_dir: Path, seed: int):
    rng = random.Random(seed)
    wav_dir = out_dir / "spelled_wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for k, (first, last) in enumerate(PROBE_PEOPLE):
        domain = rng.choice(DOMAINS)
        email = email_of(first, last, domain)
        voice = PROBE_VOICES[k % len(PROBE_VOICES)]
        # --- name item (always spelled)
        name_utt = f"My name is {first} {last}. That is spelled {spelled(last)}."
        items.append({
            "kind": "name", "voice": voice, "utt": name_utt,
            "first": first, "last": last, "email": email,
            "target": "".join(c for c in last.lower() if c.isalpha()),
            "ctx": profile_ctx(first, last, email, rng),
        })
        # --- email item (half spelled, half natural)
        local = email.split("@")[0]
        if k % 2 == 0:
            local_sp = local.replace(".", " dot ")
            email_utt = (f"My email address is {local_sp}, that is "
                         f"{spelled(local.split('.')[0])}, dot, "
                         f"{spelled(local.split('.')[1])}, at {domain_spoken(domain)}.")
            style = "spelled"
        else:
            email_utt = (f"My email address is {local.replace('.', ' dot ')} "
                         f"at {domain_spoken(domain)}.")
            style = "natural"
        items.append({
            "kind": "email", "style": style, "voice": voice, "utt": email_utt,
            "first": first, "last": last, "email": email,
            "target": email,
            "ctx": profile_ctx(first, last, email, rng),
        })
    # distractor ctx = profile of another identity (offset by 7)
    n = len(items)
    for i, it in enumerate(items):
        it["ctx_distractor"] = items[(i + 14) % n]["ctx"]
        it["id"] = i

    async def synth_all():
        sem = asyncio.Semaphore(4)
        async def one(it):
            mp3 = wav_dir / f"sp_{it['id']:03d}.mp3"
            wav = wav_dir / f"sp_{it['id']:03d}.wav"
            it["wav"] = str(wav)
            if wav.exists():
                return
            async with sem:
                await tts_save(it["utt"], it["voice"], mp3)
            mp3_to_wav(mp3, wav)
            mp3.unlink(missing_ok=True)
        await asyncio.gather(*[one(it) for it in items])

    asyncio.run(synth_all())
    (out_dir / "spelled_probe.json").write_text(json.dumps(items, indent=1))
    logger.info("Probe B: %d items -> %s", len(items), out_dir / "spelled_probe.json")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--fsdd", default="data/fsdd/recordings")
    p.add_argument("--out", default="data/probes")
    p.add_argument("--n-digit", type=int, default=50)
    p.add_argument("--seed", type=int, default=74)
    p.add_argument("--skip-a", action="store_true")
    p.add_argument("--skip-b", action="store_true")
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_a:
        build_probe_a(Path(args.fsdd), out_dir, args.n_digit, args.seed)
    if not args.skip_b:
        build_probe_b(out_dir, args.seed + 1)


if __name__ == "__main__":
    main()
