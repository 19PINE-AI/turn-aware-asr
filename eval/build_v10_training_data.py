"""Build v10 training data = all v9 examples + dictation/context schemas.

New schemas (research/74: the completeness heuristic reads digit strings as
complete, so the causal model fires inside dictation pauses; and spelled
names/emails need the profile context to be spelled right):

  digit_hold    partial phone number (3 or 6 of 10 digits) + pause 0.5-1.5 s
                -> digits text, NO marker (the pattern predicts continuation)
  digit_fire    full 10 digits, 3-3-4 groups with 0.4-1.2 s pauses INSIDE,
                + tail 0.4-1.2 s -> digits text + M (pattern complete + silence)
  digit_nosil   full 10 digits, tail <= 0.1 s -> digits text, NO marker
                (the silence half of the rule still binds)
  spell_name    TTS "My name is X, spelled K, O, ..." + tail -> verbatim + M;
                50% carry a profile ctx containing the name
  spell_email   TTS spelled/natural email + tail -> transcript with the email
                in NORMALIZED written form (anna.kowalski@pine.ai) + M;
                50% carry a profile ctx containing the email
                (~15% of spell_* get no tail -> no marker: minimal pairs)

The minimal-pair discipline of v9 is kept: the SAME digit sequences appear as
digit_hold (truncated) and digit_fire (full); fire requires pattern-complete
AND silence — both prefix-observable.

FSDD speakers here are jackson/nicolas/theo/yweweler; the probe uses
george/lucas. TTS voices here exclude the probe's Jenny/Guy.

Usage:
    .venv/bin/python eval/build_v10_training_data.py \
        --v9 data/semantic_endpoint_v9/data.pt --out data/semantic_endpoint_v10
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
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.build_dictation_probes import (          # noqa: E402
    DIGIT_WORDS, TRAIN_FSDD_SPEAKERS, load_fsdd, build_digit_sequence,
    domain_spoken, email_of, spelled, tts_save, mp3_to_wav, DOMAINS,
)

logger = logging.getLogger(__name__)
SR = 16000
END_TOK = "<END_SPEECH>"
EAGER_TOK = "<EAGER_END_SPEECH>"
M = f"{EAGER_TOK}{END_TOK}"

TRAIN_VOICES = ["en-US-AriaNeural", "en-US-ChristopherNeural",
                "en-GB-SoniaNeural", "en-AU-NatashaNeural",
                "en-US-MichelleNeural", "en-GB-RyanNeural",
                # v13: wider voice variety to lift the spelled-email ceiling
                "en-US-EricNeural", "en-US-RogerNeural", "en-US-SteffanNeural",
                "en-US-AnaNeural", "en-CA-ClaraNeural", "en-CA-LiamNeural",
                "en-GB-LibbyNeural", "en-GB-MaisieNeural", "en-IE-ConnorNeural",
                "en-GB-ThomasNeural", "en-US-BrianNeural", "en-US-EmmaNeural"]

# Disjoint from PROBE_PEOPLE in build_dictation_probes.py.
TRAIN_PEOPLE = [
    ("Grainne", "McIlhenny"), ("Tadeusz", "Krawczyk"), ("Solveig", "Andresen"),
    ("Padraig", "Boyle"), ("Malgorzata", "Nowakowska"), ("Torsten", "Lindqvist"),
    ("Caitriona", "Hennessy"), ("Jerzy", "Pawlikowski"), ("Astrid", "Nygaard"),
    ("Fionnuala", "Sweeney"), ("Ryszard", "Dabrowski"), ("Freya", "Ostergaard"),
    ("Cormac", "Duggan"), ("Agnieszka", "Sokolowska"), ("Soren", "Mikkelsen"),
    ("Bronagh", "Sheridan"), ("Waldemar", "Cieslak"), ("Sigrid", "Halvorsen"),
    ("Diarmuid", "Keane"), ("Jolanta", "Wieczorek"), ("Magnus", "Engstrom"),
    ("Sinead", "Molloy"), ("Slawomir", "Urbanski"), ("Thea", "Johannessen"),
    ("Fergus", "Rafferty"), ("Bozena", "Kaczmarek"), ("Nils", "Berntsen"),
    ("Deirdre", "Scanlon"), ("Mieczyslaw", "Gorski"), ("Liv", "Samuelsen"),
    ("Brendan", "Tierney"), ("Halina", "Lewandowska"), ("Espen", "Christophersen"),
    ("Nuala", "Geraghty"), ("Kazimierz", "Ostrowski"), ("Maren", "Iversen"),
]

CTX_TEMPLATES = [
    "User profile — name: {name}; email: {email}; customer since {year}.",
    "Caller information: {name} <{email}>. Account tier: {tier}.",
    "Session context. Known user: {name}. Contact email: {email}.",
]


def make_ctx(first: str, last: str, email: str, rng: random.Random) -> str:
    t = rng.choice(CTX_TEMPLATES)
    return t.format(name=f"{first} {last}", email=email,
                    year=rng.randint(2019, 2025),
                    tier=rng.choice(["standard", "premium", "enterprise"]))


def ex(audio: np.ndarray, text: str, schema: str, source: str, **extra) -> dict:
    return {"audio": audio.astype(np.float32), "text": text, "schema": schema,
            "source": source, "audio_end_s": len(audio) / SR, **extra}


# ------------------------------------------------------------------ digits

def trim_edges(clip: np.ndarray, thresh: float = 5e-3, keep_ms: int = 40) -> np.ndarray:
    """Trim FSDD clip edge silence so within-group gaps are controlled by us,
    not by the recordings (v11: probe misses traced to uncontrolled gaps)."""
    rms = np.sqrt(np.convolve(clip ** 2, np.ones(160) / 160, mode="same"))
    nz = np.nonzero(rms > thresh)[0]
    if not len(nz):
        return clip
    a = max(0, nz[0] - int(SR * keep_ms / 1000))
    b = min(len(clip), nz[-1] + int(SR * keep_ms / 1000))
    return clip[a:b]


def groups_for(k: int) -> tuple[int, ...]:
    """Group a k-digit prefix the way a caller dictates it (3-3-4 pattern)."""
    full, out = [3, 3, 4], []
    for g in full:
        if k <= 0:
            break
        out.append(min(g, k))
        k -= min(g, k)
    return tuple(out)


def build_digit_examples(fsdd_dir: Path, rng: random.Random,
                         n_hold: int, n_fire: int, n_nosil: int,
                         fire_tail: tuple[float, float] = (0.4, 2.0)) -> list[dict]:
    pool = load_fsdd(fsdd_dir, TRAIN_FSDD_SPEAKERS)
    pool = {d: [trim_edges(c) for c in v] for d, v in pool.items()}
    logger.info("FSDD train pool sizes: %s", {d: len(v) for d, v in pool.items()})
    out = []
    # fire + hold share digit sequences (minimal-pair discipline)
    n_seq = max(n_fire, n_hold)
    for k in range(n_seq):
        digits = [rng.randrange(10) for _ in range(10)]
        if k < n_fire:
            # v13: shorter fire tail so the final fire lands promptly (inside
            # the +1.75 s scoring window); v12's 2.0 s tail taught the model to
            # wait ~2 s after the last digit, dropping in-window final recall.
            audio, meta = build_digit_sequence(
                pool, digits, rng, groups=(3, 3, 4),
                pause_range=(0.4, 1.2), gap_range=(0.05, 0.35),
                tail_s=rng.uniform(*fire_tail))
            out.append(ex(audio, f"{meta['text']} {M}", "digit_fire", "fsdd"))
        if k < n_hold:
            # v11: ANY prefix length 1..9 (probe misses showed fires after a
            # single leading digit — training had only 3/6-digit holds)
            n_kept = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            part = digits[:n_kept]
            audio, meta = build_digit_sequence(
                pool, part, rng, groups=groups_for(n_kept),
                pause_range=(0.4, 1.2), gap_range=(0.05, 0.35),
                tail_s=rng.uniform(0.5, 1.5))
            out.append(ex(audio, meta["text"], "digit_hold", "fsdd"))
    for _ in range(n_nosil):
        digits = [rng.randrange(10) for _ in range(10)]
        audio, meta = build_digit_sequence(
            pool, digits, rng, groups=(3, 3, 4),
            pause_range=(0.4, 1.2), tail_s=rng.uniform(0.0, 0.1))
        out.append(ex(audio, meta["text"], "digit_nosil", "fsdd"))
    return out


# ------------------------------------------------------------------ spelled

def choose_ctx(first: str, last: str, email: str, people: list, k: int,
               rng: random.Random, distractor_frac: float) -> tuple[str, str]:
    """Pick a context slot for a spelled example. Returns (ctx, ctx_kind).

    Three-way split (distractor_frac D controls the conflict share):
      - distractor (prob D): profile of a DIFFERENT identity; the target still
        comes from the AUDIO, so the model is trained that audio overrides a
        conflicting context. This is the anti-intrusion signal (research/75:
        long training makes the model copy whatever profile it is given).
      - matching   (prob (1-D)/2): profile of THIS identity (ctx helps spelling).
      - empty      (prob (1-D)/2): no profile (baseline capability).
    With D=0 this reduces to the v10/v11 50/50 match/empty behavior."""
    r = rng.random()
    if r < distractor_frac:
        # a plausibly-formed profile for someone else entirely
        j = (k + 17) % len(people)
        df, dl = people[j]
        de = email_of(df, dl, rng.choice(DOMAINS))
        return make_ctx(df, dl, de, rng), "distractor"
    if r < distractor_frac + (1.0 - distractor_frac) / 2:
        return make_ctx(first, last, email, rng), "match"
    return "", "empty"


def spelled_items(rng: random.Random, n_name: int, n_email: int,
                  distractor_frac: float = 0.0) -> list[dict]:
    """Utterance specs (text to synthesize + target + ctx) before TTS."""
    specs = []
    people = TRAIN_PEOPLE * ((max(n_name, n_email) // len(TRAIN_PEOPLE)) + 1)
    for k in range(n_name):
        first, last = people[k]
        domain = rng.choice(DOMAINS)
        email = email_of(first, last, domain)
        lead = rng.choice([
            f"My name is {first} {last}. That is spelled {spelled(last)}.",
            f"This is {first} {last}, spelled {spelled(last)}.",
            f"The last name is {last}. {spelled(last)}.",
        ])
        ctx, ctx_kind = choose_ctx(first, last, email, people, k, rng, distractor_frac)
        specs.append({
            "kind": "spell_name", "utt": lead,
            "target": lead,                      # verbatim; entity is in-place
            "ctx": ctx, "ctx_kind": ctx_kind,
        })
    for k in range(n_email):
        first, last = people[k + 3]
        domain = rng.choice(DOMAINS)
        email = email_of(first, last, domain)
        local = email.split("@")[0]
        a, b = local.split(".")
        if rng.random() < 0.5:
            utt = (f"My email address is {a} dot {b}, that is {spelled(a)}, dot, "
                   f"{spelled(b)}, at {domain_spoken(domain)}.")
        else:
            utt = f"My email address is {a} dot {b} at {domain_spoken(domain)}."
        ctx, ctx_kind = choose_ctx(first, last, email, people, k + 3, rng, distractor_frac)
        specs.append({
            "kind": "spell_email", "utt": utt,
            "target": f"My email address is {email}.",   # normalized form
            "ctx": ctx, "ctx_kind": ctx_kind,
        })
    return specs


def synth_spelled(specs: list[dict], cache_dir: Path, rng: random.Random) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    async def synth_all():
        sem = asyncio.Semaphore(4)
        async def one(i, spec):
            wav = cache_dir / f"tts_{i:04d}.wav"
            spec["wav"] = wav
            if wav.exists():
                return
            mp3 = cache_dir / f"tts_{i:04d}.mp3"
            voice = TRAIN_VOICES[i % len(TRAIN_VOICES)]
            async with sem:
                await tts_save(spec["utt"], voice, mp3)
            mp3_to_wav(mp3, wav)
            mp3.unlink(missing_ok=True)
        await asyncio.gather(*[one(i, s) for i, s in enumerate(specs)])

    asyncio.run(synth_all())

    out = []
    for spec in specs:
        audio, sr = sf.read(spec["wav"], dtype="float32")
        assert sr == SR
        # trim TTS's own trailing silence to <=0.1 s, then add our own tail
        rms = np.sqrt(np.convolve(audio ** 2, np.ones(320) / 320, mode="same"))
        nz = np.nonzero(rms > 1e-3)[0]
        if len(nz):
            audio = audio[: min(len(audio), nz[-1] + int(0.08 * SR))]
        if rng.random() < 0.15:                         # minimal pair: no tail
            text = spec["target"]
        else:
            audio = np.concatenate(
                [audio, np.zeros(int(rng.uniform(0.3, 1.0) * SR), dtype=np.float32)])
            text = f"{spec['target']} {M}"
        out.append(ex(audio, text, spec["kind"], "edge_tts", ctx=spec["ctx"],
                      ctx_kind=spec.get("ctx_kind", "")))
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--v9", default="data/semantic_endpoint_v9/data.pt")
    p.add_argument("--fsdd", default="data/fsdd/recordings")
    p.add_argument("--out", default="data/semantic_endpoint_v10")
    p.add_argument("--tts-cache", default="data/v10_tts")
    p.add_argument("--n-digit-fire", type=int, default=450)
    p.add_argument("--n-digit-hold", type=int, default=450)
    p.add_argument("--n-digit-nosil", type=int, default=200)
    p.add_argument("--digit-fire-tail-min", type=float, default=0.4)
    p.add_argument("--digit-fire-tail-max", type=float, default=2.0,
                   help="v13: set to ~1.0 so digit final fires land in-window.")
    p.add_argument("--n-spell-name", type=int, default=280)
    p.add_argument("--n-spell-email", type=int, default=280)
    p.add_argument("--dup-pair-hold", type=float, default=0.0,
                   help="Duplicate this fraction of pair_hold examples "
                        "(v11: preserve the conversational hold share against "
                        "dilution by the new schemas)")
    p.add_argument("--distractor-frac", type=float, default=0.0,
                   help="Fraction of spelled examples whose context holds a "
                        "DIFFERENT identity while the target follows the audio "
                        "(v12: anti-intrusion signal). NOTE: use a fresh "
                        "--tts-cache dir when changing this, since the cache is "
                        "keyed by spec index and the rng draw order shifts.")
    p.add_argument("--seed", type=int, default=10)
    args = p.parse_args()
    rng = random.Random(args.seed)

    v9 = torch.load(args.v9, weights_only=False)
    v9_examples = v9["examples"] if isinstance(v9, dict) and "examples" in v9 else v9
    logger.info("Loaded %d v9 examples", len(v9_examples))

    digit = build_digit_examples(Path(args.fsdd), rng, args.n_digit_hold,
                                 args.n_digit_fire, args.n_digit_nosil,
                                 fire_tail=(args.digit_fire_tail_min,
                                            args.digit_fire_tail_max))
    logger.info("Built %d digit examples", len(digit))

    specs = spelled_items(rng, args.n_spell_name, args.n_spell_email,
                          distractor_frac=args.distractor_frac)
    from collections import Counter as _C
    logger.info("spelled ctx mix: %s", _C(s["ctx_kind"] for s in specs))
    spell = synth_spelled(specs, Path(args.tts_cache), rng)
    logger.info("Built %d spelled examples", len(spell))

    examples = list(v9_examples) + digit + spell
    if args.dup_pair_hold > 0:
        ph = [e for e in examples if e["schema"] == "pair_hold"]
        extra = rng.sample(ph, int(len(ph) * args.dup_pair_hold))
        examples += [dict(e) for e in extra]
        logger.info("Oversampled pair_hold: +%d copies", len(extra))
    rng.shuffle(examples)
    from collections import Counter
    logger.info("v10 schema counts: %s", Counter(e["schema"] for e in examples))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(examples, out_dir / "data.pt")
    (out_dir / "build_meta.json").write_text(json.dumps({
        "v9_source": args.v9, "n_v9": len(v9_examples),
        "n_digit": len(digit), "n_spell": len(spell),
        "seed": args.seed, "distractor_frac": args.distractor_frac,
        "dup_pair_hold": args.dup_pair_hold,
        "spell_ctx_mix": dict(Counter(e.get("ctx_kind", "") for e in spell)),
        "schema_counts": dict(Counter(e["schema"] for e in examples)),
    }, indent=1))
    logger.info("Wrote %s (%d examples)", out_dir / "data.pt", len(examples))


if __name__ == "__main__":
    main()
