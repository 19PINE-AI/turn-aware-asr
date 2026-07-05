"""Bigger spelled-entity probe (~240 items) for tighter CIs on context biasing.

Procedurally combines uncommon first/last names into N identities, disjoint
from TRAIN_PEOPLE (build_v10) and the small PROBE_PEOPLE (build_dictation_probes).
Same structure as the small probe: each identity -> a spelled-name item and an
email item; profile ctx + distractor ctx; probe voices (Jenny/Guy) held out from
training. Writes data/probes_big/spelled_probe.json.

Usage: .venv/bin/python eval/build_big_spelled_probe.py --n-people 120
"""
from __future__ import annotations
import argparse, asyncio, json, logging, random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.build_dictation_probes import (           # noqa: E402
    PROBE_VOICES, DOMAINS, PROBE_PEOPLE, domain_spoken, email_of, spelled,
    profile_ctx, tts_save, mp3_to_wav, SR,
)
from eval.build_v10_training_data import TRAIN_PEOPLE  # noqa: E402
import soundfile as sf  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIRST = ["Anneliese", "Bartholomew", "Cosima", "Dagfinn", "Evangelina", "Ferdinand",
         "Gwendolyn", "Hyacinth", "Ignatius", "Jehanne", "Kristoffer", "Leocadia",
         "Maximilian", "Nikolina", "Oswaldo", "Persephone", "Quintus", "Rosalind",
         "Sigismund", "Theodora", "Ulrich", "Valentina", "Wilhelmina", "Xanthe",
         "Yaroslav", "Zephyrine", "Alistair", "Brunhilde", "Cornelius", "Delphine",
         "Emmerich", "Fiorella", "Gottfried", "Henrietta", "Immanuel", "Josephina",
         "Konstantin", "Ludmila", "Mordecai", "Nadezhda"]
LAST = ["Aberforth", "Blomqvist", "Cavanaugh", "Dziedzic", "Eskildsen", "Fitzalan",
         "Grzybowski", "Haugland", "Iwanowski", "Jankauskas", "Kaczmarczyk", "Lindholm",
         "Mavrogiannis", "Nordskov", "Oyelaran", "Papadopoulos", "Quintanilla", "Rasmussen",
         "Steinsson", "Trzcinski", "Ueberroth", "Vandenberghe", "Wojnarowski", "Xanthopoulos",
         "Yankovic", "Zabludowicz", "Anagnostou", "Brzezinski", "Csikszentmihalyi", "Dabrowski",
         "Echeverria", "Featherstone", "Guttormsen", "Hjortshoj", "Ingebrigtsen", "Jokinen",
         "Klimowicz", "Loughnane", "Magnusdottir", "Nieminen"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/probes_big")
    p.add_argument("--n-people", type=int, default=120)
    p.add_argument("--seed", type=int, default=880)
    args = p.parse_args()
    rng = random.Random(args.seed)

    banned = {(f.lower(), l.lower()) for f, l in TRAIN_PEOPLE + PROBE_PEOPLE}
    people, seen = [], set()
    while len(people) < args.n_people:
        f, l = rng.choice(FIRST), rng.choice(LAST)
        key = (f.lower(), l.lower())
        if key in banned or key in seen:
            continue
        seen.add(key); people.append((f, l))

    wav_dir = Path(args.out) / "spelled_wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for k, (first, last) in enumerate(people):
        domain = rng.choice(DOMAINS)
        email = email_of(first, last, domain)
        voice = PROBE_VOICES[k % len(PROBE_VOICES)]
        items.append({"kind": "name", "voice": voice,
                      "utt": f"My name is {first} {last}. That is spelled {spelled(last)}.",
                      "first": first, "last": last, "email": email,
                      "target": "".join(c for c in last.lower() if c.isalpha()),
                      "ctx": profile_ctx(first, last, email, rng)})
        local = email.split("@")[0]; a, b = local.split(".")
        if k % 2 == 0:
            utt = (f"My email address is {a} dot {b}, that is {spelled(a)}, dot, "
                   f"{spelled(b)}, at {domain_spoken(domain)}.")
            style = "spelled"
        else:
            utt = f"My email address is {a} dot {b} at {domain_spoken(domain)}."
            style = "natural"
        items.append({"kind": "email", "style": style, "voice": voice, "utt": utt,
                      "first": first, "last": last, "email": email,
                      "target": email, "ctx": profile_ctx(first, last, email, rng)})
    n = len(items)
    for i, it in enumerate(items):
        it["ctx_distractor"] = items[(i + 37) % n]["ctx"]; it["id"] = i

    async def synth():
        sem = asyncio.Semaphore(4)
        async def one(it):
            wav = wav_dir / f"sp_{it['id']:04d}.wav"; it["wav"] = str(wav)
            if wav.exists():
                return
            mp3 = wav_dir / f"sp_{it['id']:04d}.mp3"
            async with sem:
                await tts_save(it["utt"], it["voice"], mp3)
            mp3_to_wav(mp3, wav); mp3.unlink(missing_ok=True)
        await asyncio.gather(*[one(it) for it in items])
    asyncio.run(synth())
    Path(args.out, "spelled_probe.json").write_text(json.dumps(items, indent=1))
    logger.info("Big spelled probe: %d items (%d identities) -> %s",
                len(items), len(people), args.out)


if __name__ == "__main__":
    main()
