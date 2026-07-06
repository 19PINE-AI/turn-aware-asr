"""Gap 1: natural-speech biasing counterfactuals, appended to a training pool.

The spelled-entity distractor fix (build_v10 choose_ctx) taught "audio overrides
a conflicting *profile*" and drove probe intrusion to 0, but did NOT cover the
Earnings-22 natural-speech regime where the biasing context is a comma-joined
*entity list* (build_system_prompt) — release distractor-hallucination is 6.3%.

This builds the natural-list analog from LibriSpeech (held out from the
Earnings-22 eval by construction — different corpus, no contamination) using the
same spaCy entity extractor and the SAME prompt format the eval scores against:

  distractor example: ctx = shuffle(own_entities + other_utt_entities),
                      target = verbatim transcript (own entities appear, the
                      other-utt distractors do NOT) -> trains "emit only what
                      the audio supports, ignore context-only entities."
  relevant  example: ctx = own_entities only, target = transcript -> reinforces
                      the biasing uplift (present entities are transcribed).

No marker, no trailing-silence tail (same shape as asr_plain) -> also reinforces
"don't fire without silence." Entities are cached to avoid recomputing spaCy.

Usage:
  .venv/bin/python eval/build_natural_biasing.py \
      --pool data/semantic_endpoint_v16/data.pt --n-distractor 1200 --n-relevant 600
"""
from __future__ import annotations
import argparse, json, logging, random
from collections import Counter
from pathlib import Path
import soundfile as sf, torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.build_v10_training_data import trim_edges, ex, SR  # noqa: E402
from eval.earnings22_biasing import build_system_prompt  # noqa: E402
from eval.llm_entities import extract_entities_llm  # noqa: E402  (Haiku 4.5, matches the eval)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_candidates(cache_path: Path, ent_cache_path: Path, min_dur: float,
                    max_dur: float, min_entities: int, max_extract: int,
                    rng: random.Random) -> list[dict]:
    """(flac, text, entities) for LS utts with >= min_entities entities.

    Entities come from the SAME Haiku-4.5 extractor the Earnings-22 eval uses
    (eval/llm_entities.extract_entities_llm), so the training entity-string
    distribution matches the eval's distractor/relevant prompts. Batched + cached.
    """
    cache = json.loads(cache_path.read_text())
    # in-range, well-formed utts, shuffled, capped so we bound the API spend
    pool = []
    for flac, text in cache.items():
        text = (text or "").strip()
        if not text or len(text.split()) < 3 or not Path(flac).exists():
            continue
        try:
            info = sf.info(flac)
        except Exception:
            continue
        if min_dur <= info.frames / info.samplerate <= max_dur:
            pool.append((flac, text))
    rng.shuffle(pool)
    pool = pool[:max_extract]
    logger.info("extracting Haiku entities for %d candidate utts…", len(pool))
    ent_lists = extract_entities_llm([t for _, t in pool], cache_path=str(ent_cache_path))
    cands = [{"flac": f, "text": t, "entities": e}
             for (f, t), e in zip(pool, ent_lists) if len(e) >= min_entities]
    logger.info("candidates with >=%d entities: %d / %d", min_entities, len(cands), len(pool))
    return cands


def load_audio(flac: str):
    audio, sr = sf.read(flac, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return trim_edges(audio.astype("float32"), thresh=5e-3, keep_ms=60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="data/semantic_endpoint_v16/data.pt",
                   help="existing pool to extend in-place")
    p.add_argument("--cache", default="data/semantic_endpoint_v9/ls_transcripts.json")
    p.add_argument("--ent-cache", default="data/semantic_endpoint_v9/ls_llm_entities.json")
    p.add_argument("--out", default=None, help="default: overwrite --pool dir")
    p.add_argument("--n-distractor", type=int, default=1200)
    p.add_argument("--n-relevant", type=int, default=600)
    p.add_argument("--min-dur", type=float, default=2.5)
    p.add_argument("--max-dur", type=float, default=14.0)
    p.add_argument("--min-entities", type=int, default=1)
    p.add_argument("--max-extract", type=int, default=3000,
                   help="cap on utts sent to the LLM extractor (bounds API spend)")
    p.add_argument("--max-relevant-hot", type=int, default=4, help="cap own entities in ctx")
    p.add_argument("--n-distractor-hot", type=int, default=3, help="distractors added to ctx")
    p.add_argument("--seed", type=int, default=16)
    args = p.parse_args()
    rng = random.Random(args.seed)

    pool = torch.load(args.pool, weights_only=False)
    logger.info("pool: %d examples", len(pool))
    cands = load_candidates(Path(args.cache), Path(args.ent_cache), args.min_dur,
                            args.max_dur, args.min_entities, args.max_extract, rng)
    if len(cands) < 50:
        raise SystemExit("too few entity-bearing candidates; expand the cache first")

    # entity bank for drawing distractors (entities NOT in the target utt)
    ent_bank = sorted({e for c in cands for e in c["entities"]})
    logger.info("distinct entity bank: %d", len(ent_bank))

    added = []

    def make(kind: str, n: int):
        picks = [rng.choice(cands) for _ in range(n)]
        for c in picks:
            own = list(dict.fromkeys(c["entities"]))[: args.max_relevant_hot]
            if kind == "distractor":
                own_lower = {e.lower() for e in c["text"].split()}  # cheap presence guard
                distractors, tries = [], 0
                while len(distractors) < args.n_distractor_hot and tries < 50:
                    d = rng.choice(ent_bank)
                    tries += 1
                    # a real distractor must NOT appear in this utt's transcript
                    if d.lower() in c["text"].lower():
                        continue
                    if d in distractors or d in own:
                        continue
                    distractors.append(d)
                hot = own + distractors
                rng.shuffle(hot)
                extra = {"ctx": build_system_prompt(hot), "distractors": distractors}
                schema = "bias_nat_distractor"
            else:  # relevant
                hot = own[:]
                rng.shuffle(hot)
                extra = {"ctx": build_system_prompt(hot)}
                schema = "bias_nat_relevant"
            audio = load_audio(c["flac"])
            added.append(ex(audio, c["text"], schema, "ls_bias", **extra))

    make("distractor", args.n_distractor)
    make("relevant", args.n_relevant)
    logger.info("added %d natural-biasing examples", len(added))

    examples = list(pool) + added
    rng.shuffle(examples)
    out = Path(args.out) if args.out else Path(args.pool).parent
    out.mkdir(parents=True, exist_ok=True)
    torch.save(examples, out / "data.pt")
    counts = dict(Counter(e["schema"] for e in examples))
    (out / "biasing_meta.json").write_text(json.dumps(
        {"pool": args.pool, "n_pool": len(pool), "n_added": len(added),
         "n_distractor": args.n_distractor, "n_relevant": args.n_relevant,
         "schema_counts": counts}, indent=1))
    logger.info("wrote %s (%d examples); schemas: %s", out / "data.pt", len(examples), counts)


if __name__ == "__main__":
    main()
