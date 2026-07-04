"""Context-biasing experiment on the entity-dense Earnings-22 corpus.

This mirrors `eval/context_biasing_experiment.py` (LibriSpeech) but on
harder, proper-noun-dense earnings-call audio. Earnings-22 transcripts are
mixed-case spontaneous speech full of company names, executives, tickers,
and products — exactly the regime where context biasing should matter and
where the crude ALL-CAPS entity heuristic of the LibriSpeech experiment
breaks down. We therefore prefer spaCy NER here.

For each utterance, run three conditions:
  1. NO_CTX:     no system prompt
  2. RELEVANT:   system prompt lists this utterance's own named entities
  3. DISTRACTOR: system prompt lists entities from a DIFFERENT random utt

Per condition, compute WER. For entity-level metrics:
  - Per-entity recall (case-insensitive substring of reference entity in hyp)
  - Distractor hallucination rate (fraction of distractor entities that
    appear in the hypothesis despite not being in the audio)

This targets the "lead chart" Qwen3-ASR's paper reports zero for: hotword
recall with a relevant text prefix (synthesis 00 §6.1 target: recall >= 80 %,
distractor hallucination <= 3 %).

Data: `distil-whisper/earnings22` (parquet shards, `chunked/test-*`), each
row = {file_id, segment_id, transcription, start_ts, end_ts, audio(wav bytes)}.

Usage:
    # data-only dry run (NO model, NO GPU):
    python -m eval.earnings22_biasing --dry-run --n-utts 10
    # full run (loads Qwen3-ASR on the GPU):
    python -m eval.earnings22_biasing --n-utts 200
"""

from __future__ import annotations
import argparse
import glob
import io
import json
import logging
import random
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm

from eval.metrics import wer

logger = logging.getLogger(__name__)

DEFAULT_DATA_GLOB = "data/earnings22/chunked/*.parquet"

# spaCy entity labels we treat as biasing hotwords.
SPACY_LABELS = {"PERSON", "ORG", "PRODUCT", "GPE"}

# Reference-cleanup: Earnings-22 (distil-whisper) marks non-speech / unclear
# spans with angle-bracket tags, e.g. "<inaudible>", "<laugh>". Strip them
# before NER and scoring so they don't pollute entities or WER.
_TAG_RE = re.compile(r"<[^>]*>")
_WS_RE = re.compile(r"\s+")


def clean_ref(text: str) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text)).strip()


# --------------------------------------------------------------------------
# Entity extraction
# --------------------------------------------------------------------------

_NLP = None  # lazily-loaded spaCy pipeline


def _load_spacy():
    """Return a loaded spaCy pipeline, or None if unavailable. CPU-only."""
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        logger.info("Entity extraction: spaCy en_core_web_sm (PERSON/ORG/PRODUCT/GPE)")
    except Exception as e:  # noqa: BLE001
        logger.warning("spaCy en_core_web_sm unavailable (%s); falling back to "
                       "the ALL-CAPS heuristic. TODO: install en_core_web_sm "
                       "(python -m spacy download en_core_web_sm) — the ALL-CAPS "
                       "heuristic is a poor fit for mixed-case earnings text.", e)
        _NLP = False
    return _NLP


# Minimal stopword set for the ALL-CAPS fallback (mirrors the LibriSpeech
# experiment). Only used if spaCy is missing — see TODO above.
_FALLBACK_STOPWORDS = {
    "THE", "AND", "OF", "TO", "A", "IN", "THAT", "IT", "IS", "WAS", "FOR",
    "ON", "AS", "WITH", "AT", "BY", "THIS", "BUT", "FROM", "OR", "BE", "ONE",
    "NOT", "AN", "WHICH", "HAD", "HAVE", "HAS", "YOU", "WE", "THEY", "ARE",
    "WILL", "WOULD", "OKAY", "YEAH", "UH", "UM",
}


def _extract_fallback(text: str, min_len: int = 4) -> list[str]:
    """ALL-CAPS token heuristic (LibriSpeech-style). Poor on mixed-case text."""
    words = re.findall(r"[A-Z]{%d,}" % min_len, text)
    seen, out = set(), []
    for w in words:
        if w in _FALLBACK_STOPWORDS or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def extract_entities(text: str, min_len: int = 2) -> list[str]:
    """Extract named-entity hotwords, deduped, order preserved.

    Prefers spaCy NER restricted to PERSON/ORG/PRODUCT/GPE. Falls back to the
    ALL-CAPS heuristic if spaCy is unavailable.
    """
    nlp = _load_spacy()
    if not nlp:
        return _extract_fallback(text)
    doc = nlp(text)
    seen, out = set(), []
    for ent in doc.ents:
        if ent.label_ not in SPACY_LABELS:
            continue
        e = ent.text.strip()
        # Drop pure function words / stray punctuation / too-short fragments.
        if len(e) < min_len or not any(c.isalpha() for c in e):
            continue
        key = e.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


# --------------------------------------------------------------------------
# Prompt + scoring helpers (identical semantics to context_biasing_experiment)
# --------------------------------------------------------------------------

def build_system_prompt(hotwords: list[str]) -> str:
    if not hotwords:
        return ""
    return (
        "The user will provide an audio recording. Transcribe it verbatim. "
        "The recording may contain these proper nouns or named entities: "
        + ", ".join(hotwords)
        + "."
    )


def entity_recall(hyp: str, ref_entities: list[str]) -> tuple[int, int]:
    hyp_lo = hyp.lower()
    hits = sum(1 for e in ref_entities if e.lower() in hyp_lo)
    return hits, len(ref_entities)


def hallucination_count(hyp: str, distractor_entities: list[str]) -> tuple[int, int]:
    hyp_lo = hyp.lower()
    hits = sum(1 for e in distractor_entities if e.lower() in hyp_lo)
    return hits, len(distractor_entities)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def iter_earnings22(data_glob: str, limit: int, min_dur: float = 0.4,
                    shuffle: bool = False, seed: int = 0):
    """Yield dicts for up to `limit` Earnings-22 utterances (no model).

    Each dict: {file_id, segment_id, ref, audio_bytes, sr, dur}. Audio is
    kept as encoded WAV bytes (compact) and decoded lazily in the loop to
    keep resident memory low.

    shuffle=True samples `limit` utterances uniformly across the whole shard
    (all earnings calls) rather than the first `limit` in file order — which
    otherwise draws almost entirely from the first 1-2 companies and biases
    the entity distribution. The shard (~610 MB) fits comfortably in RAM.
    """
    files = sorted(glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"No parquet shards match {data_glob!r}. "
                                "Download a shard from distil-whisper/earnings22.")
    cols = ["file_id", "segment_id", "transcription", "start_ts", "end_ts", "audio"]

    def _all_valid():
        for fp in files:
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(batch_size=64, columns=cols):
                d = batch.to_pydict()
                for i in range(len(d["transcription"])):
                    ref = clean_ref(d["transcription"][i])
                    dur = float(d["end_ts"][i]) - float(d["start_ts"][i])
                    if not ref or dur < min_dur:
                        continue
                    a = d["audio"][i]
                    yield {
                        "file_id": str(d["file_id"][i]),
                        "segment_id": str(d["segment_id"][i]),
                        "ref": ref,
                        "audio_bytes": a["bytes"],
                        "audio_path": a["path"],
                        "dur": dur,
                    }

    if shuffle:
        rows = list(_all_valid())
        random.Random(seed).shuffle(rows)
        for r in rows[:limit]:
            yield r
        return

    n = 0
    for r in _all_valid():
        yield r
        n += 1
        if n >= limit:
            return


def decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    return np.asarray(audio, dtype=np.float32), sr


# --------------------------------------------------------------------------
# Transcription backend (model load happens ONLY here, at run time)
# --------------------------------------------------------------------------

class Transcriber:
    """Wraps a Qwen3-ASR backend. Constructed only for a real (non-dry) run.

    Mirrors `context_biasing_experiment.py`: the `pkg` backend loads the
    official qwen-asr package and calls `model.transcribe(audio=..., context=...)`.
    """

    def __init__(self, backend: str = "pkg", model_name: str = "Qwen/Qwen3-ASR-0.6B"):
        self.backend = backend
        if backend == "pkg":
            from qwen_asr import Qwen3ASRModel  # noqa: PLC0415  (heavy import)
            self.model = Qwen3ASRModel.from_pretrained(
                model_name,
                cache_dir="data/qwen3-asr-0.6b-pkg",
                max_inference_batch_size=4,
                max_new_tokens=256,
            )
        elif backend == "native":
            # TODO: wire the transformers backend via
            # `eval.run_qwen3asr_native.transcribe` (needs mel feature
            # extraction + src.qwen3_asr_loader.load_full_qwen3_asr).
            raise NotImplementedError(
                "backend='native' not yet wired for earnings22_biasing; "
                "use --backend pkg (matches context_biasing_experiment.py).")
        else:
            raise ValueError(f"unknown backend {backend!r}")

    def transcribe(self, audio_arr, context: str = "") -> str:
        # Structured exactly like context_biasing_experiment.py.
        if context:
            res = self.model.transcribe(audio=audio_arr, context=context)
        else:
            res = self.model.transcribe(audio=audio_arr)
        return res[0].text if res else ""


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def load_utterances(data_glob: str, n_utts: int,
                    shuffle: bool = False, seed: int = 0) -> list[dict]:
    utts = []
    for u in tqdm(iter_earnings22(data_glob, n_utts, shuffle=shuffle, seed=seed),
                  total=n_utts, desc="load"):
        u["entities"] = extract_entities(u["ref"])
        utts.append(u)
    return utts


def build_conditions(utts: list[dict], rng: random.Random):
    """Attach a distractor + the RELEVANT/DISTRACTOR prompts to each utt."""
    has_ents = [u for u in utts if u["entities"]]
    for u in utts:
        candidates = [v for v in has_ents
                      if (v["file_id"], v["segment_id"]) != (u["file_id"], u["segment_id"])]
        distractor = rng.choice(candidates) if candidates else None
        u["distractor_entities"] = distractor["entities"] if distractor else []
        u["sys_relevant"] = build_system_prompt(u["entities"])
        u["sys_distractor"] = build_system_prompt(u["distractor_entities"])
    return has_ents


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-glob", default=DEFAULT_DATA_GLOB)
    p.add_argument("--n-utts", type=int, default=200)
    p.add_argument("--out", default="research/63-earnings22-biasing.json")
    p.add_argument("--backend", default="pkg", choices=["pkg", "native"],
                   help="ASR backend, matching the existing eval scripts.")
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle", action="store_true",
                   help="sample utterances uniformly across the whole shard "
                        "(all earnings calls) for entity/company diversity, "
                        "instead of the first n-utts in file order.")
    p.add_argument("--dry-run", action="store_true",
                   help="Data pipeline only: extract entities, build the three "
                        "conditions, print samples. Loads NO model, uses NO GPU.")
    args = p.parse_args()

    rng = random.Random(args.seed)

    utts = load_utterances(args.data_glob, args.n_utts,
                           shuffle=args.shuffle, seed=args.seed)
    has_ents = build_conditions(utts, rng)
    logger.info("Loaded %d utterances; %d have >= 1 named entity", len(utts), len(has_ents))

    if args.dry_run:
        n_show = min(10, len(utts))
        logger.info("== DRY RUN: printing %d built conditions (no model) ==", n_show)
        for i in range(n_show):
            u = utts[i]
            print(f"\n===== utt {i}  file={u['file_id']} seg={u['segment_id']} "
                  f"dur={u['dur']:.1f}s =====")
            print(f"  REF        : {u['ref']}")
            print(f"  ENTITIES   : {u['entities']}")
            print(f"  RELEVANT   : {u['sys_relevant'] or '(empty — no entities)'}")
            print(f"  DISTRACTOR : {u['sys_distractor'] or '(empty — no entities)'}")
        # Coverage stats useful for planning the real run.
        n_ent = sum(len(u["entities"]) for u in utts)
        logger.info("Total entities: %d (mean %.2f/utt); entity-bearing utts: %d/%d",
                    n_ent, n_ent / max(1, len(utts)), len(has_ents), len(utts))
        logger.info("Dry run complete — no model loaded, no GPU used.")
        return

    logger.info("Loading Qwen3-ASR backend=%s (%s)…", args.backend, args.model)
    tx = Transcriber(backend=args.backend, model_name=args.model)

    results = []
    for i, u in enumerate(tqdm(utts, desc="biasing-eval")):
        audio_arr = decode_audio(u["audio_bytes"])

        hyp_nc = tx.transcribe(audio_arr)                       # NO_CTX
        hyp_rel = tx.transcribe(audio_arr, u["sys_relevant"])   # RELEVANT
        hyp_dis = tx.transcribe(audio_arr, u["sys_distractor"]) # DISTRACTOR

        r = {
            "utt_id": f"{u['file_id']}-{u['segment_id']}",
            "ref": u["ref"],
            "ref_entities": u["entities"],
            "distractor_entities": u["distractor_entities"],
            "hyp_no_ctx": hyp_nc,
            "hyp_relevant": hyp_rel,
            "hyp_distractor": hyp_dis,
        }
        if u["entities"]:
            hits_nc, total = entity_recall(hyp_nc, u["entities"])
            hits_rel, _ = entity_recall(hyp_rel, u["entities"])
            r["recall_no_ctx"] = hits_nc / total
            r["recall_relevant"] = hits_rel / total
        if u["distractor_entities"]:
            hall_dis, total_d = hallucination_count(hyp_dis, u["distractor_entities"])
            r["hallucination_distractor_count"] = hall_dis
            r["hallucination_distractor_total"] = total_d
            r["hallucination_distractor_rate"] = hall_dis / total_d
        results.append(r)

        if i < 2:
            logger.info("REF[%d]: %s", i, u["ref"][:80])
            logger.info("  NO_CTX: %s", hyp_nc[:80])
            logger.info("  RELEVANT (hotwords=%s): %s", u["entities"][:5], hyp_rel[:80])
            logger.info("  DISTRACTOR (hotwords=%s): %s", u["distractor_entities"][:5], hyp_dis[:80])

    # Aggregate
    def _wer(field):
        return wer(
            " \n".join(r[field] for r in results),
            " \n".join(r["ref"] for r in results),
        ) * 100

    summary = {
        "n_utterances": len(results),
        "wer_no_ctx": _wer("hyp_no_ctx"),
        "wer_relevant": _wer("hyp_relevant"),
        "wer_distractor": _wer("hyp_distractor"),
    }

    rel_present = [r for r in results if "recall_no_ctx" in r]
    if rel_present:
        n_ents = sum(len(r["ref_entities"]) for r in rel_present)
        hits_nc_tot = sum(int(round(r["recall_no_ctx"] * len(r["ref_entities"]))) for r in rel_present)
        hits_rel_tot = sum(int(round(r["recall_relevant"] * len(r["ref_entities"]))) for r in rel_present)
        summary["entity_recall_no_ctx"] = hits_nc_tot / n_ents
        summary["entity_recall_relevant"] = hits_rel_tot / n_ents
        summary["entity_recall_uplift_pp"] = (
            summary["entity_recall_relevant"] - summary["entity_recall_no_ctx"]
        ) * 100

    dis_present = [r for r in results if "hallucination_distractor_total" in r]
    if dis_present:
        n_d = sum(r["hallucination_distractor_total"] for r in dis_present)
        h_d = sum(r["hallucination_distractor_count"] for r in dis_present)
        summary["distractor_entity_count"] = n_d
        summary["distractor_hallucination_rate"] = h_d / n_d if n_d else 0.0

    logger.info("== SUMMARY ==")
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info("  %-32s %.3f", k, v)
        else:
            logger.info("  %-32s %s", k, v)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model,
        "backend": args.backend,
        "dataset": "distil-whisper/earnings22",
        "summary": summary,
        "per_utterance": results,
    }, indent=2))
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
