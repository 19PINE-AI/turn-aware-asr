"""Context-biasing experiment on Qwen3-ASR-0.6B.

For each utterance, run three conditions:
  1. NO_CTX: no system prompt
  2. RELEVANT: system prompt lists this utterance's own proper-noun entities
  3. DISTRACTOR: system prompt lists entities from a DIFFERENT random utterance

Per condition, compute WER. For entity-level metrics:
  - Per-entity recall (case-insensitive substring of reference entity in hyp)
  - Distractor hallucination rate (fraction of distractor entities that
    appear in the hypothesis despite not being in the audio)

Entity extraction is content-word-based: ALL_CAPS tokens in LibriSpeech
that aren't in a stop-word list. Crude but effective for LibriSpeech's
literary-fiction domain (every proper noun is capitalized).

Per `research/14-qwen3asr-baseline-validated.md`: this is the first
chart Qwen3-ASR's paper doesn't report. Synthesis 00 §6.1 targets
hotword recall ≥ 80 % and distractor hallucination ≤ 3 %.
"""

from __future__ import annotations
import argparse
import json
import logging
import random
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from eval.metrics import wer
from eval.run_qwen3asr_pkg import iter_librispeech_files

logger = logging.getLogger(__name__)


# Function words / very common LibriSpeech vocab — exclude from entity set.
STOPWORDS = {
    "THE", "AND", "OF", "TO", "A", "IN", "THAT", "IT", "IS", "WAS", "FOR",
    "ON", "AS", "WITH", "HE", "SHE", "AT", "BY", "THIS", "BUT", "FROM",
    "OR", "BE", "ONE", "NOT", "I", "AN", "WHICH", "HAD", "HAVE", "HAS",
    "HIS", "HER", "HIM", "MY", "ME", "YOU", "WE", "OUR", "THEY", "THEM",
    "THEIR", "BEEN", "WERE", "ARE", "BEING", "DO", "DOES", "DID", "WILL",
    "WOULD", "SHOULD", "COULD", "CAN", "MAY", "MIGHT", "MUST", "SHALL",
    "IF", "THEN", "ELSE", "WHEN", "WHERE", "WHO", "WHAT", "WHY", "HOW",
    "ALL", "ANY", "SOME", "NO", "EVERY", "EACH", "OTHER", "SUCH", "SO",
    "THUS", "NOW", "HERE", "THERE", "THIS", "THAT", "THESE", "THOSE",
    "VERY", "MUCH", "MORE", "MOST", "LESS", "LEAST", "TOO", "ONLY", "ALSO",
    "EVEN", "STILL", "JUST", "QUITE", "RATHER", "ALMOST", "ABOUT",
    "AFTER", "BEFORE", "AGAIN", "BACK", "AROUND", "AGAINST",
    "ANOTHER", "ANYTHING", "ANYWHERE", "BECAUSE", "BEYOND", "BOTH",
    "DOWN", "EITHER", "ENOUGH", "FAR", "FEW", "FIRST", "FOREVER",
    "FORWARD", "GIVE", "GO", "GREAT", "HIGH", "HOWEVER", "INSIDE", "INTO",
    "IT'S", "LIKE", "LITTLE", "LONG", "LOW", "MADE", "MAKE", "MAN", "MEN",
    "WOMAN", "WOMEN", "MORNING", "NEW", "OLD", "OUT", "OUTSIDE", "OVER",
    "OWN", "PUT", "SAID", "SAY", "SEE", "SEEM", "TAKE", "TELL", "THINGS",
    "THINK", "TIME", "TOGETHER", "UP", "UPON", "USE", "WAY", "WAYS",
    "WELL", "WHILE", "WITHIN", "WITHOUT", "YEAR", "YET",
}


def extract_entities(transcript: str, min_len: int = 4) -> list[str]:
    """Crude proper-noun extraction for LibriSpeech (everything is uppercase).

    Returns a deduped list of UPPERCASE word tokens >= min_len chars that
    aren't in STOPWORDS. Preserves order of first occurrence.
    """
    words = re.findall(r"[A-Z]{%d,}" % min_len, transcript)
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w in STOPWORDS:
            continue
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def build_system_prompt(hotwords: list[str]) -> str:
    """Format a system prompt with hotwords, matching qwen-asr's biasing convention."""
    if not hotwords:
        return ""
    return (
        "The user will provide an audio recording. Transcribe it verbatim. "
        "The recording may contain these proper nouns or named entities: "
        + ", ".join(hotwords)
        + "."
    )


def entity_recall(hyp: str, ref_entities: list[str]) -> tuple[int, int]:
    """Return (hits, total) for case-insensitive substring recall."""
    hyp_lo = hyp.lower()
    hits = sum(1 for e in ref_entities if e.lower() in hyp_lo)
    return hits, len(ref_entities)


def hallucination_count(hyp: str, distractor_entities: list[str]) -> tuple[int, int]:
    """Return (hallucinated, total_distractors) for distractor entities."""
    hyp_lo = hyp.lower()
    hits = sum(1 for e in distractor_entities if e.lower() in hyp_lo)
    return hits, len(distractor_entities)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", default="data/librispeech_raw/LibriSpeech/test-clean")
    p.add_argument("--max-utterances", type=int, default=60)
    p.add_argument("--out", default="research/15-context-biasing-results.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)

    logger.info("Loading Qwen3-ASR-0.6B…")
    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        cache_dir="data/qwen3-asr-0.6b-pkg",
        max_inference_batch_size=4,
        max_new_tokens=256,
    )

    pairs = list(iter_librispeech_files(args.split_dir))[: args.max_utterances]
    logger.info("Loaded %d utterances", len(pairs))

    # Extract entities for every utterance once
    utts = []
    for flac, ref in pairs:
        ents = extract_entities(ref)
        utts.append({"flac": flac, "ref": ref, "entities": ents})

    n_utt = len(utts)
    has_ents = [u for u in utts if u["entities"]]
    logger.info("%d / %d utterances have ≥ 1 extracted entity", len(has_ents), n_utt)

    results = []
    for i, u in enumerate(tqdm(utts, desc="biasing-eval")):
        audio, sr = sf.read(u["flac"], dtype="float32")
        audio_arr = (np.array(audio, dtype=np.float32), sr)

        # Pick a distractor utterance with entities (and different from ourselves)
        candidates = [v for v in has_ents if v["flac"] != u["flac"]]
        distractor = rng.choice(candidates) if candidates else None
        distractor_entities = distractor["entities"] if distractor else []

        # Condition 1: NO_CTX
        hyp_nc = model.transcribe(audio=audio_arr)[0].text
        # Condition 2: RELEVANT
        sys_rel = build_system_prompt(u["entities"])
        hyp_rel = model.transcribe(audio=audio_arr, context=sys_rel)[0].text
        # Condition 3: DISTRACTOR
        sys_dis = build_system_prompt(distractor_entities)
        hyp_dis = model.transcribe(audio=audio_arr, context=sys_dis)[0].text

        r = {
            "utt_id": Path(u["flac"]).stem,
            "ref": u["ref"],
            "ref_entities": u["entities"],
            "distractor_entities": distractor_entities,
            "hyp_no_ctx": hyp_nc,
            "hyp_relevant": hyp_rel,
            "hyp_distractor": hyp_dis,
        }
        if u["entities"]:
            hits_nc, total = entity_recall(hyp_nc, u["entities"])
            hits_rel, _ = entity_recall(hyp_rel, u["entities"])
            r["recall_no_ctx"] = hits_nc / total
            r["recall_relevant"] = hits_rel / total
        if distractor_entities:
            hall_dis, total_d = hallucination_count(hyp_dis, distractor_entities)
            r["hallucination_distractor_count"] = hall_dis
            r["hallucination_distractor_total"] = total_d
            r["hallucination_distractor_rate"] = hall_dis / total_d
        results.append(r)

        if i < 2:
            logger.info("REF[%d]: %s", i, u["ref"][:80])
            logger.info("  NO_CTX: %s", hyp_nc[:80])
            logger.info("  RELEVANT (hotwords=%s): %s",
                         u["entities"][:5], hyp_rel[:80])
            logger.info("  DISTRACTOR (hotwords=%s): %s",
                         distractor_entities[:5], hyp_dis[:80])

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

    # Entity-level aggregates (only over utterances that have entities)
    rel_present = [r for r in results if "recall_no_ctx" in r]
    if rel_present:
        n_ents = sum(len(r["ref_entities"]) for r in rel_present)
        hits_nc_tot = sum(int(r["recall_no_ctx"] * len(r["ref_entities"])) for r in rel_present)
        hits_rel_tot = sum(int(r["recall_relevant"] * len(r["ref_entities"])) for r in rel_present)
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
        "model": "Qwen/Qwen3-ASR-0.6B",
        "summary": summary,
        "per_utterance": results,
    }, indent=2))


if __name__ == "__main__":
    main()
