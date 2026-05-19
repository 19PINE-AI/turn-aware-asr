"""Eval metrics per research/07-eval-harness-spec.md."""

from __future__ import annotations
import numpy as np


def wer(hypothesis: str, reference: str) -> float:
    """Levenshtein WER (whisper-norm-style lowercasing + punctuation strip)."""
    try:
        import jiwer
    except ImportError:
        raise ImportError("Install jiwer: pip install jiwer")
    norm = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    return jiwer.wer(
        reference, hypothesis,
        reference_transform=norm, hypothesis_transform=norm,
    )


def endpoint_latencies(
    endpoint_event_ms: int | None,
    truth_ms: int,
) -> tuple[float, str]:
    """Returns (latency_ms, category).

    Categories: 'premature' (>100ms before truth), 'late' (>1500ms after),
    'on_time' (otherwise), 'missing' (no endpoint fired).
    """
    if endpoint_event_ms is None:
        return float("nan"), "missing"
    latency = endpoint_event_ms - truth_ms
    if latency < -100:
        return latency, "premature"
    if latency > 1500:
        return latency, "late"
    return latency, "on_time"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, p))


def per_entity_recall(
    hypothesis: str, reference: str, entities: list[str]
) -> dict[str, float]:
    """Naive case-insensitive substring recall for the given entity list.

    Used for hotword-stress-v1 and Earnings-22 hotword metrics. For more
    rigorous NER-level recall, replace with spaCy NER hits.
    """
    h_lower = hypothesis.lower()
    r_lower = reference.lower()
    out = {}
    for e in entities:
        e_l = e.lower()
        if e_l in r_lower:
            out[e] = 1.0 if e_l in h_lower else 0.0
    return out


def hallucination_rate(
    hypothesis: str, reference: str, distractor_entities: list[str]
) -> float:
    """Fraction of distractor entities that appear in hypothesis."""
    if not distractor_entities:
        return 0.0
    h_lower = hypothesis.lower()
    n_hallucinated = sum(1 for e in distractor_entities if e.lower() in h_lower)
    return n_hallucinated / len(distractor_entities)
