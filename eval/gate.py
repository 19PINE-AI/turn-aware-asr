"""Phase gate predicate runner — exits 0 (pass) or 1 (fail).

Usage:
  python -m eval.gate --phase 1 --results eval/results/latest.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


# Gate predicates per research/00-synthesis.md §6.1 (revised post-deep-dive).
GATES: dict[int, list[dict]] = {
    0: [
        {"dataset": "librispeech-clean-v1", "metric": "wer_final", "op": "<=", "value": 5.0},
    ],
    1: [
        # Targets assume AuT-init path. Tighten/loosen via --target-path mimi.
        {"dataset": "librispeech-clean-v1", "metric": "wer_final", "op": "<=", "value": 2.5},
        {"dataset": "librispeech-other-v1", "metric": "wer_final", "op": "<=", "value": 6.0},
    ],
    2: [
        {"dataset": "librispeech-clean-v1", "metric": "wer_streaming_minus_offline", "op": "<=", "value": 1.0},
    ],
    3: [
        {"dataset": "ami-eval-v1", "metric": "endpoint_latency_p50_ms", "op": "<=", "value": 400},
        {"dataset": "ami-eval-v1", "metric": "endpoint_latency_p95_ms", "op": "<=", "value": 800},
        {"dataset": "ami-eval-v1", "metric": "eager_end_latency_p50_ms", "op": "<=", "value": 200},
        {"dataset": "disfluency-v1", "metric": "false_endpoint_premature_rate", "op": "<=", "value": 0.05},
        {"dataset": "librispeech-clean-v1", "metric": "wer_final", "op": "<=", "value": 5.0},
    ],
    4: [
        {"dataset": "hotword-stress-v1", "metric": "recall_at_relevant", "op": ">=", "value": 0.80},
        {"dataset": "hotword-stress-v1", "metric": "hallucination_at_distractor", "op": "<=", "value": 0.03},
        {"dataset": "librispeech-clean-v1", "metric": "wer_final", "op": "<=", "value": 5.0},
    ],
}


def _cmp(value, op: str, target) -> bool:
    return {
        "<=": value <= target,
        ">=": value >= target,
        "<": value < target,
        ">": value > target,
        "==": value == target,
    }[op]


def run_gate(phase: int, results_path: str) -> int:
    results = json.loads(Path(results_path).read_text())
    predicates = GATES.get(phase)
    if not predicates:
        print(f"FAIL: no gate defined for phase {phase}", file=sys.stderr)
        return 1
    failures = []
    for pred in predicates:
        ds, metric, op, target = pred["dataset"], pred["metric"], pred["op"], pred["value"]
        try:
            actual = results["datasets"][ds][metric]
        except KeyError:
            failures.append(f"missing: {ds}/{metric}")
            continue
        if not _cmp(actual, op, target):
            failures.append(f"{ds}/{metric} = {actual} (need {op} {target})")
    if failures:
        print(f"GATE FAILED for phase {phase}:")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print(f"GATE PASSED for phase {phase}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, required=True)
    p.add_argument("--results", required=True)
    args = p.parse_args()
    sys.exit(run_gate(args.phase, args.results))


if __name__ == "__main__":
    main()
