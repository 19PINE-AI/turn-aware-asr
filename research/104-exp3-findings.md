# Exp-3: schema ablation — which component of the causal recipe does the work?

**Date:** 2026-07-06. Ablation on the pure-endpointing (v9) pool, same modern recipe,
iso-count (drop a schema, upsample remainder to the same total). All arms trained on the
115 train meetings; scored on the 56 held-out eval meetings (25-stretch dev set). best.pt
per arm (holdout-composite selected; NOT re-selected on the replay set).

| arm | gated recall | gated false/min | P50 | resume | ungated silence-fires | h=1 false/min | h=1 candidates cancelled |
|---|---|---|---|---|---|---|---|
| baseline (full v9) | 0.948 | 1.08 | 0.25 | 0.93 | 8 | 0.00 | 10 |
| A −minimal-pair (drop #2) | 0.906 | **5.28** | 0.30 | 0.93 | 5 | 0.86 | **34** |
| B −silence (drop #7-8) | 0.969 | 1.94 | 0.27 | 1.00 | **1458** | 0.22 | 12 |
| C −pause-pair (drop #5) | 0.906 | 0.43 | 0.27 | 0.93 | 8 | 0.22 | 4 |

("ungated silence-fires" = dup_or_silence_fires_total, energy gate OFF; the primary metric
for arm B. "h=1 candidates cancelled" = fires the confirm horizon retires — the crutch metric.)

## Verdict per arm (each has a SPECIFIC predicted failure)
- **A −minimal-pair: PREDICTED PATHOLOGY CONFIRMED.** Dropping the no-tail twin of schema 1
  makes silence optional again: gated false fires jump 1.08 -> 5.28 per speech-minute (5x),
  and the confirm horizon has to cancel 34 candidates at h=1 (vs 10 for baseline) to stay
  usable — the row-4 pathology of Table composition returns, and the crutch it was meant to
  retire comes back. The minimal pair is load-bearing.
- **B −silence: PREDICTED PATHOLOGY CONFIRMED (strongest).** Dropping the pure/leading-silence
  schemas destroys silence-competence: ungated the model sprays 1458 markers on silence
  (vs 8 for baseline, ~180x), exactly the "hallucinate on silence" failure the benchmark's
  silence measurement predicts. The energy gate masks it in the gated numbers (false/min only
  1.94), which is why the ungated metric is the honest one. The silence schemas are what make
  the model silence-competent; the gate is belt-and-suspenders.
- **C −pause-pair: PREDICTED PATHOLOGY NOT REPRODUCED (partial redundancy).** Dropping the
  incomplete-A hold schema does NOT raise internal/continuation false fires; instead the model
  becomes mildly more conservative (recall 0.948 -> 0.906, false fires 1.08 -> 0.43, resume
  unchanged). Pause discrimination is largely covered by schema 4's complete-A gaps, so the
  pause-pair mainly contributes firing confidence at pause-ends (recall) rather than
  false-fire suppression — the "≈ baseline" branch the pre-registered prediction anticipated.

## Takeaway
The recipe -> pathology mapping is measured, not narrative: the two constructions the paper
credits most (minimal pair, silence schemas) each reproduce a *specific, different* failure
when removed (premature firing / silence hallucination), while the pause-pair is partially
redundant with the complete-A gap schema. Report per-arm, not aggregate: the arms fail in
different places, which is the point. Artifacts: research/104-replay-{base,A,B,C}-{gated,ungated,confirm1}.json.
