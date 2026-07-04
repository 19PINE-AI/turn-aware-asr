# v9 results: the causal label spec eliminates the oscillation (2026-07-04)

v9 tests the single hypothesis behind this whole line of work: the
v1–v8 "Pareto frontier" (research/50) was not a model limitation but
**contradictory supervision**. v9 changes only the data — one causal
labeling rule (research/59) — and holds the LoRA recipe, base model, and
trainer identical to v5–v8. The result is decisive.

## The headline: no oscillation

v8's holdout score bounced between two attractors it could never
reconcile — fire-mode (single ≈ 1.0, no-fire ≈ 0.4) and no-fire-mode
(single ≈ 0.1, no-fire ≈ 1.0) — every few thousand steps
(research/50). That swing was the fingerprint of labels that assigned
opposite targets to causally identical audio.

v9's holdout trajectory (composite = mean of per-schema exact-match
rates; 80 held-out examples, 10/schema):

```
step  score  single_sil  silence_only  complete_nosil  truncated  pair_fire  pair_hold
      (FIRE)  ----------------- NO-FIRE -----------------  --- FIRE ---
 1500  0.900   0.90        1.00          0.90            0.90      0.70       0.90
 3000  0.925   1.00        1.00          0.90            0.90      0.70       0.90
 4500  0.938   1.00        1.00          1.00            1.00      0.80       0.80
 6000  0.950   1.00        1.00          1.00            0.90      0.80       0.90
 7500  0.963   1.00        1.00          1.00            1.00      0.80       0.90  ← best
 9000  0.938   1.00        1.00          1.00            0.90      0.80       0.80
10500  0.950   1.00        1.00          1.00            0.90      0.80       0.90
12000  0.913   1.00        1.00          1.00            0.90      0.80       0.70
13500  0.925   1.00        1.00          0.90            1.00      0.60       0.90
```

Two things never happen that always happened in v8:

1. **`single_sil` (fire) is 1.00 from step 3000 onward AND the no-fire
   schemas (`silence_only`, `complete_nosil`, `truncated`) are
   simultaneously at 0.90–1.00 — for the entire run.** The model holds
   both modes at ceiling at once. v8 structurally could not: one global
   "fire-or-not" threshold on causally-identical audio forced a trade.
2. **The composite climbs monotonically to a plateau** (0.900 → 0.963 by
   step 7500) then wobbles in a ±0.025 noise band. Every dip is one or
   two holdout examples flipping in `pair_hold`/`truncated`/`pair_fire`
   (0.09 granularity at 10/schema), never a mode collapse. There is no
   oscillation.

`silence_only = 1.00` throughout also confirms the fix for the
silence-incompetence the replay eval exposed in v5/v8 (research/61): the
model correctly emits nothing on pure silence.

**Conclusion: the frontier was the labels.** Once the same complete
utterance is labeled `complete_nosil` (no fire) or `single_sil` (fire)
according only to the *observable* trailing silence, a single checkpoint
learns the boundary that eight prior experiments could not. The
architecture and data scale were never the bottleneck.

The one schema that never reaches ceiling is `pair_fire` (0.60–0.80):
exact 2-marker sequences (A M B M) under a strict count-match scorer.
Whether that reflects a real weakness or scorer strictness is what the
replay eval — tolerance-based boundary recall — answers below.

## Streaming-replay eval (deployment-matched, research/58)

best.pt = step 7500. Same protocol and stretches as research/61
(25 held-out AMI single-speaker stretches, energy gate, committed-prefix).

| Arm | Recall | P50 | P95 | False /min | WER med | resume |
|---|---|---|---|---|---|---|
| v5 + gate + confirm1 (ref) | 0.927 | 0.68 s | 1.27 s | 3.2 | 0.31 | 0.85 |
| v8 + gate + confirm1 (ref) | 0.938 | 0.77 s | 1.06 s | 0.0 | 0.27* | 0.85 |
| **v9 + gate** | _pending_ | | | | | |
| **v9 + gate + confirm1** | _pending_ | | | | | |

_(replay arms running; table filled when they land.)_

## Training-resilience note

The shared box's global OOM killer (confirmed in the kernel journal)
killed v9 training at random steps under RAM pressure from a neighboring
project's vLLM servers — 4 kills over the run (rc=137). An auto-restart
wrapper + `--resume` (trainable + optimizer + step + best_score +
eval_log, saved every 750 steps) absorbed every kill with ≤750 steps
lost each time; the trajectory above is stitched across 5 attempts and is
continuous. Early stopping was defeated by the resume resetting
`evals_since_best`, so training was stopped manually once the plateau was
unambiguous (best at step 7500, flat for 6 evals).

## Files

- `checkpoints/semantic_endpoint_v9_es/best.pt` — step 7500
- `checkpoints/semantic_endpoint_v9_es/eval_log.json` — full trajectory
- `data/semantic_endpoint_v9/data.pt` — 11,950 examples, 8 schemas
- `research/62-replay-v9{gate,gate-confirm1}.json` — replay arms
- `eval/build_v9_training_data.py`, `src/train_semantic_endpoint.py`
  (`--score-spec v9`, `--resume`)
