# Gap 3: oscillation reproduces under a second seed (multi-seed Fig. 1 left)

**Date:** 2026-07-06. Rebuilt the deleted opposed-pools data chain
(scripts/rebuild_v8_opposed.sh: v2->v3->v4->v5->v6->v8 from raw LibriSpeech+AMI),
then trained seed-1 with the HISTORICAL recipe (constant LR, --score-spec legacy),
changing ONLY the seed. Snapshots saved at every eval (also feed exp-2).

## Oscillation fingerprint (holdout argmax accuracy, per training step)
| run | evals | fire range | std(fire) | corr(fire, hold) |
|---|---|---|---|---|
| historical seed-0 (retained eval_log) | 12 | 0.10-1.00 | 0.296 | -0.95 |
| seed-1 rerun (this work) | 11 | 0.30-0.90 | 0.202 | -0.88 |

Both runs show the same anti-phase mode-circulation: the fire-class and hold-class
accuracies move in opposition (corr ~-0.9), bouncing for the entire run — no single
policy satisfies both opposed pools. The oscillation is a property of the opposed-pools
LABELS, reproducible across seeds, not a seed-specific instability.

## Paper impact
Upgrades Fig. oscillation (left) from single-run to multi-seed and DELETES the
Limitations caveat "the oscillation half is single-run because the historical
opposed-pools data was not retained." Data is now regenerable (rebuild_v8_opposed.sh);
seed-1 checkpoints + eval_log at checkpoints/semantic_endpoint_v8_seed1_es/.
Contrast: the causal recipe climbs monotonically (right panel) — reproduced under a
2nd seed already (checkpoints/causal_seed1). Both halves are now multi-seed.
