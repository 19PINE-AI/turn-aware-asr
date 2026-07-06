#!/bin/bash
# Gap 3: second seed of the opposed-pools (v8) run to test whether the
# OSCILLATION fingerprint reproduces. Matches the HISTORICAL v8 recipe
# (constant LR, default wd, --score-spec legacy) so the ONLY change vs the
# retained checkpoints/semantic_endpoint_v8_es/eval_log.json is the seed.
# Single-process (no resilient retry wrapper). Trains to step 18000 to match
# the historical trajectory length.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
.venv/bin/python -m src.train_semantic_endpoint \
    --data data/semantic_endpoint_v8/data.pt \
    --steps 18000 --bsz 8 --eval-every 1500 --eval-holdout-size 60 \
    --score-spec legacy --seed 1 --snapshot-evals \
    --checkpoint-dir checkpoints/semantic_endpoint_v8_seed1_es --resume
    # --snapshot-evals retains a per-eval checkpoint trajectory -> also enables
    # exp-2 (marker-token probability across the oscillation).
