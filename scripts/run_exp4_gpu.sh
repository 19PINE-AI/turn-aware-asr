#!/bin/bash
set -uo pipefail; cd "$(dirname "$0")/.."; export EXT_MODEL_DIR=data/external_models PYTHONUNBUFFERED=1
echo "=== parakeet_eou dev25 $(date -Is) ===" && .venv_ext_nemo/bin/python -m eval.external.harness --adapter parakeet_eou --n-stretches 25 --seed 0 --out research/95-exp4-parakeet-dev25.json 2>&1 | tail -3
echo "=== kyutai_vad dev25 $(date -Is) ===" && KYUTAI_VAD_HEAD=2 .venv_ext_moshi/bin/python -m eval.external.harness --adapter kyutai_vad --n-stretches 25 --seed 0 --out research/95-exp4-kyutai-dev25.json 2>&1 | tail -3
echo "=== DONE exp4 GPU $(date -Is) ==="
