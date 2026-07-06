#!/bin/bash
# exp-3 ablation pools (iso-count = drop a schema, upsample remainder to same
# total; volume-controlled). Built on the v9 pure-endpointing pool (cache present
# -> CPU). Arms per paper Table schemas: A=drop #2(complete_nosil),
# B=drop #7-8(silence_only,lead_sil), C=drop #5(pair_hold).
set -euo pipefail; cd "$(dirname "$0")/.."; PY=.venv/bin/python; export PYTHONUNBUFFERED=1
echo "=== arm A -minimal-pair $(date -Is) ===" && $PY -m eval.build_v9_training_data --drop-schemas complete_nosil --iso-count --out data/semantic_endpoint_abl_A/data.pt --transcripts-cache data/semantic_endpoint_v9/ls_transcripts.json 2>&1 | tail -2
echo "=== arm B -silence $(date -Is) ===" && $PY -m eval.build_v9_training_data --drop-schemas silence_only,lead_sil --iso-count --out data/semantic_endpoint_abl_B/data.pt --transcripts-cache data/semantic_endpoint_v9/ls_transcripts.json 2>&1 | tail -2
echo "=== arm C -pause-pair $(date -Is) ===" && $PY -m eval.build_v9_training_data --drop-schemas pair_hold --iso-count --out data/semantic_endpoint_abl_C/data.pt --transcripts-cache data/semantic_endpoint_v9/ls_transcripts.json 2>&1 | tail -2
echo "=== DONE exp3 pools $(date -Is) ==="
