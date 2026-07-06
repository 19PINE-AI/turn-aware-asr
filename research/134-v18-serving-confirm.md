# v18 release: serving re-measurement (#1) + confirm-h=1 operating point (#2)

## #1 — vLLM serving on the released v18 checkpoint (sglang-venv, vLLM 0.19.0)
E3 per-chunk latency (research/132-e3-serving-v18.json, --max-segment-chunks 40,
25 stretches, energy gate): median 21.2 ms/chunk, P95 30.8 ms — 21x faster than the
455 ms transformers simulator. (Original dev-time number 84 ms was under a shared-GPU
co-tenant; same vLLM 0.19, same merged architecture, so the gap is pure contention.)
E5 concurrency (research/133-e5-concurrency-v18.json): per-frame P95 flat and within the
500 ms budget across N=1..16 (28.1 / 30.9 / 36.7 / 44.0 / 77.5 ms) — N*=16, no cliff on
the idle GPU (the dev-time shared-GPU run cliffed at 16). Confirms rank-32 does not hurt
serving: after merging, v18 is architecturally identical to the endpoint checkpoint.
Paper: §serving sentence added (both idle 21 ms and co-tenant 84 ms conditions labeled).

## #2 — confirm-horizon dial on the released model (research/124-replay-v18-dev25-h1.json)
Dev-25 set, confirm-silent-chunks=1: boundary recall 0.958, false fires 0.00/speech-min,
P50 0.89 s, resume 0.93. The released model exposes the same eager/conservative dial as the
pure endpointer (tab:main): one-chunk horizon -> zero false fires at slightly higher latency.
Paper: §sec:results four-behaviors paragraph updated.
