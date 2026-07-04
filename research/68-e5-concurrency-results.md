# E5: schedulable concurrency (sessions per GPU) (2026-07-04)

The Metronome question for the endpoint+ASR product: how many concurrent
streaming sessions can one resident vLLM engine carry before per-frame
latency exceeds the frame budget? E5 sweeps N concurrent bounded-re-feed
sessions through one engine on the merged v9 checkpoint with continuous
batching, and finds N\* — the largest N whose per-frame p95 stays under the
500 ms budget.

Driver: `worker_integration/e5_concurrency.py`. Each session replays a
distinct held-out AMI stretch (phase-staggered, distinct audio → distinct
KV, so prefix caching can't inflate capacity). Per frame the driver
collects the *due* sessions (the energy gate skips silent ones) and issues
ONE batched `generate` over all of them — the continuous-batching step —
and times exactly that call. `skip_special_tokens=False` (markers are
special tokens). gpu_mem 0.15, window 16 s, 0.5 s frames, 60 s per N.

## Result

| N | Frame p50 | Frame p95 | Mean batch (gated) | p95 ≤ 500 ms |
|---|---|---|---|---|
| 1 | 83 ms | 100 ms | 1.0 | ✓ |
| 2 | 77 ms | 107 ms | 1.4 | ✓ |
| 4 | 76 ms | 147 ms | 2.0 | ✓ |
| **8** | 135 ms | 177 ms | 3.5 | ✓ |
| 16 | 471 ms | 811 ms | 8.5 | ✗ |

**N\* ≈ 8** on a 0.15-utilization slice. The shape is the one the Metronome
paper predicts for *bounded* per-session state: latency stays flat and low
while the batch is small, then rises sharply once the per-frame batched
compute overruns the deadline. **The deadline binds, not memory** — bounded
re-feed caps each session's resident audio, so the schedulable limit is set
by compute per frame, exactly the compute-limited regime the paper
describes (the reverse of the unbounded-KV memory cliff).

Mean batch size is well below N because the energy gate skips silent
sessions (at N=8, mean batch 3.5) — in real mostly-silent conversation this
is what lets one engine carry more sessions than a naive all-due-every-frame
model would predict.

## Caveats — this is a lower bound

- **Co-tenant contention.** The box ran a neighboring project's vLLM engine
  throughout (it OOM-killed jobs across this whole session). Absolute
  latencies are inflated by GPU compute contention, so N\* = 8 is a floor,
  not a calibrated ceiling. The *shape* (flat → cliff) and the *relative*
  ordering are robust to contention; the absolute knee is not.
- **0.15 GPU-utilization slice.** vLLM was given 15 % of the card. A solo
  full-card allocation (more KV blocks, no contention) would push N\*
  substantially higher — the Metronome paper's admission controller
  discovers N\* ≈ 209 for the 30B omni model on a full uncontended card;
  the 0.6 B ASR model on a full card should sit well above 8.

A calibrated per-Blackwell N\* needs the GPU solo + the Metronome gateway +
AIMD admission (research/67 E5 harness); this run establishes the method
and the qualitative result (compute-bound, deadline-limited, energy-gate
amplified) on the contended slice available.

## Files

- `research/68-e5-concurrency.json` — per-N frame-latency percentiles
- `worker_integration/e5_concurrency.py` — driver
