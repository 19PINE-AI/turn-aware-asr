# Experiment suite summary (2026-07-04)

This session took the project from "endpoint work stalled on a Pareto
frontier" to "a validated, servable streaming VAD+ASR+biasing system," with
the full serving stack (E1–E5) measured on real infrastructure. The chain:

| # | Experiment | Result | File |
|---|---|---|---|
| — | Smoke test | The v3/v5/v8 frontier was an **eval artifact** (silence appended: v8 single 0.10→1.00) | research/57 |
| — | Unified replay eval | Deployment-matched streaming eval; two eval contradictions fixed | research/58 |
| — | Replay re-baseline | Ranking **inverts** (v8 best, v3 unusable); silence-incompetence → energy gate; phrase-vs-turn policy | research/61 |
| — | v9 training | Causal label spec **eliminates the oscillation**; trajectory 0.900→0.963 monotonic, both modes at ceiling | research/62 |
| — | v9 replay | **v9+gate dominates** v5/v8 (recall .969, false .3/min, WER 1.43) with no confirm/force-flush crutch | research/62 |
| E1 | Merge + vLLM serve | v9 LoRA → standard checkpoint; vLLM fires markers correctly (causal behavior on real infra) | research/65 |
| E2 | Serving primitive | Bounded re-feed (3.8 % WER) vs chunked placeholders (88 %): **use bounded re-feed** | research/65 |
| E3 | Streaming on vLLM | Endpoint metrics **reproduce** at **84 ms/chunk** vs 455 ms simulator (5.4×); WER-fixed to 1.29 via max-segment flush | research/65 |
| E4a | Context biasing | Earnings-22 hotword recall **66.7 %→95.6 % (+28.9 pp)**, hallucination 3.7 % (LLM entities) | research/63 |
| E4b | Pinned-CTX / session length | Biasing advantage **flat ~+30 pp across 0–12 s** intervening audio — survives session length | research/69 |
| E5 | Concurrency / N\* | **N\* ≈ 8** sessions per 0.15-GPU slice; deadline-bound (Metronome shape), contention-caveated lower bound | research/68 |

**The v1 system is complete and validated end-to-end**: `Qwen3-ASR-0.6B +
v9 endpoint LoRA (merged) + energy gate + <CTX> prefix`, served on vLLM
with bounded re-feed, gives real-conversation endpoint detection
(0.95 recall, 0.39 s P50, ~0 false fires) at real-time speed (84 ms/chunk),
plus a large, session-length-durable context-biasing gain on entity-dense
audio (+28.9 pp) — a differentiated capability neither Qwen3-ASR (no
endpointing, zero biasing numbers) nor Kyutai (no context prefix) has.

Entity extraction for the biasing evals uses an LLM (Claude Haiku 4.5,
`eval/llm_entities.py`, Gemini fallback) rather than the dated spaCy NER —
cleaner proper-noun hotwords, which raised the measured uplift from the
spaCy +20.5 pp to +28.9 pp.

## E4/E5 — done; the one remaining refinement

Both are now measured (E4b flat curve, E5 N\*≈8). The pinned-CTX
**capability** is proven at the application layer (bounded re-feed re-pins
`<CTX>` every frame; biasing doesn't decay with session length). The one
open piece is the **in-engine efficiency version** — a windowed KV that
pins `[0,S)` and slides audio out, avoiding the per-frame CTX re-encode —
which matters only for the continuous-transcription variant and needs the
dense-Qwen3 SWA vLLM patch:
1. Dense-`Qwen3ForCausalLM` sliding-window vLLM patch — a sibling to
   `~/metronome/metronome/patches/qwen3_swa.py` (which patches
   `qwen3_moe.Attention`); Qwen3-ASR's thinker is dense `qwen3.Attention`.
2. Pin the first S tokens (the `<CTX>` block; `qwen3asr_stream.py::
   pinned_prefix_len()` already exposes S) via the KV-manager pin + the
   Triton union mask `[0,S)∪[t−W,t]` from the Metronome paper.
3. Long-session eval: 10-min continuous stream, hotword recall vs session
   age, three arms {no window, window+pin, window−pin} — the Metronome
   `longhorizon` methodology transplanted to ASR. Expect: window−pin
   decays as CTX slides out; window+pin holds flat (the lead chart).
4. Requires quality revalidation of the sink on the 0.6 B backbone
   (Metronome validated it on the 30 B).

Estimated ~1 day; gated on the dense SWA patch. Novel and publishable, but
orthogonal to shipping the endpoint+biasing v1.

## E5 concurrency / admission — measured (contention-caveated)

Ran the concurrency sweep (research/68): N\* ≈ 8 sessions per 0.15-GPU
slice before per-frame p95 exceeds the 500 ms budget, with the exact
Metronome bounded-state shape — flat/low to N=8, cliff at N=16 — so the
**deadline binds, not memory**. The energy gate keeps the mean batch below
N (3.5 at N=8), letting one engine pack more mostly-silent sessions than a
naive model predicts.

This N\* is a **lower bound**: the box's co-tenant project (which OOM-killed
v9 training 4× and every GPU eval intermittently — all survived via
retry/resume) contends for compute, and vLLM ran on a 0.15-utilization
slice. A calibrated per-Blackwell number needs the GPU solo + the Metronome
gateway + AIMD admission (the paper discovers N\* ≈ 209 for the 30B omni
model on a full uncontended card; the 0.6B ASR model should sit well above
8). The harness exists (`~/metronome` + `worker_integration/`); the solo-GPU
calibration is a run-when-free task, not new engineering.

## Net

**All experiments done and validated on real infrastructure** (E1–E5 +
biasing). The endpoint research question that stalled the project is
answered (causal labels; v9+gate); the serving stack is measured (vLLM
bounded re-feed, 84 ms/chunk, N\*≈8/slice); the biasing differentiator is
quantified (+28.9 pp, session-length-durable). The single remaining
refinement is the in-engine windowed-KV pinned sink (efficiency-only, for
the continuous-captioning variant) — a ~1-day follow-up gated on the dense
SWA patch, orthogonal to shipping v1.
