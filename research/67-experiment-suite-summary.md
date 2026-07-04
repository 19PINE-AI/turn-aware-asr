# Experiment suite summary + E4/E5 disposition (2026-07-04)

This session took the project from "endpoint work stalled on a Pareto
frontier" to "a validated, servable streaming VAD+ASR+biasing system." The
chain of results:

| # | Experiment | Result | File |
|---|---|---|---|
| — | Smoke test | The v3/v5/v8 frontier was an **eval artifact** (silence appended: v8 single 0.10→1.00) | research/57 |
| — | Unified replay eval | Deployment-matched streaming eval; two eval contradictions fixed | research/58 |
| — | Replay re-baseline | Ranking **inverts** (v8 best, v3 unusable); silence-incompetence → energy gate; phrase-vs-turn policy | research/61 |
| — | v9 training | Causal label spec **eliminates the oscillation**; trajectory 0.900→0.963 monotonic, both modes at ceiling | research/62 |
| — | v9 replay | **v9+gate dominates** v5/v8 (recall .969, false .3/min, WER 1.43) with no confirm/force-flush crutch | research/62 |
| E1 | Merge + vLLM serve | v9 LoRA → standard checkpoint; vLLM fires markers correctly (causal behavior on real infra) | research/65 |
| E2 | Serving primitive | Bounded re-feed (3.8 % WER) vs chunked placeholders (88 %): **use bounded re-feed** | research/65 |
| E3 | Streaming on vLLM | Endpoint metrics **reproduce** at **84 ms/chunk** vs 455 ms simulator (5.4×, ~6× real-time) | research/65 |
| E4 | Context biasing | Earnings-22 hotword recall **75 %→95.5 % (+20.5 pp)**, hallucination 5.1 % | research/63 |

**The v1 system is complete and validated end-to-end**: `Qwen3-ASR-0.6B +
v9 endpoint LoRA (merged) + energy gate + <CTX> prefix`, served on vLLM
with bounded re-feed, gives real-conversation endpoint detection
(0.95 recall, 0.39 s P50, ~0 false fires) at real-time speed, plus a large
context-biasing gain on entity-dense audio — a differentiated capability
neither Qwen3-ASR (no endpointing, zero biasing numbers) nor Kyutai (no
context prefix) has.

## E4 pinned-CTX-sink — deferred, and why it's not a v1 blocker

The novel serving claim in research/60 is "context biasing that survives
session length via a pinned attention sink." The **core biasing capability
is already validated** (Earnings-22, +20.5 pp). The pinned-sink refinement
matters only for the **continuous-transcription variant** (minute-scale
decode with no utterance flush), where the LLM-side context grows
unbounded. In the shipping endpoint configuration this problem does not
arise: E2/E3 established bounded re-feed, which flushes per utterance and
**re-feeds the `<CTX>` prefix every segment** — so the prefix is always
resident and attended without any windowing. The pinned sink is a research
contribution for a different product surface, not a requirement for v1.

Concrete implementation path (for the continuous-captioning follow-up):
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

## E5 concurrency / admission — blocked on a solo GPU

Sessions-per-Blackwell (N\*) via the Metronome gateway + worker + AIMD
admission requires the GPU **uncontended** to produce a calibrated number.
This box runs a co-tenant project that continuously loads/unloads vLLM
servers; it OOM-killed v9 training 4× and every GPU eval intermittently
(all survived via retry/resume). Any concurrency ceiling measured under
that churn would be an artifact of the neighbor, not the system. E5 is
deferred to a solo-GPU window; the harness exists (`~/metronome` gateway +
worker + `worker_integration/` ASR session + the merged checkpoint), so it
is a run-it-when-the-box-is-free task, not new engineering.

What E3 *does* bound: a single stream costs 84 ms median / 413 ms P95 of
compute per 0.5 s frame at 0.15 GPU-utilization — so the deadline (not
memory, with bounded re-feed) will set N\*, exactly the compute-limited
regime the Metronome paper predicts for bounded state.

## Net

Feasible high-value experiments: **done** (E1–E3, biasing). Deep-infra
experiments: **E4 deferred** (not a v1 blocker; concrete path documented),
**E5 blocked** on solo GPU (harness ready). The endpoint research question
that stalled the project is answered, and the answer is validated on
production infrastructure.
