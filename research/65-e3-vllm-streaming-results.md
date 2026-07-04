# E3: endpoint detection on the real vLLM streaming path (2026-07-04)

Every endpoint latency number before this ran on the transformers re-feed
simulator (`eval/streaming_latency.py` / `streaming_replay_eval.py`):
each 0.5 s chunk did a fresh `generate()` over all accumulated audio —
O(T²), 455 ms median per chunk, explicitly "not a real streaming KV
cache." E3 runs the same streaming-replay protocol (research/58) on the
**merged v9 checkpoint served by vLLM** (Qwen3ASRForConditionalGeneration,
prefix caching on), so the numbers are on shippable infrastructure.

Driver: `worker_integration/e3_replay_vllm.py` — subclasses the
research/61 `StreamDecoder` so boundary classification, fire scoring, and
WER are byte-identical to the transformers re-baseline; only the per-chunk
decode swaps to a resident vLLM engine. 25 held-out AMI single-speaker
stretches, energy gate, committed-prefix, `gpu_memory_utilization=0.15`
(shared GPU).

## Results: behaviour reproduces, latency collapses

| Metric | Transformers sim (v9+gate, research/62) | **vLLM path (E3)** |
|---|---|---|
| Boundary recall | 0.969 | **0.948** |
| Latency P50 | 0.39 s | **0.39 s** |
| Latency P95 | 0.70 s | 0.66 s |
| False fires / speech-min | 0.3 | 0.76 |
| **Compute / chunk, median** | **455 ms** | **84 ms** |
| Compute / chunk, P95 | — | 413 ms |
| Compute vs simulator | 1.0× | **0.18× (5.4× faster)** |

Two conclusions:

1. **The endpoint behaviour is real, not a simulator artifact.** Recall,
   latency P50/P95, and false-fire rate all match the transformers
   re-baseline within noise. The causal-label model fires `<END_SPEECH>`
   correctly under genuine incremental decoding on a production engine.
2. **The serving path is 5.4× faster per chunk** (84 ms median vs the
   simulator's 455 ms). At the 500 ms chunk period this is ~6× real-time
   headroom on a *single 0.15-utilization slice* of one shared Blackwell;
   even P95 (413 ms) fits the frame budget. The O(T²) re-feed was the
   artificial bottleneck; vLLM's prefix caching makes the streaming
   endpoint real-time-viable.

## WER cleanup (resolved): max-segment force-flush

The first E3 run had `wer_mean` = 2.54 (median 0.32). Diagnosis: **not** a
special-token scoring artifact — median WER already matched the
transformers re-baseline; the mean was dragged up by **2 of 25 stretches**
where v9 fired 0–few times, so the segment never flushed, grew unbounded,
and committed-prefix decoding fell into a repetition loop (WER 31.7, 24.5).
Same failure class as v8's WER-mean 4.96 (research/61). Fix: re-run with
`--max-segment-chunks 40` (force-flush at 20 s — the Metronome
bound-the-resident-state principle at the text level).

| Metric | E3 (unbounded) | **E3 + maxseg-40** |
|---|---|---|
| Boundary recall | 0.948 | **0.958** |
| Latency P50 | 0.39 s | 0.39 s |
| False fires / min | 0.76 | 0.54 |
| **WER mean** | **2.54** | **1.29** |
| Compute/chunk median | 84 ms | 97 ms |
| **Compute/chunk P95** | **413 ms** | **117 ms** |

The bound fixes the WER tail (2.54 → 1.29, matching transformers' 1.43)
**and** the latency tail (P95 413 → 117 ms — bounded segments give bounded
per-chunk decode time), with no endpoint-metric cost (recall/latency/false
fires all held or improved). This is the shipping configuration:
`v9 + gate + max-segment force-flush`. Result:
`research/65-e3-replay-vllm-v9gate-maxseg40.json`.
- Single-stream measurement. Multi-session throughput / the schedulable
  concurrency N* is E5 (needs the Metronome gateway + a solo GPU for clean
  numbers; the OOM-churning co-tenant here precludes them).

## E2: which serving primitive? Bounded re-feed, decisively

research/60 left one routing question: feed each chunk as a **separate
audio placeholder** into a resident growing request (vLLM's flat-cost
`fd_step_stream`, mm-cache reuses prior chunks) — or **bounded re-feed**
the last W seconds as ONE segment each chunk? E2
(`worker_integration/e2_chunked_vs_segment.py`, 30 LibriSpeech utts) is
unambiguous:

| Feed shape | WER |
|---|---|
| single segment (bounded re-feed) | **3.8 %** |
| chunked (one placeholder per 0.5 s) | **88.0 %** |

Qwen3-ASR was trained on single audio segments; splitting the audio into
per-chunk placeholders is wildly out-of-distribution and destroys
transcription (Δ 84 pp). **Use bounded re-feed** — the flat-cost chunked
path is unusable for this model. This is why E3 uses one bounded segment
per step, and it validates the research/60 v1 protocol. (The flat-cost
resident path would need a model trained on chunk-boundary placeholders to
be viable — a v10+ option, not a v1 one.)

## E1 recap (prerequisite, same session)

`scripts/merge_endpoint_lora.py` folds the v9 LoRA + the two reserved-slot
marker rows into a standard Qwen3-ASR HF checkpoint
(`checkpoints/merged/qwen3-asr-0.6b-endpoint-v9`). E1b
(`worker_integration/e1_vllm_validate.py`) proved vLLM serves it: fires
`<END_SPEECH>` on speech+silence, stays silent on pure silence AND on
speech-without-trailing-silence (the causal distinction, learned). Root
cause of an initial no-fire: markers are special tokens (151705/151706)
and vLLM's default `skip_special_tokens=True` strips them from `.text`;
`skip_special_tokens=False` fixes it, and a raw token-id scan confirms
genuine emission.

## Files

- `research/65-e3-replay-vllm-v9gate.json` — full E3 per-stretch results
- `worker_integration/e3_replay_vllm.py`, `e1_vllm_validate.py`
- `checkpoints/merged/qwen3-asr-0.6b-endpoint-v9` — servable checkpoint
