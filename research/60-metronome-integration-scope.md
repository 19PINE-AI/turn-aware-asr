# Metronome × streaming-vad-asr — integration scoping (2026-07-04)

Goal: serve the endpoint-LoRA Qwen3-ASR model as a real streaming service
on the Metronome stack (vLLM-realtime resident-ish sessions, bounded KV,
admission control), replacing the transformers re-feed simulator
(O(T²), 455 ms/chunk) that all v1–v8 latency numbers were measured on.

## Verified facts (this machine)

1. **vLLM 0.19.0** (Metronome's env, `~/.local/lib/python3.10`) natively
   registers `Qwen3ASRForConditionalGeneration` AND
   `Qwen3ASRRealtimeGeneration`
   (`vllm/model_executor/models/qwen3_asr_realtime.py`). The realtime
   class is **segment-based**: buffers 5 s of audio, each segment becomes
   a fresh request — a transcription pipe, not a resident session with
   endpoint semantics. Useful as a reference path, not the target.
2. **qwen-asr official streaming** (`qwen_asr/inference/qwen3_asr.py`,
   vLLM-only): re-feed ALL accumulated audio as ONE segment per chunk +
   **committed-prefix decoding** (prompt = base + previously decoded text
   with last K=5 tokens rolled back). In-distribution with training
   (single audio segment). Costs: AuT re-encodes the full buffer each
   chunk (linear, 180 M encoder — cheap); LLM re-prefills all audio
   positions each chunk (audio embeds change → prefix cache only covers
   the text prompt). At 12.5 Hz this is ≤ 750 positions for a 60 s buffer
   — fine per-session, but per-frame cost GROWS with session length:
   exactly the unbounded-state regime the Metronome paper diagnoses.
3. **Metronome `fd_step_stream`** (`metronome/backends/vllm_backend.py`):
   resident growing list of audio chunks, one placeholder group per chunk;
   vLLM mm-processor cache reuses prior chunks' AuT encodings and prefix
   caching reuses their KV → **flat per-frame cost**. Risk: Qwen3-ASR
   never saw N separate audio segments in one message — chunked encoding
   is out-of-distribution (AuT window boundaries fall at chunk edges).
4. **LoRA merge is trivial**: the two marker tokens sit in RESERVED vocab
   slots (151705/151706 < 151936 — `extend_tokenizer_and_model` re-inits
   existing rows, no vocab growth). Merging LoRA deltas
   (W += B·A·α/r on q/k/v/o_proj) + writing the two embed/lm_head rows
   yields a **standard Qwen3-ASR checkpoint** that vLLM serves unmodified,
   markers decode as ordinary tokens. No adapter plumbing needed.
5. **Windowed KV**: Metronome's in-engine SWA patch
   (`metronome/patches/qwen3_swa.py`) targets `qwen3_moe.Attention`; the
   ASR thinker is dense `Qwen3ForCausalLM` (`qwen3.py`) — same one-line
   patch shape, needs a sibling for the dense class. Pinned sinks: the
   KV-manager pin + Triton union mask from the paper (§limits: quality
   boundary revalidation required on this 0.6 B backbone).
6. **Context prefix as pinned sink**: Qwen3-ASR takes context in the
   system prompt (research/16 measured −37 % rel. WER from it). System
   prompt = tokens `[0, S)` of the session → precisely what the sink
   mechanism pins. Hotwords stay attended for the whole session while
   audio history slides out. This is the novel serving-side claim; the
   experiment is hotword recall vs session age, window on/off (Metronome
   §5.3 `longhorizon` methodology transplanted to ASR).

## Recommended serving protocol (synthesis of 2+3)

Committed-prefix + **bounded audio buffer**: per chunk, re-feed only the
last W_audio seconds (e.g. 16 s) as one segment (in-distribution), with
prompt = pinned `<CTX>` + committed transcript tail + rollback. After each
`<END_SPEECH>`, flush the segment and drop its audio from the buffer —
the endpointer itself keeps the buffer short in conversation (utterances,
not sessions, bound the buffer). Per-frame cost is then O(W_audio),
session-length-independent — the Metronome bound realized at the
application layer, no OOD chunked encoding, no engine patch required for
v1. The in-engine window + sinks (fact 5/6) is the v2 refinement that
additionally bounds the LLM-side text KV and keeps `<CTX>` pinned for
minute-scale continuous transcription (streaming captions), where
segments don't flush the buffer.

## Feasibility gates (ordered; each ~0.5 day or less)

| # | Experiment | Decides |
|---|---|---|
| E1 | Merge v5 (later v9) LoRA → HF checkpoint; reproduce smoke-test AMI numbers under transformers, then markers under vLLM offline | merge correctness; vLLM serves the merged model |
| E2 | 30 LS utts: WER single-segment vs 0.5 s-chunked placeholders (`fd_step_stream` format) | whether flat-cost chunked path is usable, or bounded re-feed (recommended protocol) is the path |
| E3 | Unified replay eval (research/58) on the vLLM committed-prefix path; chunk compute vs the 455 ms/chunk simulator | real per-tick latency; replaces simulator numbers |
| E4 | Long-session synthetic stream (10 min) with hotword `<CTX>`: recall vs age, {no window, window+pin, window−pin} | pinned-sink biasing claim; W_audio sizing |
| E5 | Gateway + N concurrent AMI streams, AIMD admission on | sessions-per-Blackwell headline (N*) |

E2 outcome routes the design: if chunked WER ≈ single-segment (< 0.5 pp),
adopt fd_step_stream (flat cost, resident); if not, adopt bounded re-feed
(v1 protocol above) — and note v9+ can *train* with chunk-boundary
placeholders to close the OOD gap later if flat-cost matters.

## Effort estimate

- E1 merge script + parity check: 0.5 day
- Replay-eval driver on vLLM backend (shared with research/58): 1 day
- Worker: ASR prompt template in `worker/stream_server.py::model_prompts`
  + endpoint flush logic: 0.5–1 day
- E4/E5 runs + writeup: 1 day
- In-engine dense-Qwen3 SWA + sink pin (v2): 1–2 days incl. revalidation

Total v1 (E1–E3, shippable replay numbers): **~2–2.5 days**. Full system
story (E4–E5): **+2 days**. Engine-level window: optional +2.

## Risks

- **GPU contention**: the box currently runs another project's vLLM
  server at 0.60 gpu-mem-util; E1–E3 fit in the remainder (0.6 B model),
  E5 (concurrency ceiling) needs the GPU solo for clean numbers.
- **Committed-prefix + markers interaction**: after a fire, the flushed
  prefix no longer contains the marker context; v3–v8 were trained on
  whole-utterance targets, so post-flush behavior ("fresh segment") is
  in-distribution, but rollback-across-marker edge cases need a unit test
  (never roll back past a flushed `<END_SPEECH>`).
- **Marker tokens under vLLM sampling**: greedy decode must be allowed to
  emit reserved-range tokens (no `bad_words`/special-token suppression in
  the serving config; verify detokenization keeps them visible in E1).
- **AuT windowing at buffer edges**: bounded re-feed changes the leading
  edge of audio each flush; if WER at segment starts degrades, keep a
  0.5–1 s audio overlap before the committed boundary (standard trick,
  costs nothing at these lengths).
