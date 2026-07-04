# Metronome worker integration — Qwen3-ASR endpoint checkpoint (2026-07-04)

Serving design + test plan for running the merged endpoint Qwen3-ASR checkpoint
(scripts/merge_endpoint_lora.py) as a streaming session on the Metronome worker,
per the research/60 v1 protocol: **bounded audio re-feed + committed-prefix
decoding + endpoint-marker flush**, with the pinned-CTX attention sink as a v2
refinement. Code-only design pass — nothing here was run on the GPU (a GPU job
holds the card; commands below are for the main process to execute).

## Deliverables

| File | Role |
|---|---|
| `worker_integration/qwen3asr_stream.py` | Model-side streaming session: `Qwen3ASRStreamSession` (bounded re-feed, committed prefix, gate/confirm, marker flush) + engine factories (`build_engine`/`vllm_generate_fn`, TODO-GPU). |
| `worker_integration/stream_server_qwen3asr.patch` | `model_prompts()` Qwen3-ASR entry for `~/metronome/worker/stream_server.py`. |
| `worker_integration/e2_chunked_vs_segment.py` | E2 driver (chunked-placeholder vs single-segment WER). GPU-run. |
| `worker_integration/e3_replay_vllm.py` | E3 driver (research/58 replay eval on the vLLM path; per-chunk compute vs the 455 ms simulator). GPU-run. |

## Architecture: where the code runs

The gateway → worker gRPC contract (`Step`/`Health`, `worker/server.py`,
`worker/stream_server.py`) is unchanged. Two integration surfaces:

1. **`model_prompts()`** gains a `qwen3…asr` branch (the patch). This lets the
   *existing* resident/chunked `_run_session` path (`stream_server.py`) and
   `fd_step_stream` (`vllm_backend.py`) build valid Qwen3-ASR prompts — used
   directly as the **E2 chunked arm** (one audio placeholder per chunk).
2. **`Qwen3ASRStreamSession`** (new module) is the **recommended v1 path**: one
   audio placeholder over a *bounded* re-fed buffer, with all the endpoint
   control logic (flush, gate, confirm, max-buffer cap). It is engine-agnostic —
   the decode step is injected as `generate_fn(prompt, audio) -> raw_text` — so
   it unit-tests on CPU and binds to a resident vLLM engine inside the worker via
   `vllm_generate_fn(engine)`.

### Wiring the session into the worker (v1)

`Qwen3ASRStreamSession` replaces the per-frame body of a `StreamingEngine`
session (`stream_server.py::_run_session`). Sketch of the async session loop the
main process adds (analogous to the existing `turn_eos` / windowed branches):

```python
# inside StreamingEngine._run_session, for the ASR endpoint model:
from worker_integration.qwen3asr_stream import Qwen3ASRStreamSession, vllm_generate_fn
gen = vllm_generate_fn(self.engine)                 # engine built in _make_engine (GPU)
sess = Qwen3ASRStreamSession(generate_fn=gen, tokenizer=self.tokenizer,
                             context=st_context, window_s=16.0, max_buffer_s=30.0,
                             energy_gate=True, confirm_chunks=1, chunk_s=self.block_s)
while not st.stop:
    item = await st.queue.get()
    if item is None: break
    arr, sr = item
    r = sess.feed(arr)                              # one bounded decode (or gated skip)
    st.text = sess.transcript()                     # monotonic flushed transcript
    if r.fired:                                     # emit END signal to the gateway
        st.endpoint_ms = r.fire_time_s * 1000
```

The `generate_fn` seam is deliberate: the async worker should call
`self.engine.generate(...)` (AsyncLLM) and read `out.outputs[0].text` rather than
the offline `LLM.generate` used in `vllm_generate_fn`'s reference body — same
request dict, just the async iterator. That is the one GPU-side edit flagged
`TODO(GPU)` in the module.

## Key design decisions

### 1. Bounding the buffer (two mechanisms, both O(W_audio) per frame)

* **Utterance flush (primary).** On a *terminal* `<END_SPEECH>` (nothing decoded
  after it) the segment transcript is flushed and its audio is dropped
  (`_reset_segment(keep_overlap=False)`). In conversation the endpointer itself
  keeps the buffer to one utterance — utterances, not sessions, bound the state.
* **`window_s` cap (re-feed bound).** Each decode re-feeds at most the last
  `window_s` (16 s) of audio — the in-distribution single-segment shape, never
  N chunked placeholders. This is the hard per-frame compute bound.
* **`max_buffer_s` force-flush (continuous-speech safety).** For streaming
  captions where markers rarely fire (the research/61 v8 monologue tail, WER 61
  from a repetition loop), past 30 s the live segment is force-flushed as a
  *transcription reset* — **no END fire** — carrying an `overlap_s` (0.75 s) audio
  tail so the next segment's leading edge is not a hard AuT window cut
  (research/60 risk: "AuT windowing at buffer edges"). This is the
  Metronome bound-the-resident-state principle applied to the committed *text*
  stream; it is exactly the eval's `--max-segment-chunks` behavior.

### 2. Committed-prefix × marker flush

Committed-prefix decoding (qwen_asr `streaming_transcribe`): prompt = chat prefix
+ previously committed transcript minus the last `rollback` (K=5) tokens (utf-8
safe), so the model continues rather than re-decides. First `unfixed_chunks` (2)
of a segment decode from scratch (no prefix), matching qwen_asr's
`unfixed_chunk_num`.

The interaction with the marker flush is the delicate part (research/60 risk):
**rollback never crosses a flushed marker.** `self.raw` (the committed text) is
cleared to `""` on every flush, so `_committed_prefix()` only ever rolls back
*within the current live segment* — there is no committed text spanning a fired
`<END_SPEECH>`. Post-flush the next chunk starts a fresh segment (prefix `""` for
`unfixed_chunks`), which is in-distribution: v3–v8 were trained on whole-utterance
targets, and "fresh segment after a fire" is the whole-utterance start case.

The **confirm policy** (research/61 h=1) decouples the transcript flush from the
END signal: the terminal marker *always* flushes its segment (transcription stays
continuous and bounded), but the system-level END is deferred `confirm_chunks`
silent chunks; if speech resumes inside that window the END is discarded (it was a
phrase boundary, not a turn) while the flush stands. The pending-fire resolves
*before* the energy gate so the confirming silent chunks are not skipped.

### 3. CTX prefix → pinned attention sink

Context/hotwords live in the **system block** of the prompt = tokens `[0, S)` of
every request (research/16: −37 % rel. WER from context; research/60 fact 6). In
**v1** they are simply re-fed inside `HEAD` each chunk — cheap and always attended
because every bounded re-feed re-prefills them. In **v2** (in-engine dense-Qwen3
SWA + sink pin) those `[0, S)` tokens are exactly what the KV-manager pin + Triton
union mask keep resident while audio history slides out of the window.
`Qwen3ASRStreamSession.pinned_prefix_len()` returns S (token length of the system
block up to and incl. the first `<|im_end|>\n`) — the value the v2 windowed-KV
build pins. This is the novel serving-side claim (hotword recall vs session age,
window ±pin) that E4 will test.

### 4. `model_prompts()` entry (the patch)

Qwen3-ASR differs from the omni/minicpm templates: context in the SYSTEM block
(not a helpful-assistant persona), **no instruction text**, placeholder
`<|audio_start|><|audio_pad|><|audio_end|>`:

```python
if "qwen3" in m and "asr" in m:
    aph  = "<|audio_start|><|audio_pad|><|audio_end|>"
    head = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
    return head, aph, "", "<|im_end|>\n<|im_start|>assistant\n", "\n"  # HEAD,APH,INSTR,ASST,TRAIL
```

Verified byte-identical to `data/qwen3-asr-0.6b/chat_template.json` for the
single- and N-audio cases. Marker ids 151705/151706 must NOT be suppressed in
SamplingParams (greedy, no `bad_words`, `ignore_eos=False`) so they decode as
ordinary tokens.

---

## TEST PLAN

Prereq **E1** (merge, must run first): produces the standard HF checkpoint both
E2/E3 load.

```bash
# GPU. Merge the v5 (all-round) and v8 (precision) endpoint LoRAs.
cd ~/streaming-vad-asr
python -m scripts.merge_endpoint_lora \
    --checkpoint checkpoints/semantic_endpoint_v5_es/best.pt \
    --out-dir checkpoints/merged/qwen3-asr-0.6b-endpoint-v5
python -m scripts.merge_endpoint_lora \
    --checkpoint checkpoints/semantic_endpoint_v8_es/best.pt \
    --out-dir checkpoints/merged/qwen3-asr-0.6b-endpoint-v8
# Parity smoke: transcribe one marker-bearing clip via transformers from the merged
# dir and confirm <END_SPEECH> survives detokenization (research/60 E1).
```

### E2 — chunked-placeholder vs single-segment WER (decides the primitive)

Question: does feeding 0.5 s chunks as N separate audio placeholders
(`fd_step_stream` shape, flat cost) degrade WER vs one segment? Decision rule
(research/60): Δ < 0.5 pp → adopt `fd_step_stream` (resident, flat cost); else →
bounded re-feed (`Qwen3ASRStreamSession`).

```bash
# GPU, ~30 LibriSpeech utterances. Uses the merged v5 checkpoint (markers irrelevant
# to WER; language head only). gpu-mem 0.30 fits alongside the resident job's 0.60.
cd ~/streaming-vad-asr
PYTHONPATH=. python -m worker_integration.e2_chunked_vs_segment \
    --model-dir checkpoints/merged/qwen3-asr-0.6b-endpoint-v5 \
    --ls-root data/librispeech_raw/LibriSpeech/train-clean-100 \
    --n-utts 30 --chunk-s 0.5 --gpu-mem 0.30 \
    --out research/64-e2-chunked-vs-segment.json
```

Output JSON reports `single_segment_wer`, `chunked_wer`, `delta_pp`, and a
`decision` field. Read `delta_pp`: if the chunked path is within 0.5 pp the
resident flat-cost path is usable; a v9+ can later *train* with chunk-boundary
placeholders to close any residual gap.

### E3 — replay eval on the vLLM path (real per-tick latency)

Runs the research/58 streaming-replay eval through the resident vLLM engine and
reports fire recall / false-fire / WER **identically scored** to the research/61
re-baseline (same `run_stretch`/`aggregate`), plus per-chunk compute vs the
455 ms/chunk transformers simulator. Run the two research/61 operating points:

```bash
# GPU. v5 + gate + confirm1 (the v1 endpoint policy, research/61 recommendation).
cd ~/streaming-vad-asr
PYTHONPATH=. python -m worker_integration.e3_replay_vllm \
    --model-dir checkpoints/merged/qwen3-asr-0.6b-endpoint-v5 \
    --split data/semantic_endpoint_v3/meeting_split.json \
    --ami-dir data/ami/ihm --n-stretches 25 --chunk-s 0.5 \
    --energy-gate --confirm-silent-chunks 1 --gpu-mem 0.30 \
    --out research/64-e3-replay-v5-vllm.json

# v8 + gate + confirm1 + max-segment force-flush (precision point; needs the flush).
PYTHONPATH=. python -m worker_integration.e3_replay_vllm \
    --model-dir checkpoints/merged/qwen3-asr-0.6b-endpoint-v8 \
    --split data/semantic_endpoint_v3/meeting_split.json \
    --ami-dir data/ami/ihm --n-stretches 25 --chunk-s 0.5 \
    --energy-gate --confirm-silent-chunks 1 --max-segment-chunks 40 --gpu-mem 0.30 \
    --out research/64-e3-replay-v8-vllm.json

# Optional: finer chunk on the vLLM path (research/58 mentions 0.25 s), tightens P95.
#   ... --chunk-s 0.25 --out research/64-e3-replay-v5-vllm-025.json
```

Acceptance (research/58 targets, must hold on the vLLM path): boundary recall
≥ 0.90, false fires ≤ 1/min, P50 ≤ 0.5 s / P95 ≤ 1.2 s, streaming WER ≤ offline
+ 1 pp. `chunk_compute_ms_median` is the headline that **replaces the 455 ms/chunk
simulator number** in all v1–v8 latency claims; `chunk_compute_ms_vs_sim_455` is
the speedup ratio.

Cross-check E3-vLLM against the transformers re-baseline (identical scoring path)
to confirm the merge + vLLM path did not change model behavior:

```bash
python -m eval.streaming_replay_eval \
    --checkpoint checkpoints/semantic_endpoint_v5_es/best.pt \
    --energy-gate --confirm-silent-chunks 1 --n-stretches 25 \
    --out research/64-e3-replay-v5-transformers.json   # compare recall/false/WER to vLLM
```

---

## Open questions for the GPU-execution phase

1. **Marker visibility under vLLM detok.** The session scans the *text* form
   (`out.outputs[0].text`). Confirm the merged tokenizer emits a non-blank piece
   for 151705/151706 so `<END_SPEECH>` appears in `.text` (E1 detok smoke). If
   vLLM registers them as special with a blank piece, switch the scan to token-id
   detection (`END_ID in out.outputs[0].token_ids`) — a one-line change in
   `vllm_generate_fn`.
2. **AsyncLLM vs offline `LLM.generate`.** `vllm_generate_fn`'s body uses offline
   `LLM.generate`; the worker's `StreamingEngine` uses AsyncLLM. The async
   adapter (drive one `engine.generate` per `feed`) needs writing and a latency
   check — a fresh request per chunk vs a resumable request. Bounded re-feed is a
   fresh request per chunk by construction (the audio embeds change), so prefix
   caching only covers the text prompt; confirm the per-chunk `add_request`
   overhead is negligible at W_audio ≤ 16 s (research/60 estimate: ≤ 750 audio
   positions).
3. **E2 outcome routing.** If chunked WER ≈ single-segment, the resident
   `fd_step_stream` path (flat cost) supersedes bounded re-feed and the session
   logic (flush/gate/confirm) moves on top of the resident request. If not,
   bounded re-feed ships as-is. Either way the control logic in
   `Qwen3ASRStreamSession` is reused.
4. **`window_s` vs `max_buffer_s` sizing.** 16 s / 30 s are research/60 defaults;
   E3 with `--max-segment-chunks` sweeps validate the force-flush point against
   the v8 monologue tail. E4 (long-session hotword) tunes them under the pinned
   sink.
5. **v2 sink pin depends on a dense-`Qwen3ForCausalLM` SWA patch** (the existing
   `patches/qwen3_swa.py` targets the MoE class). `pinned_prefix_len()` supplies
   S; the Triton union-mask sibling for the dense class is the +2-day v2 item and
   needs quality revalidation on the 0.6 B backbone (research/60 fact 5).
6. **Energy-gate threshold on real mic input.** `gate_rms=1e-3` was set on AMI
   ihm; deployment mics may need recalibration (research/61 note: the gate can
   miss low-energy speech onset — v9's silence schemas are the defense-in-depth).
