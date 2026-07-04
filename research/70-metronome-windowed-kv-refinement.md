# The efficiency refinement: in-engine windowed KV + pinned CTX sink (2026-07-04)

The one open item after E1–E5 (research/67) was the **in-engine efficiency
version** of the pinned-CTX capability — bounded resident KV + flat per-frame
latency for a minute-scale continuous session, with the `<CTX>` hotword
prefix kept attended without re-encoding it every frame. This is Metronome's
own mechanism, so the work lands in the **Metronome** repo, not this one. It
has two halves.

## Part A — dense-Qwen3 windowed KV (DONE, validated, shipped)

Metronome's sliding-window patch (`metronome_vllm_plugin` /
`metronome/patches/qwen3_swa.py`) covered only `qwen3_moe.Attention` (the
Qwen3-Omni-MoE text backbone). The Qwen3-ASR endpoint model's thinker is
**dense `Qwen3ForCausalLM`** (vLLM `qwen3.py`), which was unpatched → full
attention, unbounded resident KV. Extended the patch to also cover
`qwen3.Attention`, so `METRONOME_SWA_TOKENS=W` builds the thinker's decoder
attention with `per_layer_sliding_window=W` → vLLM's `SlidingWindowSpec`
bounds both per-frame attention compute and resident KV (frees blocks behind
the window). Audio tower / vision keep their own attention classes,
untouched.

Validated on the merged v9 checkpoint (`worker_integration/e6_dense_swa.py`):

- The dense patch fires inside EngineCore:
  `[metronome-swa] EngineCore: patched qwen3_moe+qwen3 Attention, per_layer_sliding_window=256`.
- Endpoint behavior preserved under the window: fires `<END_SPEECH>` on
  speech+silence, silent on pure silence — the causal endpoint behavior is
  intact with the window active.

(The short-context latency A/B — 5–60 s ≈ ≤750 audio positions — is
contention-noise: attention over ≤750 tokens is cheap on a 0.6 B model, so
the window's payoff is in *resident KV* at minute-scale continuous sessions,
the Metronome cliff regime, not in per-decode latency at these lengths. The
mechanism, not a micro-benchmark, is the deliverable, and it's confirmed
active.)

**Metronome commit** `cb7b4a9` "swa: extend sliding-window patch to dense
Qwen3 (Qwen3-ASR thinker)". This is the deployable efficiency refinement:
the endpoint model can now be served with bounded resident KV for the
continuous-transcription variant.

## Part B — pinned CTX-token sink (IMPLEMENTED + validated at the mechanism level)

The window alone evicts the `<CTX>` prefix once the session grows past W, so
biasing would decay on a no-flush continuous session. The sink keeps the
first S tokens (the CTX block) attended AND resident — the StreamingLLM
`[0,S) ∪ [t-W, t]` union (arXiv:2309.17453). Fully implemented as
`metronome/patches/metronome_sink_kernel.py` (a copy of vLLM 0.19's
`triton_unified_attention.py` with the sink edits) + the block-pin in
`metronome/patches/metronome_sink.py`.

**The kernel change is not two lines — the 2d unified kernel applies the
window in THREE places, and all three must learn the sink:**

1. **Score mask** — `seq_mask &= ((q_abs - k) < W) | (k < SINK_TOKENS)`.
2. **Tile loop** — extend the window's tile range down to tile 0 when a sink
   is active (`tile_start *= 0`, kept Triton-typed to avoid a py-int/tl-value
   miscompile) so the `[0,S)` tiles are actually visited; the per-key mask
   excludes the middle.
3. **V-side window filter** — the kernel *also* zeros `V` for keys older than
   the window, right before the `P@V` accumulation. It must keep the sink:
   `V = tl.where(in_window | (k < SINK_TOKENS), V, 0.0)`. **This was the real
   bug.** Patching only the score mask (1) left sink keys carrying softmax
   weight in the denominator `L` but a *zeroed* `V` → the output was
   under-normalized garbage that matched no mask at all. It was invisible to
   the CPU `self_test` (which checks the pure-torch reference, not the Triton
   kernel) and would have silently corrupted every transcription.

Requires the `TRITON_ATTN` backend (`FLASH_ATTN` is compiled and cannot
express the union — the constraint the Metronome paper flags). vLLM 0.19
dropped the `VLLM_ATTENTION_BACKEND` env var; the backend is now an engine
arg, `LLM(attention_backend="TRITON_ATTN")`. The dispatch forces the 2d path
when a sink is active (the 3d segmented path is left unpatched and never runs
with a sink). The block-pin subclasses `SlidingWindowManager` and frees only
the middle `[sink_blocks, window)`, faithful to the base otherwise.

**Validated at the mechanism level — stronger than a transcription diff, and
runnable on the contended box where full vLLM init OOMs behind the co-tenant:**

- **E9 `worker_integration/e9_kernel_gpu_test.py`** drives the *actual*
  `metronome_sink_kernel.unified_attention` entry point against a pure-torch
  union reference on GPU. **Exact match (<2e-3, fp16)** across the prefill AND
  decode paths, GQA 8:1 / 8:2 / 4:1, block sizes 16/32, T = 40…256, and
  window/sink combinations including the middle-freed and sink>window regimes.
  This is the test that caught bug (3). `e9b_reverse.py` reverse-engineers the
  effective mask (used during the debug).
- **E10 `worker_integration/e10_blockpin_test.py`** drives the shipped
  `remove_skipped_blocks` with a mocked KV state across a growing session:
  keeps the sink `[0, ceil(S/blk))`, frees the middle, keeps the window, and
  differs from stock vLLM *only* in the pinned sink region — across
  non-block-aligned S/window edge cases.

**Full-engine correctness gate (E7) — PASSED.** With the GPU free, the full
sink (sink-aware kernel + block-pin, `S=8192` covers-all, `W=512`, TRITON) in
the real vLLM engine on the merged Qwen3-ASR endpoint model produces
transcriptions **identical to the full-attention FLASH baseline** — including
the 44 s probe that window-only-512 garbles (the sink extends the attention
span past the window). The patched paged kernel + block eviction do not corrupt
the serving path. Two init-time deadlocks had to be fixed first, both from
eager CUDA init during EngineCore plugin-load (before the worker's device
init), which hung the engine in `gpu_input_batch`:
  1. `register()` called `self_test()`, whose `torch.manual_seed()` initializes
     the CUDA context. Removed from the serving path (E9/E10 validate the math
     out-of-band; `python -m metronome_sink` still runs it for a dev check).
  2. `_install_kernel` imported `metronome_sink_kernel` eagerly (Triton runtime
     + `torch.finfo(fp8_dtype())`). Now a lazy wrapper defers the import to the
     first attention forward.
These are why the sink-covers-all config initially failed while the FLASH
baseline and window-only (Part A) succeeded — the failure was in the sink
*install*, at a layer before the kernel/block-pin ever execute.

**Metronome commits** `4138fb7` (initial sink), `cb7b4a9` (dense SWA),
`5cb5d0f` (V-side filter fix — the union kernel becomes correct),
`d3992a6` (init-deadlock fix — no CUDA init during plugin-load).

## Why this ordering is the right call

The **capability** — context biasing surviving session length — is already
proven at the application layer (E4/research/69: bounded re-feed re-pins CTX
every frame, holding biasing flat at +30 pp across 0–12 s of intervening
audio). The Metronome paper itself notes application-level re-feed "achieves
the same memory horizon" as the in-engine sink. So Part B's in-engine version
is a pure **re-encode-cost optimization** for the continuous-captioning
variant, not a capability gap. Both halves are now implemented and validated
at the mechanism level (E9 kernel, E10 block-pin); nothing unvalidated ships
into the serving path.

## Files

- Metronome: `metronome/patches/qwen3_swa.py` (dense + MoE window),
  `metronome/patches/metronome_sink_kernel.py` (sink-aware Triton unified
  attention — union mask + tile loop + V-side filter), `metronome/patches/
  metronome_sink.py` (reference/self-test + kernel install + block-pin),
  `metronome_vllm_plugin` (EngineCore plugin, `METRONOME_SWA_TOKENS` /
  `METRONOME_SINK_TOKENS`; symlinks the two patch modules).
- This repo `worker_integration/`: `e6_dense_swa.py` (dense-SWA validation),
  `e9_kernel_gpu_test.py` (kernel union correctness, GPU), `e9b_reverse.py`
  (effective-mask reverse-engineer), `e10_blockpin_test.py` (block-pin unit
  test), `e7_sink_e2e.py` (full-engine correctness gate, GPU-gated),
  `e8_prep_data.py` + `e8_sink_biasing.py` (biasing-preservation, GPU-gated).
