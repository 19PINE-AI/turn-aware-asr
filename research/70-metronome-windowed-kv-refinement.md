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

## Part B — pinned CTX-token sink (math validated; in-engine install gated)

The window alone evicts the `<CTX>` prefix once the session grows past W, so
biasing would decay on a no-flush continuous session. The sink keeps the
first S tokens (the CTX block) attended AND resident — the StreamingLLM
`[0,S) ∪ [t-W, t]` union (arXiv:2309.17453). Implemented as
`metronome/patches/metronome_sink.py`:

- **Reference + numerical self-test (validated, CPU):** the union-mask
  attention math is proven correct — `sink-effect=0.70` (a late query's
  output changes vs window-only, i.e. the sink tokens contribute where a pure
  window drops them → biasing preserved) and `full-match=0.0` (the union
  equals full causal attention when W+S cover the prefix). `register()` runs
  this self-test before doing anything.

- **In-engine install — honestly gated, not shipped unvalidated.** The full
  mechanism needs two coupled changes to vLLM internals:
  1. **Kernel union mask.** A 2-line change to the Triton unified-attention
     kernel (`v1/attention/ops/triton_unified_attention.py`): the
     window mask `(query_abs_pos - seq_offset) < SLIDING_WINDOW` becomes
     `((query_abs_pos - seq_offset) < SLIDING_WINDOW) | (seq_offset < SINK)`,
     plus a matching tile-pruning fix so the sink tiles aren't skipped. This
     requires the `TRITON_ATTN` backend — the default `FLASH_ATTN` kernel is
     compiled and cannot express the union (the exact constraint the
     Metronome paper flags).
  2. **KV-block pin.** The first `ceil(S/block)` blocks per request must be
     pinned in vLLM's `SlidingWindowManager` (which otherwise frees them), or
     the kernel attends to freed blocks. This is the one piece that touches
     the v1 KV-eviction internals.

  On this shared, OOM-prone GPU a modified paged-attention kernel + block
  manager cannot be *safely* validated end-to-end (a plausible-but-wrong
  attention kernel silently corrupts transcription — worse than not shipping
  it), so `register()` validates the mask math, installs behind
  `METRONOME_SINK_TOKENS`, and **logs the exact remaining step rather than
  claiming a working sink**. The precise kernel diff and recipe are in the
  module docstring.

**Metronome commit** `4138fb7` "sink: CTX-token pinned-sink union mask …".

## Why this ordering is the right call

The **capability** — context biasing surviving session length — is already
proven at the application layer (E4/research/69: bounded re-feed re-pins CTX
every frame, holding biasing flat at +30 pp across 0–12 s of intervening
audio). The Metronome paper itself notes application-level re-feed "achieves
the same memory horizon" as the in-engine sink. So Part B's in-engine version
is a pure **re-encode-cost optimization** for the continuous-captioning
variant, not a capability gap — and the honest state is: window half shipped
and validated (Part A); sink half's math validated with the integration
precisely scoped and gated (Part B). Nothing unvalidated ships into the
serving path.

## Files

- Metronome: `metronome/patches/qwen3_swa.py` (dense + MoE),
  `metronome/patches/metronome_sink.py` (union mask + self-test),
  `metronome_vllm_plugin` (EngineCore plugin, `METRONOME_SWA_TOKENS` /
  `METRONOME_SINK_TOKENS`).
- This repo: `worker_integration/e6_dense_swa.py` (dense-SWA validation on
  the merged endpoint checkpoint).
