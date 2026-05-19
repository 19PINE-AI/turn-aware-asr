# Phase 0 Open Questions — Resolved

Date: 2026-05-19. Target: 1× RTX Pro 6000 Blackwell, Qwen3-0.6B + Mimi + delayed streams.

---

## Q1 — Mimi vs Whisper-small encoder vs custom 25 Hz codec

### Evidence

The literature is split, and the split correlates with whether the LLM backbone is reused.

- **arXiv 2409.00800** (LLaMA2 backbone, LibriSpeech): HuBERT-CTC discrete supervised tokens reach **1.69 / 3.03 WER (test-clean/other)**, beating the same backbone with continuous HuBERT at **6.26 / 7.09**. Advantage came from tokens "matched with LLaMA2's pretraining vocabulary." For unsupervised tokens, continuous wins decisively (7.17/12.21 vs 70.13/71.91).
- **arXiv 2411.08742** (Qwen1.5-0.5B — same family as Bo's plan): continuous features generally outperform discrete tokens on semantic tasks; the gap is attributed to "limited token granularity and inefficient information retention." Closest published analog to Bo's setup.
- **Kyutai DSM (arXiv 2509.08753, Table 1):** DSM-ASR on Mimi tokens hits **1.7 / 4.3 WER** on LibriSpeech test-clean/other with a 2.6B model and 2.5 s delay — competitive with Whisper-large-v3. Whisper-small itself sits at **5.0 / 12.2**.
- **Discrete Audio Tokens survey (arXiv 2506.10274):** discrete tokens can match fbanks on multilingual ASR with training time <35 % of dense-feature baselines and ~1.7 % absolute WER reduction — but with CTC heads, not LLM heads.
- No published evidence for higher-rate (25 Hz) custom codecs in an ASR-first LLM setup; SNAC and DAC are TTS-oriented. A custom codec is a 2–4 week detour with unclear payoff.

### Tradeoffs at Bo's scale

| Axis | Mimi (12.5 Hz × 32 codes) | Whisper-small encoder (50 Hz continuous) |
|---|---|---|
| LM-head reuse | Native (extend vocab) | Adapter MLP required |
| Existing recipe | Kyutai DSM, end-to-end | Mini-Omni adapter |
| Streaming alignment | 12.5 Hz frames, drop-in | Needs causal/chunked re-encoder |
| Path to generation | Free | Closed |

### Recommendation

**Commit to Mimi in Phase 0. Whisper-small is the gated fallback.** (1) The plan's delayed-streams architecture is *defined* by Mimi's 12.5 Hz rate; switching abandons the published Kyutai recipe. (2) Kyutai already showed Mimi hits 1.7 WER — the ceiling is fine. (3) The Phase 0 sanity run already runs a Mimi-vs-Whisper smoke test — let empirics decide. **Decision rule:** after the 6 h LibriSpeech-200h run, if Mimi WER is within 3 absolute points of Whisper-small, ship Mimi. Do **not** build a custom 25 Hz codec — no evidence it beats Mimi for ASR; off the critical path.

---

## Q2 — FP8 attention stability on Blackwell

### Evidence

- **NVIDIA, *Floating-Point 8: An Introduction*** explicitly partitions transformers into "regions safe for FP8" (linear GEMMs) and "regions unsafe for both FP8 and FP16" (softmax, layernorm). Attention crosses both.
- **NVIDIA, *Per-Tensor and Per-Block Scaling Strategies*** shows MXFP8 (Blackwell's hardware-native block-scaled FP8) tracking BF16 validation loss on Nemotron-2B and 8B for the full run, with no divergence — **but all examples are GEMM-only FP8, not full FP8 attention.**
- **FlashAttention-3 (arXiv 2407.08608):** FP8 FA-3 achieves **9.1e-3 RMSE vs FP16 reference**, ~2.6× lower error than naive per-tensor FP8 attention (2.4e-2) — but still ~50× the error of FP16 FA-3 (1.9e-4). Block quantization of Q/K/V and incoherent processing of outliers are required to make it usable.
- **arXiv 2505.20524** documents that unmitigated FP8 training causes instability from outlier values in residual streams and recommends keeping a **BF16 fallback path**. LMSYS *Unified FP8* (2025-11) confirms attention outliers remain the dominant instability source on Blackwell.
- No published FP8-attention divergence reports specific to <2B models on Blackwell; the failure-mode reports are from 7B+ runs at long token horizons.

### Recommendation

**Phase 0–1: BF16 attention, MXFP8 on linear GEMMs only.** This is the NVIDIA-recommended default and the only configuration validated end-to-end on Nemotron-2B/8B. Speedup vs full BF16 is ~30 % at no loss; the marginal 5–10 % from full FP8 attention isn't worth a multi-day debugging detour.

**Phase 2+: enable FP8 attention via `fp8_autocast` only after Phase 1 loss curves are clean** over the full ~45B-token horizon. Keep `TE_FORCE_BF16_ATTENTION=1` as a one-line fallback. Concretely in `transformer_engine.pytorch`: `recipe=DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=1024)` for GEMMs; leave the fused-attention backend on the F16 path until Phase 2.

---

## Q3 — `<PARTIAL>` token vs finalize-on-`<END_SPEECH>`

### Evidence — what 2025–2026 systems do

- **Kyutai STT (DSM, arXiv 2509.08753):** No `<PARTIAL>` token. Tokens emitted with a fixed delay (0.5 s for 1B-en_fr, 2.5 s for 2.6b-en) and **never revised**. A separate semantic-VAD head predicts end-of-turn probability. The "flush trick" drains the delay buffer once VAD fires.
- **Whisper Streaming (arXiv 2307.14743):** External **LocalAgreement-2** policy commits a prefix only when two consecutive Whisper passes agree. ~3.3 s average computationally-aware latency. Tokens revisable until agreement.
- **Deepgram Nova-3 (production):** Three-flag API — `is_final=false` interim (~150 ms, revisable), `is_final=true` finalized segments mid-utterance, `speech_final=true` end-of-utterance from VAD pause (recommended 300–500 ms). Interim revisions are part of the contract.
- **Mini-Omni / Mini-Omni2 (arXiv 2408.16725, 2410.11190):** Delayed-parallel text+audio generation; tokens emitted as decoded, no PARTIAL marker, no revision.

### Tradeoffs

| Approach | Latency | Revision cost | Complexity |
|---|---|---|---|
| Finalize-only (Kyutai, Mini-Omni) | High (≥ delay) | None | Lowest |
| Per-tick commit, no PARTIAL (Bo's plan) | Low | No fix path | Low |
| `<PARTIAL>` token, revisable | Lowest | Client re-render | Medium |
| LocalAgreement wrapper | High (~3 s) | Hidden | Medium |

### Recommendation

**Do not introduce a `<PARTIAL>` token. Commit tokens at every tick under the 1.0 s delay, and rely on delay + semantic endpointing for stability.** Matches Kyutai STT, the architectural reference for this plan.

Justification: (1) The 1.0 s text-stream delay already exceeds Kyutai's 0.5 s lookahead — a revision channel's marginal accuracy win is small, but it adds eval surface (revision rate, flicker, downstream-LLM re-processing). (2) Deepgram's interim/final split is a SaaS-API ergonomic choice for clients that can't tolerate buffering, not a model-quality decision. Bo's downstream is an LLM, which doesn't benefit from revisable text. (3) `<END_SPEECH>` already provides the only "act now" signal the downstream needs.

**Optional v1.1:** if v1.0 downstream eval shows the LLM would benefit from earlier signal, add a single `<PARTIAL_OK>` flag emitted alongside tokens with confidence > 0.95 (the WhisperLiveKit early-commit threshold). One-day post-v1.0 change; no architecture lock-in now.

---

## Phase 0 decisions

1. **Mimi codec, frozen, 12.5 Hz × 32 codebooks** — as planned. Whisper-small is the gated fallback (>3 WER gap).
2. **BF16 attention + MXFP8 GEMMs.** Enable FP8 attention only as a Phase 2 perf optimization.
3. **No `<PARTIAL>` token.** `<END_SPEECH>` is the sole finalization signal.

---

## Sources

- [Comparing Discrete and Continuous Space LLMs for ASR (arXiv 2409.00800)](https://arxiv.org/html/2409.00800v1)
- [A Comparative Study of Discrete Speech Tokens for Speech LLMs (arXiv 2411.08742)](https://arxiv.org/pdf/2411.08742)
- [Discrete Audio Tokens: More Than a Survey (arXiv 2506.10274)](https://arxiv.org/html/2506.10274v3)
- [Streaming Sequence-to-Sequence with Delayed Streams Modeling (arXiv 2509.08753)](https://arxiv.org/html/2509.08753v1)
- [Kyutai STT project page](https://kyutai.org/stt)
- [kyutai/stt-2.6b-en model card](https://huggingface.co/kyutai/stt-2.6b-en)
- [FlashAttention-3 (arXiv 2407.08608)](https://tridao.me/publications/flash3/)
- [NVIDIA, Floating-Point 8: An Introduction](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
- [NVIDIA, Per-Tensor and Per-Block Scaling Strategies for FP8 Training](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/)
- [Towards Fully FP8 GEMM LLM Training at Scale (arXiv 2505.20524)](https://arxiv.org/pdf/2505.20524)
- [LMSYS, Unified FP8 (2025-11)](https://www.lmsys.org/blog/2025-11-25-fp8-rl/)
- [Turning Whisper into Real-Time Transcription (arXiv 2307.14743)](https://arxiv.org/html/2307.14743v2)
- [WhisperLiveKit LocalAgreement backend](https://deepwiki.com/QuentinFuxa/WhisperLiveKit/3.2-localagreement-backend)
- [Deepgram, Configure Endpointing and Interim Results](https://developers.deepgram.com/docs/understand-endpointing-interim-results)
- [Mini-Omni (arXiv 2408.16725)](https://arxiv.org/html/2408.16725v1)
