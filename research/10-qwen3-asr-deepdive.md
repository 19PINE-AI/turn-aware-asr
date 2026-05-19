# Qwen3-ASR — technical deep-dive

**Sources:** arXiv:2601.21337v2 (Shi et al., Qwen Team, Feb 2026) and its HTML render; HF cards `Qwen/Qwen3-ASR-1.7B`, `Qwen3-ASR-0.6B`, `Qwen3-ForcedAligner-0.6B`; `QwenLM/Qwen3-ASR` README; `antirez/qwen-asr/MODEL.md` (reverse-engineered weight inspection, used only where the report is silent). Today: 2026-05-19. Where the paper does not state a value, this report says **"not reported"**.

---

## 1. Audio frontend

| Item | Value | Source |
|---|---|---|
| Input feature | 128-dim FBank (mel filterbank) | report §architecture; "8 times downsampling to Fbank feature with 128 dimensions" |
| Input sample rate | 16 kHz mono | HF card |
| Pre-encoder frame rate | 100 Hz (standard 10 ms FBank hop) | not stated, conventional |
| Downsampling factor | 8× total (three Conv2D layers, each 2× in both freq and time per the reverse-engineered MODEL.md) | report (8×); antirez (×3 Conv2D) |
| Post-encoder token rate | **12.5 Hz** | report verbatim |
| Encoder positional encoding | **Sinusoidal**, applied per chunk from position 0 (encoder is not RoPE) | antirez MODEL.md |
| Encoder attention | Non-causal windowed attention, **window ≈ 104 tokens** in the encoder (≈ 8 s of audio at 12.5 Hz) | antirez MODEL.md |
| Encoder freezing | Not reported. Report says model is "posttrained from the Qwen3-Omni foundation model" — implies encoder is jointly tuned, not strictly frozen. | report §2 |

Frontend lineage is conventional FBank + Conv subsampler + Transformer encoder (Whisper / Paraformer / FunASR family) — **not** discrete codec tokens. Same 12.5 Hz rate as Bo's Mimi plan, but continuous embeddings, not discrete codes.

---

## 2. Audio encoder ("AuT")

| Variant | Params | d_model | Layers | Heads | FFN | Output proj |
|---|---|---|---|---|---|---|
| 1.7B model | **300 M** | 1024 | 24 | 16 | 4096 | 2048 |
| 0.6B model | **180 M** | 896 | 18 | 14 | 3584 | 1024 |

AED-style Transformer encoder (not Conformer; no per-block conv module). Conv2D stem: 3 layers each 2× downsample on both axes; 128 mel bins → 16, then linear-projected to `d_model`. Pretrained on **~40 M hours of pseudo-labeled ASR data** (majority zh+en) — paper's headline number. Param counts (300 M / 180 M) verbatim from paper; layers/heads/FFN are derived from weight inspection.

---

## 3. Projector / LLM bridge

2-layer MLP: `d_model → d_model (GELU) → output_dim`, bias on both (antirez); paper says only "projector". Output dim = LLM hidden (2048 / 1024). **No Q-Former, no Perceiver.** Continuous frame embeddings replace `<|audio_pad|>` placeholder embeddings in the LLM stream. Vocab is **not** extended for audio; only special tokens added (`<|audio_start|>`=151669, `<|audio_end|>`=151670, `<|audio_pad|>`=151676, `<asr_text>`=151704); tokenizer stays at 151,936. This is the **opposite** of Bo's discrete-Mimi plan.

---

## 4. LLM backbone

| | 0.6B | 1.7B |
|---|---|---|
| Total params | 0.6 B | 1.7 B |
| LLM hidden | 1024 | 2048 |
| LLM layers | 28 | 28 |
| Heads / KV heads | 16 / 8 (GQA 2:1) | 16 / 8 (GQA 2:1) |
| Head dim | 128 | 128 |
| FFN intermediate | 3072 | 6144 |
| Tied embeddings | Yes (`embed_tokens == lm_head`) | Yes |
| Position encoding | **MRoPE** (`mrope_section=[24,20,20]`, interleaved) — reduces to standard RoPE for audio-only; theta=1e6 | same |
| **Q/K RMSNorm** (per-head, after projection, before RoPE) | Yes | Yes |

- Only **0.6B and 1.7B** are released. **No 4B, no 8B** Qwen3-ASR variant exists as of 2026-05-19.
- Backbone starts from Qwen3-Omni 0.6B / 1.7B, then is ASR-posttrained — not raw Qwen3-base.

---

## 5. Dynamic flash-attention window

Single most copy-worthy idea: window size **1 s to 8 s**, sliding, **same weights / same forward pass** for streaming and offline; mode is just an inference-time attention hyperparameter. Training uses a randomized window in 1–8 s; exact distribution **not reported**. KV strategy for long audio **not reported** explicitly — window bound implies effectively bounded-context; long audio handled by chunked re-encoding (§6). Conceptually similar to Uni-ASR (Bo already cites), now proven at 1.7B / 40 M hours.

---

## 6. Streaming protocol

- Streaming eval: **2 s chunk, 5-token fallback, last 4 chunks "unfixed"**.
- "Unfixed" ⇒ partial transcripts in last 4 chunks **are revised** when new audio arrives — partial revisions explicit.
- "5-token fallback" = drop last 5 tokens of prior transcription when prepending as text prefix to next chunk (prefix rollback to reduce boundary jitter).
- First 2 chunks decode with no text prefix; chunks 3+ prepend prior transcript minus 5 tokens.
- TTFT (avg, c=1): **92 ms (0.6B)**, **102 ms (1.7B)**. P95 spans 105–6227 ms across concurrency (Table 2).
- RTF: 0.00923 (0.6B, c=1), 0.01482 (1.7B, c=1). Throughput: 2000× realtime at c=128 on 0.6B.
- VAD/endpointing: **not built in**. Silent chunks yield empty `<asr_text>`; turn boundaries are not marked. Endpointing is left to the caller.

---

## 7. Context biasing / hotwords

Entire paper discussion is one sentence (§2.2): "The model learns to utilize the context tokens inside the system prompt as background knowledge, allowing users to obtain customized ASR results."

- **Mechanism:** plain text in the chat system prompt. No retriever, no BM25/dense, no phoneme lexicon, no logit bias, no cross-attention head.
- **Format:** standard `<|im_start|>system ... <|im_end|>` slot. Exact key/format **not reported**; HF README inference API exposes only `language` and `audio`, not a `context=` param. Marketing page allows "keywords, paragraphs, and mixed text of any length" — no structure.
- **Max hotword set size: not reported.** **Hotword recall benchmarks: none reported** in the paper (Tables 1–8 cover language ID, multilingual/dialect/accent WER, noise, streaming — no NER recall).

This is the **single biggest gap for Bo's purposes.** The plan's whole differentiator (context-prefix biasing with measured hallucination rate) is *not* benchmarked by Qwen3-ASR. The paper claims the capability exists but provides zero numbers.

---

## 8. Training

| Stage | Data | Method |
|---|---|---|
| AuT pretraining | **~40 M hours** pseudo-labeled ASR (majority zh+en) | not detailed; cross-entropy implied |
| Omni pretraining | 3 T tokens multimodal | (inherited from Qwen3-Omni) |
| ASR SFT | "substantially smaller, disjoint from pretraining" — exact hours **not reported** | CE |
| ASR RL | **~50 k utterances**: 35% zh+en, 35% multilingual, 30% "functional" | **GSPO** (Group Sequence Policy Optimization) |

- Loss: **CE only** for SFT. No CTC/MWER/MMI/min-latency. RL uses GSPO (GRPO lineage).
- Per-language hours **not reported** beyond "majority zh+en". Compute (GPU-days, cluster) **not reported**. Encoder freeze policy **not reported**.

40 M hours dwarfs Bo's curated ~15–20 k by ~3 orders of magnitude — the structural reason Qwen3-ASR reaches 1.63 WER on LS test-clean.

---

## 9. Benchmarks (verbatim from Tables in the report and HF cards)

**English (1.7B / 0.6B):**

| Set | 1.7B | 0.6B |
|---|---|---|
| LibriSpeech clean | **1.63** | 2.11 |
| LibriSpeech other | **3.38** | 4.55 |
| GigaSpeech | 8.45 | 8.88 |
| CommonVoice-en | 7.39 | 9.92 |
| Fleurs-en | 3.35 | 4.39 |
| TEDLIUM | — | 3.85 |
| VoxPopuli | — | 9.96 |
| MLS-en | — | 6.00 |
| AMI (leaderboard) | — | 11.66 |
| Earnings22 (leaderboard) | — | 11.06 |
| SPGISpeech (leaderboard) | — | 3.03 |

**Chinese:**

| Set | 1.7B | 0.6B |
|---|---|---|
| WenetSpeech net / meeting | 4.97 / 5.88 | 5.97 / 6.88 |
| AISHELL-2 test | 2.71 | 3.15 |
| Fleurs-zh | 2.41 | 2.88 |
| SpeechIO | — | 3.44 |
| CV-zh | — | 6.89 |

**Multilingual / dialect:** MLS-8: 13.19; CV-13: 12.75; Fleurs-12: 7.57; KeSpeech: 7.08; Dialog-Accented-en (16 accents): 16.62.

**Noise:** Internal "ExtremeNoise" — Qwen3-ASR-1.7B **16.17%** vs Whisper **63.17%** (Table 4).

**Streaming vs offline (Table 8, average WER):**

| Model | Offline | Streaming |
|---|---|---|
| 1.7B | 2.69 | 3.33 (+0.64) |
| 0.6B | 3.48 | 4.40 (+0.92) |

Streaming costs ~0.6–0.9 pp absolute on the 0.6B — a number Bo should treat as the **competitive floor** for his streaming-mode WER.

**Language ID accuracy:** 97.9% (1.7B), 96.8% (0.6B).

**Hotword recall:** **not reported.**

---

## 10. Open release

- Weights: **open**, Apache 2.0 (0.6B, 1.7B, ForcedAligner-0.6B).
- **Finetuning** code published (`QwenLM/Qwen3-ASR/finetuning`); full pretraining recipe + 40 M-hour data pipeline **not released**.
- AuT encoder weights live under `thinker.audio_tower.*` — extractable, but no standalone HF checkpoint.

## 11. Gaps requiring own ablations

Window-sampling distribution; AuT freeze policy; streaming WER vs delay curve; hotword recall + hallucination rates; max hotword count; KV-cache strategy for long audio; streaming-mode TTFT (paper's 92 ms is offline c=1); endpoint detection (Qwen3 doesn't do it); GPU-days.

---

## Implications for Bo's plan

Each bullet cites a Qwen3-ASR design choice (from sources above) and recommends *keep / change / test* in Bo's plan (`streaming-vad-asr-plan.md` and synthesis 00/06).

1. **Frontend: continuous FBank vs discrete Mimi.** Qwen3-ASR uses 128-dim FBank → 8× Conv2D → 12.5 Hz **continuous** embeddings, despite Alibaba having its own codec. **Test in Phase 0:** add an FBank+Conv2D arm to the existing Mimi-vs-Whisper-small bake-off (synthesis §5). Mimi must now beat three arms, not two.

2. **Encoder size band 180–300 M.** AuT is 30% / 18% of total params. **Keep** Bo's ~150–300 M encoder budget; don't over-invest above this band.

3. **Dynamic 1–8 s window unifies streaming/offline.** Same weights, same forward pass, mode-switched by inference-time window. **Change Phase 2:** replace the static 1.0 s delay with **randomized 1–8 s windows in training** so a single checkpoint covers both modes (this also implements the "learned minimum-latency policy" already flagged in synthesis 00).

4. **No VAD / endpointing in Qwen3-ASR.** Silent chunks → empty `<asr_text>`; no turn-boundary tokens. Bo's `<START_SPEECH>` / `<END_SPEECH>` / `<EAGER_END_SPEECH>` plus the decoupled 0-delay endpoint head (synthesis 06 Issue 2) is **the** actual differentiator vs Qwen3-ASR. **Keep, and lead the model card with it.**

5. **Hotword biasing is just system-prompt text — no retriever, no logit bias, no benchmarks.** Bo's `<CTX>` design is already congruent. **Keep**, and treat Qwen3-ASR's *missing* hotword-recall / distractor-hallucination numbers as Bo's **opportunity**: Earnings-22 recall and the distractor-hallucination rate from synthesis §6 should be the lead chart of Bo's release.

6. **RL via GSPO on 50 k utterances.** Bo's Phase 5 plans DPO on endpoint timing. **Test:** swap DPO for GSPO/GRPO with a forced-alignment reward (synthesis 06 Issue 5). Cheaper than DPO pair construction; matches Qwen3-ASR and Kyutai Hibiki-Zero.

7. **Streaming costs 0.6–0.9 pp WER vs offline at 0.6B** (Table 8). Bo's Phase 2 gate of "<1 pp" is **realistic** — set 0.9 pp soft / 0.6 pp stretch. Below 0.6 pp at 0.6B would be a paper-level result.

8. **Continuous projector replaces embedding extension.** Bo's §3.2 plans to extend the embedding matrix for discrete Mimi tokens. If the Phase 0 bake-off picks a continuous-encoder arm (bullet 1), swap to Qwen3-ASR-style `<|audio_pad|>` substitution — cheaper and no vocab growth. If discrete Mimi wins, the extension stays but **explicitly note the parallel-RVQ depth-decoder** (Moshi-style); otherwise reviewers will assume 32× the KV cost vs Qwen3-ASR (750 LLM tokens/min continuous vs 24 000 naïve-serial Mimi).

9. **Partial revisions are normal practice.** Qwen3-ASR keeps last 4 chunks unfixed + 5-token rollback. Bo's §3.3 is silent on revisability. **Add:** text revisable for last 4 ticks (~1 s); control tokens irrevocable on emission.

10. **MRoPE / Q-K RMSNorm / GQA 2:1 inherited from Qwen3-0.6B.** **Keep** — these are free when initializing from the same backbone.

11. **No 4B / 8B Qwen3-ASR exists.** Qwen stopped at 1.7B for ASR. **Keep** the 0.6B → 1.7B stretch ladder; don't climb further.

12. **Ship a forced-alignment side-model.** Qwen3 releases `Qwen3-ForcedAligner-0.6B` (42.9 ms AAS) alongside ASR. Bo's plan only uses alignment for *labels* in Phase 3. **Optional M5 addition:** publish a small alignment head — already implicit in the pipeline and gives users word timestamps for free.

Sources:
- [arXiv:2601.21337 — Qwen3-ASR Technical Report](https://arxiv.org/abs/2601.21337)
- [arXiv HTML v2](https://arxiv.org/html/2601.21337v2)
- [Qwen/Qwen3-ASR-1.7B (HF)](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Qwen/Qwen3-ASR-0.6B (HF)](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [QwenLM/Qwen3-ASR (GitHub)](https://github.com/QwenLM/Qwen3-ASR)
- [antirez/qwen-asr MODEL.md — reverse-engineered weight inspection](https://github.com/antirez/qwen-asr/blob/main/MODEL.md)
