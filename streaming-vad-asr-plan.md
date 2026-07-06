# Streaming Context-Aware VAD + ASR Model — Research Plan

**Author:** Bojie Li (Pine AI) and Noah Shi (University of Washington)
**Hardware:** 1× NVIDIA RTX Pro 6000 Blackwell (96GB GDDR7)
**Target:** unified streaming model that replaces the cascaded VAD + ASR pipeline, with low-latency endpoint detection and context-aware transcription.

---

## 1. Problem & Motivation

Production voice agents today use a brittle pipeline:

```
mic ──► VAD ──► ASR ──► (LLM) ──► TTS
        │       │
        │       └── no awareness of speaker, history, or domain
        └────────── context-free; misfires on hesitations, background noise
```

Two structural problems:

1. **VAD and endpoint detection are context-free.** Silero / WebRTC VAD trigger on acoustic energy. They cannot distinguish "user is thinking mid-sentence" from "user has finished." This produces premature cut-offs or long, awkward pauses.
2. **ASR has no semantic context.** Whisper transcribes audio in isolation. Proper nouns, domain jargon, prior turns, and user-specific entities (names, email addresses, product SKUs) are systematically mis-transcribed.

Both problems originate from the same architectural choice: the speech-perception layer is a stateless acoustic model, divorced from the conversation.

**Goal of this project:** train a single small streaming model that consumes audio at 200–300 ms granularity and emits:

- `<START_SPEECH>` when the user begins speaking,
- `<END_SPEECH>` when the utterance is semantically complete,
- the transcript itself, biased by an arbitrary text context prefix (user profile, recent transcripts, domain hotwords).

Output is text-only. No speech generation, no thinking tokens, no offline mode.

---

## 2. Goals and Non-Goals

### Goals

- **Streaming inference**: fire every 200–300 ms on a 96GB Blackwell, <30 ms added latency per tick.
- **Unified VAD + endpointing + ASR**: one model, one forward pass per tick.
- **Context biasing via text prefix**: arbitrary LLM-style prefix (user profile, recent dialogue, hotword list) influences transcription without retraining.
- **Robust to disfluent speech**: hesitations, fillers, mid-sentence pauses do not trigger false `<END_SPEECH>`.
- **Open release**: weights + training code + eval harness, comparable in spirit to Kyutai STT but with the context-prefix capability.

### Non-goals

- Not full-duplex (no speech output).
- Not multimodal beyond audio + text (no vision).
- Not multilingual at first — English-only v1; multilingual is a v2 extension.
- Not offline / chunked transcription — streaming only.
- Not chasing absolute SOTA WER on clean benchmarks; the value is *contextual* WER on proper nouns and *latency* of endpointing.

---

## 3. Core Design

### 3.1 System inputs and outputs

**Input per session:**
- Text context prefix (processed once, KV-cached):
  - User profile / known entities
  - Recent transcript (last N turns)
  - Optional domain hotword list
- Streaming audio (16 kHz mono PCM)

**Output per 200 ms tick:**
- Zero or more tokens from the streaming vocabulary:
  - `<NO_SPEECH>` (often suppressed — usually nothing is emitted)
  - `<START_SPEECH>`
  - text tokens (transcript)
  - `<END_SPEECH>` (always followed by EOS for the utterance)

### 3.2 Architecture

Based on Kyutai's **delayed streams modeling**: a causal transformer over two time-aligned token streams.

```
Text stream  : [<CTX>...prefix...</CTX>] [<NO_SPEECH>] [<NO_SPEECH>] ... [<START_SPEECH>] [hello] [world] [<END_SPEECH>] ...
Audio stream :                            [a0 a1 ... a31] [a32 ... a63] ...
                                           └── 1 Mimi frame (80 ms, 32 tokens) ──┘
```

- **Audio tokenizer:** Mimi codec (Kyutai, frozen). 12.5 Hz frame rate, 32 codebook entries per frame. 200 ms = 2.5 frames; we operate on 240 ms (3-frame) windows for clean alignment.
- **Backbone:** initialized from Qwen3-0.6B (primary) or Qwen3-1.7B (stretch). Embedding layer extended to accommodate Mimi audio token vocabulary + streaming control tokens.
- **Heads:** single LM head over the unified vocabulary `{text tokens} ∪ {audio tokens} ∪ {<NO_SPEECH>, <START_SPEECH>, <END_SPEECH>, <EOS>}`. Audio tokens are loss-masked at output (we only predict text + control).
- **Delay:** text stream lags audio by ~0.5–1.0 s (configurable hyperparameter) to allow lookahead for endpoint accuracy.

### 3.3 Streaming protocol

At each 240 ms tick:

1. Encode the new 240 ms audio chunk → 3 Mimi frames → 96 audio tokens.
2. Append audio tokens to the KV cache.
3. Decode text-stream positions corresponding to the new audio window (delayed by configured lookahead).
4. Emit any non-`<NO_SPEECH>` tokens to downstream.

The context prefix is encoded once at session start and lives in the KV cache for the entire session.

### 3.4 Control token vocabulary

| Token | Meaning |
|---|---|
| `<CTX>` ... `</CTX>` | Wrap the context prefix |
| `<NO_SPEECH>` | Emitted every text-stream position where no transcript content and no transition |
| `<START_SPEECH>` | Speech onset detected. Emitted once per utterance. |
| `<END_SPEECH>` | Utterance semantically complete. Followed by `<EOS>`. |
| `<EOS>` | Marks utterance/turn boundary. Caller flushes transcript on this signal. |

`<NO_SPEECH>` dominates the output stream (>90% of ticks emit it). At inference, it can be silently suppressed.

### 3.5 Context prefix design

Three classes of context, concatenated in this order:

```
<CTX>
[USER PROFILE]
Name: Bo Li
Role: ML researcher
Org: Anthropic
[DOMAIN VOCAB]
TTS, Mimi, Qwen3, Whisper, MiniCPM-o, RTX Pro 6000, Blackwell
[RECENT TRANSCRIPT]
Bo: I've been training a streaming ASR model.
Assistant: How is endpoint detection going?
</CTX>
```

- All sections optional.
- Combined prefix capped at 2048 tokens to bound KV memory.
- Backbone is causal — no architectural change needed for prefix.

---

## 4. Datasets

### 4.1 ASR foundation (Phase 1)

| Dataset | Hours | Domain | License | Status |
|---|---|---|---|---|
| LibriSpeech | 1,000 | read speech, audiobooks | CC BY 4.0 | open |
| MLS (English subset) | 44,500 | audiobooks | CC BY 4.0 | open |
| GigaSpeech (M+L tags) | 10,000 | podcasts, YouTube, audiobooks | varied | open |
| People's Speech | 30,000 | varied web | CC-BY-SA | open |
| Common Voice 17 (English) | ~3,000 | crowdsourced read | CC0 | open |
| **Total** | **~88,500** | | | |

Target curated subset: **~15–20k hours** after filtering for quality (no music, no excessive overlap, clean alignments). This is the bulk of Phase 1.

### 4.2 Conversational / endpointing (Phases 3, 5)

| Dataset | Hours | Notes |
|---|---|---|
| AMI Meeting Corpus | 100 | natural meetings, real overlap, free |
| ICSI Meeting Corpus | 70 | similar |
| CHiME-6 | 50 | dinner party, real overlap |
| Spotify Podcast Dataset (subset) | 5,000+ | conversational podcasts |
| TED-LIUM v3 | 450 | monologue, proper-noun dense |
| Earnings-22 / Earnings-21 | ~125 | named-entity dense, domain jargon |
| Fisher Corpus (LDC, **paid**) | 2,000 | gold standard for natural turn-taking |

**Decision point:** purchase Fisher (~$2k) or synthesize equivalent overlap data. Recommendation: defer; synthesize first, evaluate, buy if needed.

### 4.3 Context biasing data (synthetic, Phase 4)

This is built, not downloaded.

**Construction pipeline:**

1. Take any ASR utterance with transcript.
2. Run NER (spaCy or a small LLM) over the transcript to extract: person names, organizations, locations, products, technical terms, numbers.
3. Generate a synthetic `<CTX>` prefix that:
   - **80% of the time** contains the relevant entities ("relevant prefix")
   - **20% of the time** contains entities from a *different* random utterance ("distractor prefix" — critical for preventing prefix hallucination)
4. For each utterance, also generate variants with no prefix, prefix only, recent-transcript prefix, hotword-list prefix.

Target: ~50–100k synthetic context-utterance pairs covering all of Phase 1's curated audio.

### 4.4 Endpoint detection data

Train signal for `<START_SPEECH>` / `<END_SPEECH>`:

- **Real conversational** (AMI/ICSI/CHiME-6): force-align transcripts, mark utterance boundaries.
- **Synthesized turns:** take single-utterance ASR data, concatenate with realistic inter-turn gaps (200–2000 ms uniform), inject filler noises (breathing, throat clear) at boundaries to teach robustness.
- **Disfluency robustness:** TalkBank's CallHome, Switchboard subsets if available; otherwise synthesize hesitations using TTS.

Endpoint labels are derived from forced alignment + 200ms boundary smoothing.

### 4.5 Evaluation sets

Strict held-out, never seen during training:

| Eval | Purpose |
|---|---|
| LibriSpeech test-clean / test-other | baseline WER |
| TED-LIUM v3 test | proper-noun WER |
| Earnings-22 test | domain-jargon WER, context-biasing measurement |
| AMI eval | natural turn endpoint latency |
| Synthetic disfluency suite (custom) | false-endpoint rate |
| Custom hotword stress test (custom) | context biasing recall + hallucination rate |

---

## 5. Training Phases

Overall framing: this is **fine-tuning a pretrained Qwen3 backbone for streaming audio understanding**, not LLM pretraining. The phases progressively specialize the model.

### Phase 0 — Infrastructure (Week 1)

- Set up PyTorch + Transformer Engine + FlashAttention-3 on Blackwell.
- Bring up Mimi codec inference; pre-tokenize ASR datasets to disk (~2 TB after tokenization).
- Implement delayed-streams data loader and forward pass; verify against Kyutai STT inference.
- Build streaming eval harness: WER, endpoint latency P50/P95, false-endpoint rate, hotword recall.
- **Sanity run on the real target architecture**: train Qwen3-0.6B (with extended audio-token embedding) on a small data slice — LibriSpeech-clean ~200 hours, 1 epoch, ~6 hours of compute. Same code path as the production run; only the data is reduced. This catches integration bugs without committing to a multi-day training run.

**Why no smaller "toy" model:** Qwen3's smallest dense checkpoint is 0.6B. A from-scratch 300M would be a different architecture with a different bug surface, and findings wouldn't transfer to the production run. On a Blackwell the throughput difference between 300M and 600M is too small to justify the divergent code path.

**Deliverable:** working pipeline; Qwen3-0.6B sanity model with WER ≤ 25% on LibriSpeech test-clean after the short run (only as a smoke test — not the real WER target).

### Phase 1 — ASR Foundation (Week 2)

**Goal:** the model can transcribe streaming audio with reasonable WER, no endpointing yet.

- Initialize: Qwen3-0.6B weights + extended embedding for audio tokens.
- Data: ~15k hours curated (LibriSpeech + MLS + GigaSpeech-M).
- Objective: cross-entropy on text-stream positions only.
- Hyperparameters:
  - LR: 1e-4 with cosine decay, 1500 warmup steps
  - Batch: 64 sequences × 30 s audio each
  - Sequence length: ~10k tokens (mostly audio)
  - Mixed precision: bf16, FP8 attention if stable
  - Gradient checkpointing on
- 3 epochs ≈ 45B tokens ≈ **~5 days of compute**.

**Eval gate:** WER ≤ 8% on LibriSpeech test-clean, ≤ 18% on test-other.

### Phase 2 — Streaming Adaptation (Week 3 first half)

**Goal:** model produces low-latency partial transcripts.

- Same data, but trained with **bounded lookahead**: text stream delayed by 1.0 s relative to audio (configurable). At training time, mask future audio beyond delay window.
- Add `<NO_SPEECH>` token to vocabulary; emit at every text-stream position with no transcript content.
- 1 epoch over Phase 1 data ≈ **~2 days of compute**.

**Eval gate:** WER degrades by <1% absolute vs Phase 1; partial transcripts within 1.0 s of word completion.

### Phase 3 — VAD + Endpoint Detection (Week 3 second half + Week 4 first half)

**Goal:** model emits `<START_SPEECH>` / `<END_SPEECH>` reliably, including on disfluent speech.

- Data: AMI + ICSI + CHiME-6 + synthesized turn sequences (mix of clean + filler + hesitation).
- Inject ~10% of Phase 1 data to prevent ASR regression.
- Add boundary labels via forced alignment.
- Objective: same CE; the model is now learning when to emit control tokens.
- Sampling: ensure 30%+ of training sequences contain a turn boundary.
- **~3 days of compute.**

**Eval gate:**
- Endpoint detection P50 latency ≤ 400 ms post end-of-speech (faster than Kyutai's 500 ms target).
- False endpoint rate ≤ 5% on the synthetic disfluency suite.
- WER on LibriSpeech does not regress beyond Phase 2 by >2% absolute.

### Phase 4 — Context Prefix Biasing (Week 4 second half + Week 5)

**Goal:** the model uses the text context prefix to improve transcription of in-context entities, without hallucinating from irrelevant prefixes.

- Data: all Phase 1+3 data, but each example wrapped with synthesized `<CTX>` prefix per Section 4.3.
- **Critical:** 20% of examples have a distractor prefix (random entities not in the utterance). The model must learn to ignore them.
- Optional: include "empty prefix" examples (no `<CTX>` block at all) to preserve unbiased ASR ability.
- Loss weighting: 2× on positions immediately following biased entities, to focus learning.
- 1–2 epochs ≈ **~3–4 days of compute**.

**Eval gate:**
- Hotword recall on Earnings-22 ≥ 80% (baseline without context: ~50–60%).
- Hallucination rate on distractor-prefix eval ≤ 3% (false insertion of prefix terms).
- General WER does not regress >1% absolute.

### Phase 5 — Polish + Preference Optimization (Week 6)

**Goal:** clean up failure modes.

- Targeted SFT on a small (~1k hours), highly-curated dataset matching production-style usage.
- **DPO** for endpoint timing: build preference pairs where the "winner" emits `<END_SPEECH>` at a more natural turn boundary than the "loser." This is the only RL/preference step.
- ~1–2 days of compute.

**Final eval:** full benchmark suite, latency profiling under load, model card.

---

## 6. Evaluation

### 6.1 Metrics

| Metric | What it measures | Target |
|---|---|---|
| WER (clean) | baseline transcription | ≤ 8% on LibriSpeech test-clean |
| WER (other) | noisy transcription | ≤ 18% on test-other |
| WER (proper noun) | named-entity transcription | report alongside |
| Hotword recall | context biasing utility | ≥ 80% on Earnings-22 with relevant prefix |
| Hallucination rate | distractor robustness | ≤ 3% |
| Endpoint latency P50 / P95 | how fast `<END_SPEECH>` fires after silence | P50 ≤ 400 ms, P95 ≤ 800 ms |
| False endpoint rate | misfires mid-utterance | ≤ 5% on disfluency suite |
| Per-tick wall-clock (Blackwell) | streaming feasibility | ≤ 25 ms |
| Memory per session | concurrent capacity | ≤ 500 MB |

### 6.2 Eval harness design

Build the harness **in Phase 0**, not later. Iteration speed is gated by eval throughput.

- Streaming simulator: replays audio at 1× real-time, calls the model every 240 ms, records emit timestamps.
- Reference timestamps from forced alignment.
- Standardize all numbers in a single JSON report per run.

---

## 7. Compute Budget

Single Blackwell, 96 GB.

| Phase | Compute | Wall-clock (with overhead) |
|---|---|---|
| 0 — Infra | <1 day GPU | 1 week (setup-heavy) |
| 1 — ASR foundation | ~5 days | 1 week |
| 2 — Streaming adaptation | ~2 days | 3–4 days |
| 3 — Endpoint detection | ~3 days | 4–5 days |
| 4 — Context biasing | ~3–4 days | 1 week |
| 5 — Polish + DPO | ~1–2 days | 3–4 days |
| **Total** | **~15 days GPU** | **~4–6 weeks calendar** |

If 1.7B variant pursued: add ~2–3 weeks at the end with the validated recipe.

---

## 8. Risks & Open Questions

### High-risk

1. **Prefix hallucination.** Model emits entities from the prefix that weren't spoken. Mitigation: distractor training, hallucination metric in eval, ablate distractor ratio.
2. **Endpoint detection on disfluent speech.** Long hesitations / mid-sentence pauses get classified as end-of-turn. Mitigation: hard examples in Phase 3, possible DPO refinement in Phase 5.
3. **Mimi vs custom audio encoder tradeoff.** Mimi is designed for speech generation; its tokens may not be ASR-optimal. Mitigation: in Phase 0 sanity, compare Mimi vs frozen Whisper-small encoder; pick the better one.

### Medium-risk

4. **Context length explosion.** Long conversations grow the prefix. Mitigation: cap prefix at 2048 tokens; LRU-evict older turns.
5. **Qwen3 init may not help.** Audio tokens are far from text; the LLM-init advantage could be small. Fallback: train from scratch — adds ~1 week.
6. **Data licensing.** Some MLS / GigaSpeech subsets have research-only terms. Audit before any public release.

### Open questions to resolve in Phase 0

- Audio token rate: Mimi 12.5 Hz, or a custom 25 Hz codec for lower latency?
- Should the model emit partial transcripts (`<PARTIAL>` token) or only finalize on `<END_SPEECH>`?
- Streaming with FP8 on Blackwell: numerical stability OK or fall back to bf16?

---

## 9. Milestones

| Milestone | Date (from start) | Deliverable |
|---|---|---|
| M0 — Pipeline working | end of Week 1 | Qwen3-0.6B small-data sanity run, eval harness |
| M1 — ASR foundation | end of Week 2 | 600M base, ≤8% WER LibriSpeech test-clean |
| M2 — Streaming | mid Week 3 | partial transcripts within 1.0 s |
| M3 — Endpointing | end of Week 4 | `<END_SPEECH>` P50 ≤ 400 ms, false rate ≤ 5% |
| M4 — Context biasing | end of Week 5 | hotword recall ≥ 80%, hallucination ≤ 3% |
| M5 — v1.0 release | end of Week 6 | weights, code, eval report, model card |
| (stretch) M6 — 1.7B variant | +3 weeks | scaled-up model |
| (stretch) M7 — multilingual | +6 weeks | French/Chinese variants |

---

## 10. Stretch Goals (Post-v1.0)

- **1.7B model** with the validated recipe.
- **Multilingual** (French, Chinese) — re-run Phases 1+4 with MLS-fr and AISHELL-3/WenetSpeech.
- **Retrieval-augmented context** (à la BR-ASR): instead of a static prefix, retrieve top-k relevant hotwords from a large vocabulary per session.
- **Joint training with a downstream LLM** for end-to-end optimization of conversational accuracy.

---

## Appendix A — Key Reference Work

- Kyutai STT (Kyutai Labs, 2025) — delayed streams modeling, semantic VAD. Closest open analog.
- Mini-Omni / Mini-Omni2 (Tsinghua, 2024) — small streaming audio LLM recipe, Qwen2-0.5B base.
- Moshi (Kyutai, 2024) — full-duplex speech-text foundation model; source of Mimi codec and delayed streams.
- Uni-ASR (arXiv 2603.11123) — unified LLM-based streaming + non-streaming ASR.
- BR-ASR (arXiv 2505.19179) — bias retrieval framework for contextual biasing in speech LLMs.
- "Contextual Biasing for LLM-Based ASR with Hotword Retrieval and RL" (arXiv 2512.21828) — GRPO-finetuned hotword retrieval.
- FastTurn-Unified (arXiv 2604.01897) — fused semantic + acoustic cues for turn detection.
- LiveKit Smart Turn v2 (2025) — open semantic endpointing model.

## Appendix B — Key Open Components

| Component | Source | Role |
|---|---|---|
| Mimi codec | kyutai-labs/moshi (HF) | audio tokenization |
| Qwen3-0.6B / 1.7B | Qwen team (HF) | LLM backbone init |
| Whisper-small (fallback encoder) | OpenAI | alternative audio frontend |
| LibriSpeech, MLS, GigaSpeech, People's Speech | various | ASR pretraining |
| AMI, ICSI, CHiME-6 | various | conversational / endpointing |
| Emilia / Emilia-Large | Amphion | optional ASR data augmentation |
| spaCy or small LLM (Qwen3-0.6B-instruct) | various | NER for synthetic context generation |
