# Plan Delta — what to change in `streaming-vad-asr-plan.md`

Synthesis of the eight research artifacts in this directory. Read this first; the supporting files (`01-`–`10-`) have the evidence and math.

Today: 2026-05-19. Bo confirmed sufficient compute budget, so recommendations are made on accuracy/architecture grounds rather than compute-saving.

**Update 2026-05-19 (after Qwen3-ASR deep-dive):** the synthesis below has been revised. The single most important change since the original synthesis: **Qwen3-ASR's AuT encoder is Apache 2.0, extractable from open weights, and pretrained on 40 M hours of ASR data** — three orders of magnitude more than Bo's curated corpus could ever cover. This makes it almost certainly the right audio frontend, and reshapes the Phase 0 bake-off, the WER targets, and the projector design. See [file 10](./10-qwen3-asr-deepdive.md).

---

## TL;DR — what must change before Phase 1 starts

1. **Build on Qwen3-ASR's AuT encoder, not Mimi or Whisper-small.** AuT (180 M @ 0.6B) was pretrained on ~40 M h pseudo-labeled ASR data and is extractable from the open `Qwen/Qwen3-ASR-0.6B` safetensors under Apache 2.0. Qwen3-ASR-0.6B reaches **2.11 / 4.55 WER** on LibriSpeech clean/other; Mimi-DSM and Whisper-small are far behind at Bo's scale. **Phase 0 bake-off should be: AuT (frozen) vs AuT (trainable) vs Mimi (frozen).** Whisper-small drops to fourth-choice fallback. See [file 10](./10-qwen3-asr-deepdive.md) §1–§4, Implications 1.
2. **Drop the embedding extension; use `<|audio_pad|>` substitution + 2-MLP projector.** If a continuous encoder (AuT or Whisper) wins the bake-off, vocab stays at 151,936; audio embeddings replace placeholder tokens via a GELU-activated 2-layer MLP. Cheaper, no vocab growth, no random-init shock. See [file 10](./10-qwen3-asr-deepdive.md) §3, Implication 8.
3. **Replace static delay with randomized 1–8 s training windows.** Qwen3-ASR proves a single checkpoint can serve both streaming (1 s window) and offline (8 s window) by training with a randomized window in that range. Replaces Bo's Phase 2 "static 1.0 s text delay" — same checkpoint, two modes, no extra training cost. See [file 10](./10-qwen3-asr-deepdive.md) §5, Implication 3.
4. **Decouple the endpoint head from the transcription window.** Acoustic-only endpoint head at 0-delay (~5 M params), independent of the transcription window. Without this, §6.1's 400 ms P50 target is unreachable, and Meta's hierarchical EOT (36 ms median) and Deepgram Flux's `EagerEndOfTurn` are the public bar. **This is Bo's actual differentiator vs Qwen3-ASR**, which has no endpoint detection. See [file 06](./06-architecture-review.md) Issue 2, [file 10](./10-qwen3-asr-deepdive.md) §6, Implication 4.
5. **Tag prefix subsections separately** (`<PROFILE>`, `<HOTWORDS>`, `<HISTORY>`) with differentiated distractor rates in Phase 4 (5%, 20%, 0%). **Hotword recall + distractor-hallucination is the lead chart** — Qwen3-ASR claims context biasing but reports zero numbers. See [file 06](./06-architecture-review.md) Issue 4, [file 10](./10-qwen3-asr-deepdive.md) §7, Implication 5.
6. **Update WER targets ~5×.** Phase 0 sanity ≤25 % → ~3 %; Phase 1 ≤8 % clean / ≤18 % other → **≤2.5 % / ≤6 %** if AuT init wins, or ≤3.5 % / ≤8 % conservative. The plan's current 8 % target is the 2022 Whisper-base baseline and would be a failure to launch. See [file 10](./10-qwen3-asr-deepdive.md) §9.
7. **Drop GigaSpeech and Spotify Podcast from training.** GigaSpeech's audio Terms of Access are non-commercial (the plan's "varied" is wrong); Spotify access ended Dec 2023. Replace with Emilia-YODAS + expanded MLS-en + CC-BY-only People's Speech. See [file 04](./04-dataset-licenses.md).
8. **Re-budget Phase 1 to ~13.5 GPU-days FP8 (not 5).** Plan's token math undercounts: 65 B tokens not 45 B. With AuT init (no codec tokens), audio is **150 LLM-positions/sec** continuous, not 400 discrete tokens/sec — cutting Phase 1 compute by ~2× and bringing it back near the original 5–7 GPU-day budget. See [file 05](./05-compute-sanity.md), [file 10](./10-qwen3-asr-deepdive.md) Implication 8.

Items 1–6 are architectural and addressable in Phase 0. Items 7–8 are operational. All independent of each other.

---

## §-by-§ changes

### §1 — Problem & Motivation
No changes.

### §2 — Goals
Add: "Beat published EOT baselines (Meta hierarchical 36 ms, FastTurn) on conversational endpoint latency, with no transcript regression."

### §3.1 — Inputs/outputs
Add `<EAGER_END_SPEECH>` to the output vocab (fires on high-confidence semantic completion, before silence, so downstream LLM can prefill). Models Deepgram Flux's design. See [file 02](./02-recent-sota.md) recommendation 2.

### §3.2 — Architecture
- **Audio frontend:** initialize from `Qwen/Qwen3-ASR-0.6B` AuT encoder (`thinker.audio_tower.*` weights, 180 M params, 12.5 Hz continuous output). Extract under Apache 2.0. Bake off frozen vs trainable in Phase 0. Fallback ladder: AuT-frozen → AuT-trainable → Mimi-frozen → Whisper-small.
- **Projector:** 2-layer MLP with GELU, output_dim = LLM hidden (1024 for Qwen3-0.6B). Audio embeddings replace `<|audio_pad|>` (token ID 151676 in Qwen3-Omni tokenizer) placeholders in the LLM stream. **Do not extend the embedding matrix**; vocab stays at 151,936 + ~8 control tokens.
- **Add second head:** acoustic-only endpoint head at 0-delay, sharing the audio-stream KV. Outputs P(`<END_SPEECH>`) and P(`<EAGER_END_SPEECH>`). This is the architectural differentiator vs Qwen3-ASR, which has no endpoint head.
- **Replace masked LM head with two-head decoder:** audio positions skip LM head; text positions emit text+control vocab only (~152 k entries, not 152 k + 65 k). With continuous AuT embeddings, audio positions don't need any LM head at all.
- **Window strategy:** train with randomized attention window 1–8 s (Qwen3-ASR §5). Same weights serve streaming (1 s window) and offline (8 s window) at inference. Endpoint head trained on the 1 s window only.

### §3.4 — Control tokens
Expand the table:

| Token | Meaning |
|---|---|
| `<CTX>` ... `</CTX>` | (unchanged) |
| `<PROFILE>` ... `</PROFILE>` | User identity facts |
| `<HOTWORDS>` ... `</HOTWORDS>` | Domain vocab list |
| `<HISTORY>` ... `</HISTORY>` | Recent transcript |
| `<NO_SPEECH>` | (unchanged; loss-weight at 0.1×) |
| `<START_SPEECH>` | (unchanged) |
| `<EAGER_END_SPEECH>` | NEW: high-confidence semantic completion, pre-silence |
| `<END_SPEECH>` | (unchanged; confirmed end, after pause + semantic) |
| `<EOS>` | (unchanged) |

### §4.1 — ASR foundation datasets
Replace the table with:

| Dataset | Hours | License | Status |
|---|---|---|---|
| LibriSpeech | 1,000 | CC BY 4.0 | open |
| MLS English | 10,000 | CC BY 4.0 | open |
| People's Speech (CC-BY filter) | 15,000 | CC BY 4.0 | open |
| Common Voice 17 | 3,000 | CC0 1.0 | open |
| Emilia-YODAS (en) | 10,000 | CC BY 4.0 | open |
| **Total** | **~39,000** | | |

**Removed:** GigaSpeech (NC ToA). Target curated subset ~15-20k h remains achievable.

### §4.2 — Conversational data
**Removed:** Spotify Podcast Dataset (discontinued Dec 2023), Fisher Corpus (LDC redistribution blocked).
**Kept with SA disclosure:** AMI (BY), ICSI (BY), CHiME-6 (BY-SA), Earnings-22 train (BY-SA — small volume), TED-LIUM v3 (NC-ND, **eval-only**).

For disfluency training data: add Switchboard or TalkBank/CallHome if budget allows; synthesizing via TTS produces artifacts the model overfits to. See [file 08](./08-risk-register.md) R3.

### §5 Phase 0 — Infrastructure
Replace the "Sanity run" paragraph with: see [file 09](./09-phase0-engineering-plan.md) for day-by-day. Key change: run **three parallel sanity training runs**:
- Run A: Qwen3-0.6B + **AuT-frozen** (extracted from Qwen3-ASR-0.6B safetensors)
- Run B: Qwen3-0.6B + **AuT-trainable** (same init, last 6 layers unfrozen)
- Run C: Qwen3-0.6B + **Mimi-frozen** (the original plan)

Same data slice (LibriSpeech-200h), 6 GPU-h each = ~18 GPU-h total. Optional Run D: Whisper-small encoder if any of A/B/C disappoints. **Decision rule:** the best of A/B beats C by ≥1 pp WER on LS test-clean → ship AuT. Otherwise, decide by the second-best metric (streaming-mode WER from the bake-off's continuous-window eval). The original Mimi-vs-Whisper question becomes obsolete if AuT wins.

Build the eval harness per [file 07](./07-eval-harness-spec.md) in the same week. **Critical path** — no Phase 1 without it. The Phase 0 sanity-run WER smoke test changes from ≤25 % → **≤5 % LibriSpeech-clean** (with AuT init, even on 200 h fine-tune, this is realistic).

### §5 Phase 1 — ASR foundation
**If AuT wins Phase 0:** budget compresses to **5–8 GPU-days FP8** (audio is now ~150 continuous positions/sec, not 400 discrete tokens/sec — roughly halving sequence length). If Mimi wins, budget stays at 12–15 GPU-days FP8. Compute-precision: BF16 attention + MXFP8 GEMMs only; FP8 attention deferred to Phase 2 as a perf optimization. See [file 03](./03-phase0-questions.md) Q2 and [file 05](./05-compute-sanity.md).

Loss weighting: `<NO_SPEECH>` at 0.1×, control tokens at 2.0×, normal text at 1.0×.

If AuT-frozen wins: only the projector + LM is trained in Phase 1. ~30 M trainable params vs ~600 M; another ~2× wall-clock savings. Caveat: domain transfer is bottlenecked by the frozen encoder; if Phase 4 context-biasing eval shows the encoder mis-tokenizes hotwords, unfreeze the top 6 layers as a Phase 4 ablation.

### §5 Phase 2 — Streaming adaptation
Replace static delay with **randomized 1–8 s attention windows** per Qwen3-ASR (see §3.2 above). Same checkpoint runs both streaming (1 s) and offline (8 s) — no separate offline-mode model needed. Combine with a learned minimum-latency policy (MoChA-style, arXiv:2601.22779, 62.5 % token-emission delay reduction) if the randomized-window approach alone fails to hit the streaming WER target. See [file 02](./02-recent-sota.md) recommendation 3, [file 10](./10-qwen3-asr-deepdive.md) §5.

**Streaming-mode WER target:** ≤1 pp absolute over offline (matches Qwen3-ASR-0.6B's 0.92 pp gap as the realistic floor; ≤0.6 pp would be paper-grade).

### §5 Phase 3 — Endpoint detection
- Train **both** heads: the new 0-delay endpoint head and the transcription head's `<END_SPEECH>` token.
- Endpoint head loss: BCE on `<END_SPEECH>` and `<EAGER_END_SPEECH>` per tick, against forced-aligned ground truth.
- Re-baseline §6.1 latency targets: P50 ≤ 200 ms for `<EAGER_END_SPEECH>`, ≤ 400 ms for `<END_SPEECH>`. Stretch: ≤ 100 ms eager. (Meta's 36 ms is acoustic-only on a 1.14M model — different problem.)

### §5 Phase 4 — Context biasing
- Differentiate distractor rates per section: `<HOTWORDS>` at 20% distractor, `<PROFILE>` at 5%, `<HISTORY>` at 0%.
- Ablate distractor ratios at 10%/20%/35% and pick by held-out hallucination rate ([file 08](./08-risk-register.md) R2).
- Optional add-on: **logit-bias injection** mechanism (LOGIC-style, arXiv:2601.15397 — note this paper was withdrawn; cite the architecture not the paper). Cheaper than full retraining for v1.1.

### §5 Phase 5 — Polish + RL
Operationalize preferences per [file 06](./06-architecture-review.md) Issue 5: winners fire within ±200 ms of forced-aligned end-of-word; losers fire >500 ms late or >300 ms during internal pause. No LLM judge.

**Algorithm choice:** swap DPO → **GSPO** (Group Sequence Policy Optimization, Alibaba). Qwen3-ASR uses GSPO on ~50 k utterances for the ASR RL step; it avoids the pair-construction cost of DPO and is the same family as GRPO (used by Kyutai Hibiki-Zero). The reward signal is the same forced-alignment timing criterion above, applied to sampled rollouts. See [file 10](./10-qwen3-asr-deepdive.md) §8, Implication 6.

### §6.1 — Metrics table
Update targets (revised against Qwen3-ASR-0.6B numbers, [file 10](./10-qwen3-asr-deepdive.md) §9):

| Metric | Old target | New target | Notes |
|---|---|---|---|
| WER LibriSpeech clean (final) | ≤ 8 % | **≤ 2.5 %** (AuT init); ≤ 3.5 % (Mimi init) | Qwen3-ASR-0.6B = 2.11 |
| WER LibriSpeech other (final) | ≤ 18 % | **≤ 6 %** (AuT init); ≤ 8 % (Mimi init) | Qwen3-ASR-0.6B = 4.55 |
| WER streaming gap vs offline | not specified | ≤ 1 pp | Qwen3-ASR-0.6B = 0.92 pp |
| `<EAGER_END_SPEECH>` P50 | n/a | ≤ 200 ms | new token; Deepgram Flux precedent |
| `<END_SPEECH>` P50 | ≤ 400 ms | ≤ 400 ms | unchanged |
| `<END_SPEECH>` P95 | ≤ 800 ms | ≤ 800 ms | unchanged |
| Per-tick wall time | ≤ 25 ms | ≤ 25 ms | [file 05](./05-compute-sanity.md) comfortable |
| Hotword recall (Earnings-22, relevant prefix) | ≥ 80 % | ≥ 80 % | **Qwen3-ASR reports zero**: lead chart |
| Hallucination@distractor | ≤ 3 % | ≤ 3 % | **Qwen3-ASR reports zero**: lead chart |
| TTFT (offline, c=1, 0.6B) | not specified | ≤ 150 ms | Qwen3-ASR-0.6B = 92 ms — beat or match |

### §6.2 — Eval harness
Replace with [file 07](./07-eval-harness-spec.md).

### §7 — Compute budget
Two scenarios depending on Phase 0 bake-off outcome:

**Scenario A: AuT-frozen wins (most likely).** Audio at 150 continuous pos/sec; ~30 M trainable params in Phase 1.

| Phase | Old | New (FP8) | Wall-clock |
|---|---|---|---|
| 0 | <1 d | ~0.8 d (18 GPU-h sanity, 3 arms) | 1 week |
| 1 | 5 d | 5–8 d | 1.5–2 weeks |
| 2 | 2 d | 2–3 d | 1 week |
| 3 | 3 d | 4–5 d | 1 week |
| 4 | 3–4 d | 4–6 d | 1–1.5 weeks |
| 5 | 1–2 d | 2–3 d | 1 week |
| **Total** | **~15 d** | **~18–25 d** | **6.5–7.5 weeks** |

**Scenario B: Mimi-frozen wins.** Audio at 400 discrete tokens/sec, vocab extension, all per original plan analysis.

| Phase | Old | New (FP8) | Wall-clock |
|---|---|---|---|
| 0 | <1 d | ~0.8 d | 1 week |
| 1 | 5 d | 12–15 d | 2.5–3 weeks |
| 2 | 2 d | 3–4 d | 1 week |
| 3 | 3 d | 5–6 d | 1.5 weeks |
| 4 | 3–4 d | 5–7 d | 1.5 weeks |
| 5 | 1–2 d | 2–3 d | 1 week |
| **Total** | **~15 d** | **~28–35 d** | **8–9 weeks** |

Bo confirmed budget is sufficient for the worst case. Scenario A would return the project to ~6-7 calendar weeks.

### §8 — Risks
Replace with the prioritized register in [file 08](./08-risk-register.md). Two critical risks (endpoint-latency-vs-delay; prefix hallucination not bounded) are architectural and addressed by §3 changes above.

### Appendix A — References
- Rename "LiveKit Smart Turn v2 (2025)" → **"Pipecat Smart Turn v2 (Daily, 2025)"**. LiveKit ships a separate text-only `turn-detector` (Qwen2.5-0.5B distilled), most recently v0.4.1-intl (Dec 2025).
- Correct "FastTurn-Unified" → **"FastTurn: Unifying Acoustic and Streaming Semantic Cues for Low-Latency and Robust Turn Detection"** (arXiv:2604.01897).
- Add: **Qwen3-ASR Technical Report (arXiv:2601.21337)** — direct competitor at the same 12.5 Hz frontend; source of the AuT encoder.
- Add: **Qwen3-ASR HF model cards** — `Qwen/Qwen3-ASR-0.6B`, `Qwen/Qwen3-ASR-1.7B`, `Qwen/Qwen3-ForcedAligner-0.6B`. Apache 2.0.
- Add: Meta Hierarchical EOT (arXiv:2603.13379) — endpoint latency SOTA.
- Add: Deepgram Flux (2026, production) — model-integrated EOT with `EagerEndOfTurn`.
- Add: FlexiCodec (arXiv:2510.00981) — adjustable 3-12.5 Hz ASR-feature codec.
- Add explicit arXiv IDs for Mini-Omni (2408.16725), Mini-Omni2 (2410.11190), Moshi (2410.00037), Kyutai DSM (2509.08753).
- Add: GSPO (Group Sequence Policy Optimization) — Phase 5 RL algorithm. Find primary cite via Qwen3-ASR references.

---

## What did **not** change

- Backbone choice: Qwen3-0.6B → 1.7B stretch. Confirmed. (Qwen3-ASR also stopped at 1.7B; no 4B variant exists.)
- 12.5 Hz audio token/frame rate. Confirmed — Qwen3-ASR uses the same rate but continuous.
- Delayed-streams interleaving idea. Confirmed (also used by Qwen3-ASR with windowed attention).
- Text context prefix as the central differentiator. **Strongly** confirmed — Qwen3-ASR claims it but reports zero numbers.
- VAD/endpointing as the architectural differentiator. **Strongly** confirmed — Qwen3-ASR has none.
- Single-GPU training on RTX Pro 6000 Blackwell. Memory and per-tick latency very comfortable.
- 5-phase ordering. Phase 5 algorithm changed (DPO → GSPO) but phase intent unchanged.

## What did change since the original synthesis (2026-05-19 Qwen3-ASR deep-dive)

| Before deep-dive | After deep-dive | Why |
|---|---|---|
| Audio frontend: Mimi (12.5 Hz × 32 codes), Whisper-small fallback | AuT-frozen (Qwen3-ASR encoder) → AuT-trainable → Mimi-frozen → Whisper-small | AuT open weights + 40 M-h pretrain make it dominant |
| Embedding extension by 65 k Mimi tokens | `<|audio_pad|>` substitution + MLP projector; no vocab growth | Continuous encoder removes the need |
| Static 1.0 s text delay (Phase 2) | Randomized 1–8 s attention window (Phase 2); single checkpoint serves both modes | Qwen3-ASR §5 |
| LS clean WER target ≤ 8 % | ≤ 2.5 % (AuT) / ≤ 3.5 % (Mimi) | 8 % is 2022 Whisper-base; Qwen3-ASR-0.6B = 2.11 % |
| Phase 5 DPO on endpoint timing | GSPO with forced-alignment reward | Cheaper than pair construction; Qwen3-ASR precedent |
| Phase 1 budget 12–15 GPU-d | 5–8 (AuT path) / 12–15 (Mimi path) GPU-d FP8 | Continuous audio halves sequence length |

---

## Files in this directory

| File | Content |
|---|---|
| [00-synthesis.md](./00-synthesis.md) | This document |
| [01-references.md](./01-references.md) | Appendix A citation verification |
| [02-recent-sota.md](./02-recent-sota.md) | Nov 2025 – May 2026 SOTA scan |
| [03-phase0-questions.md](./03-phase0-questions.md) | Encoder / FP8 / `<PARTIAL>` decisions (superseded on encoder by file 10) |
| [04-dataset-licenses.md](./04-dataset-licenses.md) | License audit, recommended corpus |
| [05-compute-sanity.md](./05-compute-sanity.md) | GPU-day budget math |
| [06-architecture-review.md](./06-architecture-review.md) | Five architectural issues |
| [07-eval-harness-spec.md](./07-eval-harness-spec.md) | Phase 0 deliverable spec |
| [08-risk-register.md](./08-risk-register.md) | Prioritized risks + triggers |
| [09-phase0-engineering-plan.md](./09-phase0-engineering-plan.md) | Day-by-day Phase 0 (Mimi-first; needs updating if AuT wins bake-off) |
| [10-qwen3-asr-deepdive.md](./10-qwen3-asr-deepdive.md) | Qwen3-ASR full tech-stack read; source of the encoder pivot |
