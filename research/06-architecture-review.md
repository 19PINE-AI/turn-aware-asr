# Architecture Review — internal critique

**Purpose:** Stress-test the architectural decisions in §3 of the plan before any GPU-day is spent. Five issues worth resolving in Phase 0.

---

## Issue 1 — The `<NO_SPEECH>` token wastes capacity

**Plan says (§3.4):** `<NO_SPEECH>` is emitted at every text-stream position where there is no transcript content. >90% of ticks emit it. At inference, suppressed silently.

**Problem.** The model spends most of its softmax mass on a token it almost always emits and that downstream ignores. Two costs:

1. **Training signal dilution.** Cross-entropy averages over all positions; if 90%+ of positions are trivially-predicted `<NO_SPEECH>`, the gradient on the *interesting* tokens (text, `<START_SPEECH>`, `<END_SPEECH>`) is washed out. The 2× loss weighting in Phase 4 only helps with biased entities, not with this.

2. **Inference compute on null output.** Every 240 ms tick computes a full forward pass to emit `<NO_SPEECH>`. Cheap, but it adds up across many concurrent sessions.

**Alternative.** Kyutai STT uses a **PAD** token over text positions during silence, and the model is trained with the audio-stream loss masked but the text-stream loss kept — including on PADs. They report this works at production scale. The functional difference vs `<NO_SPEECH>` is mostly cosmetic. **Worth keeping for now** but consider:

- **Down-weight `<NO_SPEECH>` loss by 0.1×** during Phase 1-2 so the model focuses CE on content tokens.
- **Or:** sparsify — only run the text head every K ticks, with a fast learned gate predicting "interesting next K ticks." This is more invasive but saves real inference compute.

**Recommendation:** keep the token as designed; add the 0.1× loss weight in Phase 1. Revisit sparsification only if inference latency becomes a problem.

---

## Issue 2 — Delay hyperparameter is the entire endpointing tradeoff

**Plan says (§3.2):** "Delay: text stream lags audio by ~0.5–1.0 s (configurable)."

**Why this needs more thought.** This single number determines:

- **Endpoint detection P50 latency.** The model cannot fire `<END_SPEECH>` until the delayed text position catches up to the end of the utterance audio. A 1.0 s delay means P50 endpoint latency is **floor-bounded at ~1.0 s**, not the 400 ms target in §6.1. There is no way around this for the chosen architecture.
- **WER on streaming output.** More lookahead → better partial transcripts. Kyutai STT uses 0.5 s and reports it's sufficient for accuracy; below 0.5 s, accuracy drops noticeably on disfluent speech.

**The math doesn't reconcile.** §6.1 targets P50 endpoint latency ≤ 400 ms but §3.2 sets delay at 0.5-1.0 s. These are inconsistent. Either:

- (a) drop delay to **~250-300 ms** and accept the WER hit — Kyutai's ablation suggests +0.5-1.0% WER absolute
- (b) raise the latency target to **500-700 ms P50**, which is closer to what Kyutai STT actually achieves
- (c) **decoupled delay** — endpoint head runs at 0 delay (acoustic-only signal) while transcription head runs at 0.5 s delay. More complex but allows fast endpoint + accurate transcript. This is what FastTurn and LiveKit Smart Turn approaches resemble.

**Recommendation:** option (c). Add a tiny acoustic-only endpoint head sharing the audio-stream KV. It's ~5M params, negligible compute, decouples the two latency targets. Validate in Phase 3.

---

## Issue 3 — Loss masking on audio tokens is wasteful

**Plan says (§3.2):** "Audio tokens are loss-masked at output (we only predict text + control)."

**The cost.** The model still computes the full logits matrix over the unified vocab (~152k Qwen3 text vocab + 32×2048 Mimi tokens = potentially 200k+ entries) at every audio-stream position, then throws them away. That's a 200k-wide softmax per audio position, ~12.5 × 3600 × 15000 = **670M positions per epoch** in Phase 1. At 200k vocab × bf16 = 400KB per position-logit-tensor, this is the dominant memory cost in the forward pass.

**Fix.** Two-head decoder à la Moshi:

- Audio positions: skip the LM head entirely (or use a tiny audio-only head if you ever want to model audio for ablations).
- Text positions: full LM head over text + control vocab only (~152k entries).

This is what Mini-Omni and Moshi actually do, and it's why their training fits on smaller GPUs.

**Recommendation:** implement two-head decoder from Phase 1. Don't compute audio logits if you're not training on them.

---

## Issue 4 — Context prefix in §3.5 conflates three different signal types

**Plan says (§3.5):** the prefix is `[USER PROFILE] + [DOMAIN VOCAB] + [RECENT TRANSCRIPT]`, all in one `<CTX>...</CTX>` block.

**Problem.** These three signals want different attention behavior:

- **USER PROFILE** ("Bo's name is Bo Li") — should rarely fire; only when ambiguous proper noun is heard.
- **DOMAIN VOCAB** ("TTS, Mimi, Qwen3") — high prior for these tokens; should strongly bias output distribution.
- **RECENT TRANSCRIPT** — full conversational context; should bias *style* (formality, register) and resolve coreference, not necessarily inject specific words.

Lumping them gives the model no way to learn differential strengths. The 80%/20% relevant/distractor split (Phase 4) is the right intuition for HOTWORDS but is wrong for RECENT TRANSCRIPT (where the prefix is almost always relevant) and probably overkill for USER PROFILE.

**Recommendation:**

- Tag each subsection: `<PROFILE>`, `<HOTWORDS>`, `<HISTORY>` with distinct opening tags.
- Phase 4 distractor sampling: 20% distractor for `<HOTWORDS>` only. `<HISTORY>` should be 100% relevant. `<PROFILE>` should be 95% relevant.
- This costs nothing extra and gives the model explicit cues for how to weight each section.

---

## Issue 5 — Phase 5 DPO on endpoint timing needs a careful preference definition

**Plan says (§5 Phase 5):** "DPO for endpoint timing: build preference pairs where the 'winner' emits `<END_SPEECH>` at a more natural turn boundary than the 'loser.'"

**The problem.** "More natural" is undefined. Two failure modes:

- **Preference leak.** If preferences are constructed by an LLM judge looking at the *transcript*, the model will learn to fire `<END_SPEECH>` at every sentence boundary regardless of audio — because that's where text feels complete. This breaks on disfluent speech.
- **Preference inconsistency.** Human raters disagree on 100-400 ms timing for the same audio at >40% rates (see CHiME-6 turn-detection IAA studies).

**Fix.** Define preferences operationally, not subjectively:
- WINNER: endpoint fires within ±200 ms of forced-aligned end-of-final-word.
- LOSER: endpoint fires >500 ms after final word (slow) OR fires during a >300 ms internal pause (premature).
- Generate preference pairs synthetically by running the Phase 4 model with different decoding temperatures; pair winners with losers from the same audio.

**Recommendation:** rewrite the Phase 5 spec to use these operational criteria. Avoid LLM judges; use forced-alignment ground truth.

---

## Summary of recommended changes to plan

| # | Section | Change | Risk if not done |
|---|---|---|---|
| 1 | §3.4 | Add 0.1× loss weight on `<NO_SPEECH>` | Diluted training signal; slow convergence |
| 2 | §3.2 / §6.1 | Decoupled endpoint head at 0-delay; transcription head at 0.5 s delay | 400 ms latency target unreachable |
| 3 | §3.2 | Two-head decoder: skip audio logits | Wasted memory + compute in every phase |
| 4 | §3.5 | Tag prefix subsections; differentiate distractor rates | Sub-optimal context biasing across all three signal types |
| 5 | §5 Phase 5 | Operational preference definition for DPO | DPO trains on noisy/leaky preferences |

All five are addressable in Phase 0; none require new infra.
