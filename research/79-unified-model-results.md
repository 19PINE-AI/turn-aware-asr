# 79 — Unified model (v12/v14): recipe fixes, Muon A/B, WER recovery

**Date:** 2026-07-05. Goal (Bo): ONE model with conversational endpointing +
dictation + context biasing + low intrusion. Raw: research/76-* (v12), 77-*
(v13 + WER), 78-muon-*, 79-* (v14).

## The intrusion fix (the key result)

v11's dictation+context extension copied whatever profile it was given:
wrong-profile intrusion 11.3% at the composite-best checkpoint, 40% at the
dictation-best. Cause: every context-bearing training example had a *matching*
profile, so the model learned "profile present → emit its entity."

**Distractor-ratio training** (v12+): 35% of spelled examples carry a
*different* identity's profile while the target follows the audio. This teaches
audio-is-ground-truth. Result: intrusion ≈0 at EVERY checkpoint of every run
since (v12 max 0.013; v13, v14 all 0.000), at no cost to dictation or matching
context. This is what makes a single unified model possible.

## Checkpoint sweeps (probes; intrusion 0.000 throughout unless noted)

| model | best step | prem/seq | digit rec | email+ctx | intrusion |
|---|---|---|---|---|---|
| v12 (distractor) | 6000 | 0.28 | 0.72 | 0.85 | 0.000 |
| v13 (tight tails, richer TTS) | 10500 | 0.22 | 0.80 | 0.72 | 0.000 |
| **v14 (v12 data + wd-fix + cosine)** | **7500** | **0.24** | **0.80** | **0.82** | **0.000** |

v14@7500 matches v12@6000 on context, beats it on dictation recall (0.80 vs
0.72), same zero intrusion. Conversational replay comparison: TBD (running).
v13 rejected: tight digit tails made it over-fire conversationally (2.05
false/min @10500) or under-fire (recall 0.875 @3000).

## Recipe findings (Bo asked to strengthen the recipe)

**1. Weight-decay bug (real bug, but NOT the WER culprit).** AdamW's decoupled
weight decay (0.01) shrinks the ENTIRE tied embed/lm_head every step — only the
2 marker rows get gradients, but decay hits all 151k rows (~2.4% over 12k steps,
confirmed empirically). Fix = wd=0 group. Offline WER (LibriSpeech full splits):

| model | clean | other | fire rate |
|---|---|---|---|
| base Qwen3-ASR-0.6B | 2.79% | 5.14% | 0% |
| v9 (wd-bug, constant LR, v9 data) | 3.93% | 7.87% | 4.7/5.9% |
| v13 (wd-fix+cosine, v13 data) | 3.84% | 7.16% | 9.1/13.6% |
| **v12@6000 (wd-bug, v12 data)** | **3.60%** | **6.96%** | 20/25% |
| **v14@7500 (wd-fix+cosine, v12 data)** | **3.94%** | **7.27%** | 10/14% |

**CLEAN A/B (v14 vs v12, IDENTICAL data): the fix does NOT help WER** — v14
(fixed) is 3.94/7.27 vs v12 (unfixed) 3.60/6.96, slightly WORSE and within
checkpoint noise. The earlier v13-vs-v9 "0.71pp recovery" was CONFOUNDED by
different data. Honest conclusion: the offline-WER regression is narrow-schema
drift, NOT the weight-decay bug. Keep the fix for base-model-preservation
rationale (shrinking embedding is a latent hazard at scale), not a WER claim.
Paper corrected to say this (nearly shipped the overclaim). Plain-ASR replay =
the real unexercised mitigation.

**2. Cosine LR decay** (2e-4 → 1e-5): sharpens the final checkpoint; v14
composite trajectory is the highest+most stable (0.992 from step 6000).

**3. Muon — tested, LOSES.** User asked. Implemented (src Muon class,
Newton-Schulz), decoupled LR (muon-lr 2e-2 ≈ 100× AdamW since the orthogonalized
update is ~unit-norm). On identical v13 data: Muon plateaus holdout composite
0.846 by step 3000-4500 with pair_fire collapsing to 0.20; AdamW reaches 0.985.
Muon descends raw loss FASTER early (4.2 vs 8.2 @ step20) but converges worse on
the task. Cause: rank-16 LoRA factors (1024×16) are too rectangular — Newton-
Schulz orthogonalization is poor (O^T O off-diag ~0.5 in a unit test). AdamW
retained. research/78-muon-eval-log.json.

## Decision: v14@7500 = FINAL unified model

Conversational replay: v14@7500 recall 0.938 / false 0.65 (v14@6000 under-fires
0.844; rejected). vs v12@6000 (0.958/0.75). Genuinely close:

| axis | v12@6000 | v14@7500 |
|---|---|---|
| conv recall / false | 0.958 / 0.75 | 0.938 / 0.65 |
| dict premature / recall | 0.28 / 0.72 | **0.24 / 0.80** |
| email+ctx | 0.85 | 0.82 |
| intrusion | 0.000 | 0.000 |
| offline WER clean/other | 3.60/6.96 | 3.94/7.27 |

v14@7500 wins dictation (the capability we set out to add) + false-fires; v12
wins conv-recall + WER + email (all within noise/small). Picked v14@7500 as the
unified release (best at the hard-won dictation, fewest false fires, intrusion
0). Merged to checkpoints/merged/qwen3-asr-0.6b-endpoint-unified. Both are valid
one-checkpoint unified models; a deployer prioritizing raw turn-recall could
prefer v12@6000. Paper Sec 9 rewritten as ONE unified model; recipe appendix
reports wd-fix (bug, no WER win), cosine, Muon-loses honestly.
