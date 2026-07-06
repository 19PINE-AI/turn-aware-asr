# v16 (gaps 1 + 4): natural-speech distractor + larger ASR-replay

**Date:** 2026-07-06. **Checkpoint:** v16@4500 (`checkpoints/semantic_endpoint_v16_es/snap_step4500.pt`,
merged at `checkpoints/merged/v16-step4500`). **Pool:** v12 + 5000 ASR-replay (23%) +
1200 natural-distractor + 600 natural-relevant biasing counterfactuals (LibriSpeech, Haiku entities).
v16 sweep: 5 checkpoints pass the unified gate; 4500 = best premature+email+.

## Gap 1 — natural-speech distractor hallucination (Earnings-22, n=300, GPU, apples-to-apples)
| model | distractor halluc | uplift | no_ctx -> relevant |
|---|---|---|---|
| release (GPU rebaseline) | 5.0% | +35.3 pp | 0.376 -> 0.729 |
| **v16@4500** | **1.8%** | +25.9 pp | 0.365 -> 0.624 |

- Distractor hallucination **5.0% -> 1.8%** (a 64% cut; below the base model's 3.7%).
  The natural-list counterfactuals (context entities absent from the audio, target follows
  the audio) transferred from LibriSpeech training to Earnings-22 eval.
- Honest cost: absolute biasing uplift drops +35.3 -> +25.9 pp (relevant recall 0.729 -> 0.624)
  — the counterfactual makes the model more conservative about emitting context entities, which
  suppresses distractors AND a fraction of true entities. +25.9 pp still matches the paper's
  released +27.5 pp claim. This is the natural-speech analog of the spelled-entity fix, with a
  mild recall cost the spelled case did not show.
- Note: the paper's original 6.3% was a CPU run; GPU re-baseline of the same release is 5.0%.
  Report v16 vs the GPU rebaseline (same setup).

## Behaviors held (big probes, 250 dictation / 240 spelled)
| metric | release | v16@4500 |
|---|---|---|
| dictation premature/seq | 0.16 | 0.19 |
| dictation final recall | 0.84 | 0.81 |
| digit accuracy | 0.998 | 1.00 |
| spelled name+ / email+ | 1.00 / 0.94 | 0.99 / 0.95 |
| email natural+ | 0.95 | 0.95 |
| intrusion | 0.0 | 0.0 |
Dictation softened slightly (the predicted replay-dilution of the fire signal); spelled
recognition held/improved; intrusion still 0.

## Gap 4 — offline WER: the replay lever works (checkpoint-dependent)
| model | clean | other | offline marker-fire | base gap |
|---|---|---|---|---|
| base Qwen3-ASR | 2.79 | 5.14 | — | — |
| release v15@6000 | 3.73 | 7.06 | 1.8% / 1.8% | +0.94 / +1.92 |
| v16@4500 | 4.73 | 8.94 | 9.5% / 16.6% | (too eager — early ckpt over-fires) |
| **v16@10500** | **3.45** | **6.69** | 5.3% / 7.8% | **+0.66 / +1.55** |

- Larger ASR-replay (23% vs 14.5%) **improves offline WER** at the later checkpoints:
  v16@10500 beats the release on both splits (clean 3.73->3.45, other 7.06->6.69),
  shrinking the base gap ~30%. The lever named in the paper is validated.
- Tension: marker-fire suppression develops with training (fire 9.5%@4500 -> 5.3%@10500),
  and v16@10500 still fires more offline than the release (5.3% vs 1.8%) — yet WER is
  lower, so the transcription-quality gain from more replay outweighs the extra truncation.
- Consequence: the probe-behavior-optimal checkpoint (4500) and the WER-optimal checkpoint
  (10500) differ — a capacity trade-off across the run, not a free lunch. Checking whether
  10500 is a viable SINGLE unified checkpoint (halluc + behaviors) — research/98-eval-v16-10500.log.

## v16@10500 as a single unified checkpoint (closes gap 1 AND gap 4)
| metric | release v15@6000 | v16@10500 |
|---|---|---|
| WER clean/other | 3.73/7.06 | **3.45/6.69** |
| Earnings-22 distractor halluc | 5.0% | **1.8%** |
| Earnings-22 uplift | +35.3pp | +24.7pp |
| dictation premature/seq | 0.16 | 0.256 |
| dictation final recall / dacc | 0.844 / 0.998 | 0.816 / 0.998 |
| spelled name+ / email+ | 1.00 / 0.94 | 1.00 / 0.90 |
| intrusion | 0.0 | 0.0 |
| offline marker-fire | 1.8% | 5.3% |

**Verdict:** v16@10500 validates BOTH fixes on one checkpoint — WER improves (gap 4) and
natural distractor hallucination drops 5.0->1.8% (gap 1) — but NOT for free: endpointing
precision (premature 0.16->0.26, offline fire 1.8->5.3%) and biasing uplift (35->25pp)
regress. This is the multi-task capacity trade-off the paper already frames (folding more
behaviors into one rank-16 adapter costs precision); the pure-endpointing / prior release
checkpoint remains available. Both gap fixes are demonstrated; the trade-off is measured,
not hidden.
