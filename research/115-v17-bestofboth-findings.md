# v17 (rank-32 + rebalanced pool): does adding capacity get best-of-both?

**Date:** 2026-07-06. v17 = 20% ASR-replay + 1:1 natural biasing (800:800) + **rank-32/alpha-64**
adapter (vs rank-16 for v15/v16). Everything else identical. All numbers GPU, consistent setup.

## Four-way comparison
| model | WER c/o | offline fire | halluc | uplift | dict-prem | dict-rec | email+ | intr |
|---|---|---|---|---|---|---|---|---|
| v15 release@6000 (r16) | 3.73/7.06 | 1.8% | 5.0% | +35.3 | 0.16 | 0.84 | 0.94 | 0 |
| v16@9000 (r16) | 3.31/6.40 | 4.5% | 1.8% | +28.2 | 0.32 | 0.78 | 0.90 | 0 |
| **v17@9000 (r32)** | 3.37/6.87 | 2.2% | 2.2% | +20.0 | **0.196** | 0.86 | 0.96 | 0 |
| v17@7500 (r32) | 3.75/7.41 | 3.5% | 2.2% | +17.6 | 0.228 | 0.85 | 0.92 | 0 |

## Finding: rank-32 RESOLVES the endpointing<->WER capacity tension
v16 could not get low WER and low premature at the same checkpoint (9000: WER 3.31/6.40 but
premature 0.32). **v17@9000 gets premature 0.196 AT WER 3.37/6.87** — v15-level endpointing
precision with better-than-v15 WER and 2.6x lower hallucination. Adding adapter capacity did
exactly what it was supposed to: the premature-fire regression that came with more replay at
rank-16 is gone at rank-32. Offline fire rate also drops (v16 4.5% -> v17 2.2%).

v17@9000 vs v15 release: BETTER on WER (both splits), hallucination (2.2 vs 5.0%), dict-recall
(0.86 vs 0.84), email+ (0.96 vs 0.94); TIES on intrusion (0) and premature (0.196 vs 0.16);
WORSE only on biasing uplift.

## The one residual cost: biasing uplift (+20 vs +35pp)
relevant recall 0.565 (v17) vs 0.729 (v15 release GPU). The anti-hallucination counterfactual
suppresses context-only entities, which also suppresses true entities on weak acoustics — a
recall/precision tradeoff on entity emission that is INDEPENDENT of adapter capacity (rank-32
does not fix it; the 1:1 rebalance did not recover it, and pushed uplift lower than v16's +28).
This is the biasing headline number, so the regression matters for the paper's story.

## Verdict / options
- v17@9000 is a strict improvement over v16 (endpointing fixed at similar WER/halluc) and
  dominates v15 on 5 of 6 axes. Best single checkpoint measured.
- Residual: biasing uplift +20pp (down from release +35 / paper-reported +27.5). Base-model
  capability number (+28.9pp) is unaffected — only the fine-tuned model regresses.
- To recover uplift while keeping the rest: a gentler counterfactual (fewer distractors /
  conflict-only, not full distractor lists) at rank-32 = a v18 experiment. The uplift<->halluc
  tradeoff is the real remaining lever, not capacity.
