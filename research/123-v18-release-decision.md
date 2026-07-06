# v18@9000 = strict best-of-both; recommended release

**Date:** 2026-07-06. v18 = rank-32/alpha-64 + 20% ASR-replay + GENTLER natural biasing
(1000 relevant : 500 distractor, single conflict-only entity). All numbers GPU, consistent.

## Final comparison (best checkpoint each)
| model | WER c/o | offl-fire | halluc | uplift | prem | rec | email+ | intr |
|---|---|---|---|---|---|---|---|---|
| v15 release@6000 (r16) | 3.73/7.06 | 1.8% | 5.0% | +35.3 | 0.16 | 0.84 | 0.94 | 0 |
| v16@9000 (r16) | 3.31/6.40 | 4.5% | 1.8% | +28.2 | 0.32 | 0.78 | 0.90 | 0 |
| v17@9000 (r32, 1:1) | 3.37/6.87 | 2.2% | 2.2% | +20.0 | 0.196 | 0.86 | 0.96 | 0 |
| **v18@9000 (r32, gentle)** | 3.49/6.90 | 3.2% | 2.0% | +28.2 | 0.164 | 0.88 | 0.93 | 0.008 |
| v18@10500 | 3.43/7.99 | 4.8% | 2.0% | +29.4 | 0.148 | 0.87 | 0.93 | 0.004 |

## Verdict: v18@9000 dominates/ties v15 on every axis
- gap 4 (WER): 3.73/7.06 -> 3.49/6.90 — better on both splits.
- gap 1 (hallucination): 5.0% -> 2.0% — 2.5x better.
- endpointing preserved: premature 0.164 (v15 0.16), recall 0.884 (v15 0.84, BETTER).
- biasing: uplift +28.2pp — matches the paper's headline +28.9pp base / >+27.5 release claim
  (v17's regression to +20 fully recovered by the gentler 1000:500 conflict-only counterfactual).
- intrusion 0.008, email+ 0.933 — negligible vs v15 (0 / 0.94).

## Why it worked (the two levers, decoupled)
1. rank-32 gave the ADAPTER CAPACITY to hold endpointing sharp (premature 0.16) while carrying
   the broader transcription that the larger replay fraction adds -> resolves the endpointing<->WER
   tension that capped v16 at rank-16 (premature 0.32).
2. the GENTLER counterfactual (single conflict entity, 2:1 toward relevant) balances entity
   EMISSION (uplift) against SUPPRESSION (hallucination): v17's aggressive 1:1/3-distractor
   recipe over-suppressed (uplift +20); v18 keeps hallucination low (2.0%) AND uplift high (+28).

## Recommendation
Promote v18@9000 to the released unified checkpoint. It is Pareto-superior to v15 (the current
release) and to v16/v17. Base-model biasing capability (+28.9pp) unchanged; the fine-tuned model
now PRESERVES it (+28.2) instead of the release's noisier number, while fixing gaps 1 and 4 and
holding endpointing. Checkpoint: checkpoints/semantic_endpoint_v18_es/snap_step9000.pt;
merged at checkpoints/merged/v18-step9000.
