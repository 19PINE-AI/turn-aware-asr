# 75 — Dictation enhancement (v10/v11): results

**Date:** 2026-07-05. Raw: `75-dictation-probe-v10.json`, `75-replay-v10gate.json`,
`75-dictation-probe-v11.json`, `75-replay-v11gate.json`, `75-*s9000*.json`.
Baseline: `74-dictation-probe-v9.json` (findings in 74-*.md).

## Headline table

| model | digit prem/seq ↓ | digit final recall ↑ | digit acc ↑ | name (none/prof) | email (none/prof) | intrusion ↓ | replay recall | replay false/min | replay P50 |
|---|---|---|---|---|---|---|---|---|---|
| v9 (causal, paper flagship) | 4.80 | 0.76 | 0.910 | 0.30 / 0.825 | 0.00 / 0.40 | 0.013 | **0.969** | **0.30** | 0.39 s |
| timeout X=0.5 s | 2.00 | 1.00 | — | — | — | — | — | — | 0.50 s |
| timeout X=1.0 s | 0.74 | 1.00 | — | — | — | — | — | — | 1.00 s |
| v10 @4500 | 0.80 | 0.74 | 0.948 | 1.00 / 1.00 | 0.125 / 0.90 | 0.050 | 0.938 | 1.29 | 0.35 s |
| **v11 @3000 (selected)** | **0.36** | 0.78 | **0.996** | **1.00 / 1.00** | 0.025 / **0.925** | 0.113 | 0.917 | 0.97 | 0.45 s |
| v11 @9000 (rejected) | 0.12 | 0.88 | 0.998 | 0.90 / 1.00 | 0.00 / 0.95 | **0.400** | 0.938 | 0.86 | 0.39 s |

Note on @9000: it dominates @3000 on BOTH dictation and the conversational
replay — its only flaw is 40% distractor intrusion (it copies whatever
profile it is given). The trade-off along training time is specifically
context-trust vs everything else; distractor-ratio training should decouple
them and is the top follow-up.

## Findings

1. **The dictation gap closes with data alone.** Same architecture/recipe;
   five added schemas (digit_hold/fire/nosil + spell_name/email) take
   premature fires per dictated phone number from 4.80 (baseline; worse than
   a 0.5 s timeout) to 0.36 — better than every timeout setting, at 0.51 s
   final latency and 0.996 digit accuracy. v11's digit_hold covers any prefix
   length 1–9 (v10's 3/6-only left single-digit-start fires; misses traced in
   74/75 jsons) with FSDD edge-trimming for controlled gaps.
2. **Profile context makes spelled entities work.** Names: 0.30 -> 1.00 exact
   even WITHOUT context (the spell schemas alone teach letter-assembly);
   1.00 with profile. Emails: 0.00 -> 0.925 with profile (normalized written
   form; v9 could not do this at any condition above 0.40).
3. **The cost surface is real and monotone in schema exposure.** Conversational
   replay: recall 0.969 -> 0.938 (v10) -> 0.917 (v11), false fires 0.30 ->
   1.29 -> 0.97/min; pair_hold holdout mirrors it (0.9 -> 0.8 -> 0.9 after
   ×1.5 oversampling, which recovered false fires but not recall). And
   context-copying overshoots with training: distractor intrusion 1.3% (v9)
   -> 5.0% (v10) -> 11.3% (v11@3000) -> 40% (v11@9000). Checkpoint selection
   along one run now trades dictation discipline against context gullibility
   — a capability-breadth vs precision tension at fixed LoRA capacity, NOT an
   oscillation (training is monotone; both classes high simultaneously).
4. **Remaining digit misses are benign-ish:** of 11/50 sequences without an
   in-window final fire, ~6 are shredded by a residual premature fire (the
   segment then reads <10 digits and correctly holds) and ~5 fire LATE
   (~1.8–2.2 s after speech end, just outside the 1.75 s window) — a
   downstream agent still endpoints, delayed.

## Selection

v11@3000 (`checkpoints/semantic_endpoint_v11_es/best.pt`) is the
dictation-enhanced model: dominant on the target axes (premature 0.36,
email-with-profile 0.925) with the replay trade-off reported as-is.
v11@9000 rejected: 40% distractor intrusion (copies the wrong user's email
from context). v9 remains the conversational flagship; the paper reports
both and the trade-off.

## Levers not yet pulled

- Mixing plain-ASR replay data (the known WER mitigation) may also relieve
  the conversational-precision cost.
- Distractor-ratio training for intrusion (paper already names it; now
  measurably urgent at 11.3%).
- Larger LoRA rank / longer schedule to test the fixed-capacity hypothesis.
- Late-fire tail: digit_fire tails 0.4–2.0 s teach some end-of-audio waiting;
  tightening to 0.3–1.0 s may pull the ~2 s late fires into the window.
