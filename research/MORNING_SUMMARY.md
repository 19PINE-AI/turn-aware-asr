# Morning summary (overnight 2026-07-05)

## The released checkpoint
`checkpoints/merged/qwen3-asr-0.6b-endpoint-unified` (= v15@6000: v12 data balance
+ distractor-ratio + plain-ASR replay + weight-decay/cosine fixes). ONE model,
four behaviors, chosen as best-overall across every benchmark and validated on
enlarged, CI-backed test sets.

## Final numbers (CI-backed, bigger benchmarks)
| benchmark | released model | 95% CI |
|---|---|---|
| conversational recall (100 stretches, 384 bounds) | 0.938 | [0.899, 0.970] |
| conversational false fires / speech-min | 1.03 | [0.69, 1.42] |
| P50 end-of-turn latency | 0.39 s | — |
| dictation premature fires / number (250 seqs) | 0.16 | [0.12, 0.21] |
| dictation final recall / digit acc | 0.84 / 0.998 | — |
| spelled name / email +ctx (240 items) | 1.00 / 0.94 | email [0.88,0.97] |
| wrong-profile intrusion | 0.000 | [≤0.016] |
| offline WER clean / other | 3.73% / 7.06% | — |
| context biasing uplift (Earnings-22) | +27.5 pp | (base model +28.9) |
| seed robustness (3 seeds) | prem 0.17±0.03, email 0.90±0.04, intr 0.004±0.006 | — |

The released model dominates or ties the two prior candidates (v12@6000, v14@7500)
on every axis. Bigger benchmarks *tightened and improved* the estimates (the small
probes were noisy: premature 0.22→0.16, email 0.88→0.94).

## What changed in the paper (28 pp, compiles clean, committed)
1. Restructured to the unified-model story; Section 9 reports the release with
   CI-backed big-benchmark numbers.
2. Two-axis causal principle promoted into intro + contributions + new Figure 2.
3. Figure 1 now shows BOTH operating points — the pure endpointer (0.97/0.3, proves
   the phantom-frontier thesis) and the unified release (0.94/1.03, all behaviors) —
   making the "unification costs a little precision" trade-off explicit, not hidden.
4. New robustness table (3 seeds); dictation table now has conversational-only and
   release rows on the SAME enlarged probes.
5. Consistency pass: plain-ASR replay now correctly described as EXERCISED (recovers
   WER 3.9→3.7, cuts offline over-firing 10-20%→1.8%); base vs release biasing both
   labeled; stale numbers (0.24, 0.65, 3.94-as-release) reconciled.
6. De-densified the densest prose (moved intervals/timeout multipliers to tables).
7. Recipe appendix: weight-decay bug (real, fixed, but NOT the WER culprit — clean
   A/B), cosine, plain-ASR replay, Muon-loses — all reported honestly.

## Honest gaps (documented in the paper, not hidden)
- Earnings-22 natural-speech distractor hallucination (6.3%) is NOT covered by the
  spelled-entity distractor fix; applying the same counterfactual recipe to natural-
  speech biasing is the clear next step.
- ~10% of dictation final-fires land ~2 s late (model keys on silence duration, not
  digit-counting); a learned completeness judge is the general lever.
- The oscillation-vs-monotone multi-seed contrast is incomplete: the causal (monotone)
  side reproduces under a 2nd seed, but the opposed-pools (oscillation) side is single-
  run because its historical data (v6 chain) was deleted. Noted as future work.
- Fine-tune still costs offline WER (+0.9/+1.9 pp vs base); larger ASR-replay fraction
  is the lever.

## Process notes for next time
- The GPU had an intermittent co-tenant; single-process launches + kill-by-PID
  (pkill -f fails in this sandbox) + PYTHONUNBUFFERED. Cap concurrency at 2 GPU jobs
  (3+ dropped utilization to 30%). Watch for HUNG procs (0% cpu, alive), not just crashes.

## Key artifacts
- Paper: paper/main.pdf (commit b2e1f54). Plan/log: research/OVERNIGHT_PLAN.md.
- Release numbers: research/80,81,82,83-*.json. CI tool: eval/bootstrap_ci.py.
- Big probes: data/probes_big/ (regenerable via eval/build_dictation_probes.py +
  build_big_spelled_probe.py). Seeds: checkpoints/semantic_endpoint_v15_seed{1,2}_es.
