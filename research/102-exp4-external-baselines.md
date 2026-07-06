# Exp-4: external turn-aware baselines on the deployment-matched replay benchmark

**Date:** 2026-07-06. dev 25-stretch set, same audio/detector/scoring as our arms
(eval/external/ harness; RMS gate + [-0.25,+1.5]s window). Adapters in separate venvs;
GPU scoring for Parakeet/Kyutai. As-shipped public thresholds swept, not retrained.

| system | recall | false/min | P50 (s) | resume | streaming WER | note |
|---|---|---|---|---|---|---|
| **ours (causal release)** | **0.97** | **0.3** | **0.39** | 1.0 | (Table main) | in-model semantic EOT |
| Smart Turn v3 (thr 0.3-0.7) | 0.64-0.71 | 3.1-3.5 | 0.64 | 0.56-0.59 | n/a (classifier) | acoustic, VAD-gated |
| LiveKit turn detector (oracle text) | 0.96 | 6.25 | 0.65 | 1.0 | 0.0 (oracle UB) | text EOU, VAD-gated |
| Parakeet-Realtime-EOU-120m | 0.167 | 0.2 | 0.47 | 0.22 | 0.75 (OOD) | <EOU> token; non-OSI |
| Kyutai STT semantic-VAD (2s head, thr 0.6-0.7) | 0.88 | 12.5-14.6 | 0.53 | 0.81 | 0.65 | pause-pred head |

## Finding
No external system reaches our operating corner (high recall AND low false fires):
- Parakeet-EOU is ultra-conservative (recall 0.167) — inside the curve but far below usable recall.
- Kyutai and LiveKit reach high recall (0.88-0.96) only at 6-15 false fires/min — far above ours.
- Smart Turn is mediocre on both axes; its latency floor is VAD-silence + inference (0.64s).
The bolt-on detectors are gated on VAD silence by construction, so they slide along the
timeout curve; the in-model semantic decision is what escapes it. LiveKit's number uses an
ORACLE transcript (WER 0, upper bound) and STILL sits at 6.25 false/min — the limiter is the
VAD-silence gating, not transcript quality. Parakeet/Kyutai WER is high because AMI close-talk
replay is out of their training domain (as-shipped comparison, not best-achievable).

## Paper impact
Rows for tab:main + points/curves for fig3_tradeoff; deletes the "no external baselines"
Limitations sentence. Caveat to state: as-shipped on a common protocol, public thresholds
swept, not retrained; Parakeet non-OSI (eval-only). Artifacts: research/95-exp4-*-dev25.json.
