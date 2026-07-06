# Gap 2 re-analysis: the dictation "~2 s late" claim does not reproduce on the release

**Date:** 2026-07-06. **Checkpoint:** `semantic_endpoint_v15_es/unified_release.pt` (v15@6000).
**Data:** `research/81-probe-v15release-big.json` (250 digit sequences) joined against
`data/probes_big/digit_probe.json` (per-sequence `last_speech_end_s`).

## Claim under test
MORNING_SUMMARY gap: "~10% of dictation final-fires land ~2 s late (model keys on
silence duration, not digit-counting); a learned completeness judge is the general lever."

## Method
For every sequence, classify each fire by `delta = fire_time - last_speech_end_s`:
- genuine mid-sequence premature: `delta < -0.30` (fires during the number, before the last digit ends)
- on-time end fire: `-0.30 <= delta <= +1.75`
- genuine late fire: `delta > +1.75`

The shipped eval (`dictation_probe_eval.py:104-105`) instead uses `premature = t <= last_end+0.25`
and `final = 0.25 < t <= 1.75`, so any fire in the `(-inf, +0.25]` band is booked as
"premature" and denied final-recall credit — even when it lands essentially at the sequence end.

## Result (n=250)
| metric | shipped window | delta-based |
|---|---|---|
| final recall | 0.844 | **0.992** (248/250) |
| genuine mid-sequence premature / seq | 0.16 | **0.008** (2/250) |
| genuine late fire (>+1.75 s) | — | **0.000** (0/250) |
| end-fire latency p50 / mean | 0.43 s | 0.40 / 0.41 s (min -0.17, max 0.76) |

- **Zero fires are late.** The "~2 s late" characterization is false for the release.
  It was a property of the 2.0 s-tail checkpoints (v9/v12; see `build_v10_training_data.py:141`
  "v12's 2.0 s tail taught the model to wait longer"), which the v13→v15 shorter-tail
  curriculum already fixed.
- The residual is **eager, not late**: fires land ~0.14-0.25 s after the last digit
  (quantized up to the 0.5 s chunk grid at 7.5/8.0/8.5/9.0 s), just past the +0.25 s
  premature cutoff, so the shipped window double-penalizes them (counted premature AND
  denied final credit). Only 2/250 are genuine mid-number fires (id=66 fires at 1.0 & 5.0 s).
- The model correctly holds through the 0.6-1.2 s between-group pauses (else mid-seq
  premature would be high) yet fires within ~0.4 s of the true end — i.e. it is *not*
  purely silence-duration-keyed; it discriminates final-vs-mid pause well.

## Conclusion / how the gap is closed
1. The empirical claim is **corrected**: release dictation has 0% late fires,
   0.8% genuine mid-sequence prematures, 99.2% on-time recall, p50 latency 0.40 s.
2. The shipped `(0.25, 1.75]` window's lower bound is too strict for a low-latency
   endpointer; report the delta-based recall alongside it (or symmetric ±window).
3. The mechanistic note (silence is the dominant learned cue; a content/completeness
   signal is the principled general lever) remains honest **future work**, but it is not
   needed to fix a "2 s late" problem that the release does not exhibit.

Reproduce: `python3 -c` snippet re-scoring `research/81-probe-v15release-big.json` by delta
(see this file's Method). No GPU needed. Optional confirmation: re-run
`eval.dictation_probe_eval --chunk-s 0.25` to show latency is grid-limited, not model-late.
