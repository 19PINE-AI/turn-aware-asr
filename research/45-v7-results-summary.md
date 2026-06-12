# v7 results: no_fire kept streaming P50, broke AMI single (2026-06-12)

v6 showed AMI+silence diluted the silence-cue prior. v7 added the
counter-signal: 1000 AMI utts with the markers stripped (audio
unchanged, transcript only). Goal: make silence NECESSARY for
firing, not optional.

Streaming result: **success.** P50 +0.32 s, matches v5.
AMI result: **AMI single collapsed to 0 %.**

The intervention was right in shape but wrong in execution:
the no_fire examples cloned the SAME 500 AMI singles that were
already in training as fire-with-markers. Model resolved the
contradiction by globally defaulting to "don't fire on AMI-style
audio."

## Composition

| | v6 | **v7** |
|---|---|---|
| Total examples | 9994 | **10994** |
| no_fire | 0 | **1000 (9 %)** |
| Trailing-silence | 3500 (35 %) | 3500 (32 %) |
| Truncated | 2494 (25 %) | 2494 (23 %) |
| AMI total | 4284 (43 %) | 5284 (48 %) |
| Early-stop best step | 13500 (score 2.200) | **3000 (score 2.450)** |

## Holdout trajectory

```
  step    S    D  disfl-✓ disfl-✗ trunc-✓ trail-✓  nf-✓  nf-✗  score
  1500  15%  85%   10%    75%    100%   100%    60%   40%   1.625
  3000   0%  75%   70%    30%    100%   100%   100%    0%   2.450 ← BEST
  4500   5%  80%   60%    40%    100%   100%    95%    5%   2.350
  6000   0%  85%   40%    60%    100%   100%    95%    5%   2.200
  7500  10%  50%   85%     0%    100%   100%    95%    5%   2.375
  9000  10%  95%   50%    50%     90%   100%   100%    0%   2.350
```

Holdout S=0 at the chosen best is a strong red flag — the model
has lost LibriSpeech synthetic single firing entirely. Composite
score was misleading because no_fire+trail+trunc all hit perfect
1.0, masking the single collapse.

## AMI conversational eval (50 utts/schema)

| Metric | v3 | v5 | v6 | **v7** |
|---|---|---|---|---|
| Single | 100 % | 82 % | 88 % | **0 %** |
| Double turn detection | 90 % | 76 % | 84 % | **64 %** |
| Disfluency correct | 82 % | 66 % | 56 % | **72 %** |
| Disfluency over-fire | 18 % | 22 % | 40 % | **24 %** |

Disfluency robustness improved (the strict silence cue helps the
model wait through internal pauses). Double dropped because some
AMI doubles end the second turn without trailing silence and the
model under-fires. Single is 0 % — predicted.

## Streaming latency

| Metric | v5 | v6 | **v7** |
|---|---|---|---|
| Detection rate | 100 % | 100 % | **100 %** |
| **P50** | **+0.32 s** | −5.17 s | **+0.32 s** |
| **P95** | +1.38 s | +0.43 s | **+1.07 s** |
| Mean | −2.65 s | −5.53 s | **−1.24 s** |
| Max | +1.95 s | +0.86 s | **+1.59 s** |
| Median chunk infer. | 455 ms | 374 ms | **676 ms** |

**Streaming P50 target hit again.** v7 is the second model to land
P50 between 0 and +1 s. P95 +1.07 s is the cleanest yet.

## The Pareto frontier — four points

| Model | AMI single | AMI double | Stream P50 | Stream P95 |
|---|---|---|---|---|
| v3 | 100 % | **90 %** | −9.57 s | −0.06 s |
| v5 | 82 % | 76 % | **+0.32 s** | +1.38 s |
| v6 | 88 % | 84 % | −5.17 s | +0.43 s |
| v7 | 0 % | 64 % | **+0.32 s** | **+1.07 s** |

v7 is Pareto-equivalent to v5 on streaming, but worse on AMI.
The frontier is now firmly two-point: **v3 for meeting,
v5 for voice assistant.** v6 and v7 are interesting waypoints
but not deployment-grade.

## What v7 actually proves

1. **The no_fire schema concept works for streaming.** P50 +0.32 s
   reproduced cleanly. Adding strict no-fire training preserves the
   silence-cue prior under data dilution.

2. **But the model can't distinguish "fire" from "no_fire" when
   audio is identical.** Cloning the same 500 AMI singles as both
   fire-with-markers and no-fire collapses the firing on that
   audio class entirely.

3. **Single utt detection on real AMI requires the model to fire
   even when trailing silence is short.** The acoustic difference
   between "AMI utterance with 100 ms tail" (should fire in a
   single-utt context) and "first 0.5 s of a long sentence"
   (shouldn't fire mid-stream) is not learnable from these
   features alone.

## Two remaining options

### Option 1: ship two checkpoints, document the trade-off
Production deployment:
  - **v3** for transcription where every turn boundary matters
  - **v5** for low-latency streaming voice assistant
The two checkpoints are now well-characterized. Ship today.

### Option 2: v8 with disjoint audio for fire vs no_fire
Fix the v7 data flaw:
  - Keep 500 AMI singles as fire-with-markers (existing)
  - Sample a FRESH 1000 AMI utts (different meetings or different
    utts within meetings) as no_fire
  - Audio overlap: ~0
The model can then learn audio-conditioned policies without
contradictory training signals on identical audio.

Risk: even with disjoint audio, AMI singles and AMI no_fire
look acoustically very similar (both natural utt endings).
The model may still over-generalize.

Estimated cost: 1 hour data + 1 hour training + 30 min eval.

## Files

- `data/semantic_endpoint_v7/data.pt` — 10994 examples
- `checkpoints/semantic_endpoint_v7_es/best.pt` — step 3000
- `research/43-ami-v7-eval.json` — AMI per-example
- `research/44-streaming-latency-v7.json` — streaming per-example
