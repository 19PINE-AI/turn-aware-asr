# v3 results: AMI fixed, streaming partially fixed (2026-06-10)

After the three-experiment findings (`research/28`) exposed that the
step-6000 model failed on AMI real conversation (18 % double-utt
accuracy) and emitted markers ~13 seconds too early in streaming
mode, I rebuilt the training set to address both failure modes
simultaneously:

- **Truncated examples** (n=494): for each LibriSpeech single-utt source,
  cut the audio at a random 30-80 % point, use the proportional first
  words as text, label with NO end markers. Teaches the LM that
  partial audio should *not* trigger end markers.
- **AMI training examples** (n=1500): real meeting audio from
  10 train-meetings (held out 6 meetings for eval). 500 each of
  single / double-speaker / same-speaker schemas.
- **Existing v2 data** (n=2500): the LibriSpeech-synth data from
  the previous run.

Plus the early-stopping plumbing now exists, so the model is selected
automatically by a composite metric:
```
score = double_acc
      + 0.5*disfl_correct
      + 0.5*trunc_correct
      - 0.5*disfl_overfire
      - 0.5*trunc_misfire
```
Early stopping triggered at step 18000 (patience 4). Best at step 12000.

## Training trajectory (80-utt holdout)

```
  step  single  double disfl-✓ disfl-✗ trunc-✓ trunc-✗  score
  1500   90.0%   80.0%   95.0%    5.0%   65.0%   35.0%  1.400
  3000  100.0%   90.0%   85.0%   15.0%   40.0%   60.0%  1.150
  4500   95.0%   85.0%   75.0%   20.0%   90.0%   10.0%  1.525
  6000  100.0%   95.0%   90.0%   10.0%   60.0%   40.0%  1.450
  7500  100.0%   90.0%   85.0%   15.0%   40.0%   60.0%  1.150
  9000   90.0%  100.0%   80.0%   10.0%   75.0%   25.0%  1.600
 10500   80.0%   95.0%  100.0%    0.0%   70.0%   30.0%  1.650
 12000   85.0%   95.0%   95.0%    5.0%   80.0%   20.0%  1.700  ← BEST
 13500  100.0%  100.0%   85.0%   15.0%   65.0%   35.0%  1.500
 15000  100.0%  100.0%   75.0%   20.0%   65.0%   35.0%  1.425
 16500   95.0%   90.0%   95.0%    5.0%   60.0%   40.0%  1.450
 18000   90.0%   75.0%  100.0%    0.0%   75.0%   25.0%  1.500
```

## AMI conversational eval (50 utts/schema on held-out meetings)

| Metric | v3-step6000 (LS only) | **v3 best (LS + trunc + AMI)** | Δ |
|---|---|---|---|
| Single (≥ 1 marker) | 98 % | **100 %** | +2 pp |
| **Double turn detection** | 18 % | **90 %** | **+72 pp** |
| Disfluency correct | 78 % | 82 % | +4 pp |
| Disfluency over-fire | 20 % | 18 % | −2 pp |

**Massive AMI win.** Turn detection on real conversational audio went
from broken (18 %) to working (90 %). 72 pp improvement from adding
1500 AMI examples to training. This validates the diagnosis: the
prior failure was distribution mismatch, not architectural.

## Streaming latency (30 utts, 0.5 s chunk re-encode protocol)

| Metric | v3-step6000 (LS only) | **v3 best (LS + trunc + AMI)** | Δ |
|---|---|---|---|
| Detection rate | 100 % | 96.7 % | −3.3 pp |
| **Latency P50** | −13.08 s | **−9.57 s** | +3.51 s |
| **Latency P95** | −3.41 s | **−0.06 s** | +3.35 s |
| Latency mean | −11.70 s | −8.53 s | +3.17 s |
| Latency min | −15.94 s | −15.94 s | 0 |
| **Latency max** | −1.68 s | **+0.32 s** | +2.00 s (first positive) |

**Streaming partially fixed.** Adding 494 truncated examples
(~11 % of training mix) moved every percentile in the right
direction. The 95th percentile is essentially at zero. Some
examples now have positive latency (model correctly waits for
audio to end).

But the median is still firing 9.5 s too early. To fully fix
streaming we need more truncated examples — probably 30-40 % of the
training mix instead of 11 %.

## Overall v3 picture

What works now that didn't before:
- **AMI conversational turn detection: 90 %** (was 18 %)
- **Streaming P95 latency: −0.06 s** (was −3.4 s; basically at zero)
- **First positive-latency examples on streaming** — proof the
  truncated training works in principle
- **Early stopping automatically picks the optimum** instead of
  hand-eyeballing the step

What's still broken:
- **Streaming P50 latency: −9.57 s** (was −13 s; better but still
  unusable for production)
- **AMI disfluency over-fire still 18 %** (was 20 %)

The fix for both is the same: **more truncated training examples**.
With 30-40 % truncated, we'd expect:
- Streaming P50 to drop to near zero
- AMI disfluency over-fire to fall (since truncated training teaches
  the model "ambiguous audio = wait" which also helps with mid-utt
  pauses)

## Recommended next experiment

**v4: scale truncated training data.** Generate 4-5× more truncated
examples by also truncating doubles (at points before/after the
first marker) and disfluency examples. Mix to be 30-40 % of training
data.

Estimated effort:
- Data gen: 20 min (mostly truncating existing utts at various
  positions)
- Training: ~1 hour with early stopping
- Eval: 15 min

Goal: P50 streaming latency ≥ −1 s (so 95 % of detections fire within
1 second of true audio end) without regressing the offline metrics.

## Files

- `data/semantic_endpoint_v3/data.pt` — 4500 examples
- `data/semantic_endpoint_v3/meeting_split.json` — AMI train/eval split
- `checkpoints/semantic_endpoint_v3_es/best.pt` — best by composite score
- `checkpoints/semantic_endpoint_v3_es/eval_log.json` — full trajectory
- `research/29-ami-v3-eval.json` — AMI per-example results
- `research/30-streaming-latency-v3.json` — streaming per-example results
