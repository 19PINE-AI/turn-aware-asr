# v4 results: more truncated, diminishing returns on streaming (2026-06-10)

After v3 showed that adding 11 % truncated training examples moved
streaming P95 latency from −3.4 s to −0.06 s, I scaled v4 to 38 %
truncated and re-trained.

## Setup

| | v3 | **v4** |
|---|---|---|
| Total examples | 4494 | **6494** |
| Truncated examples | 494 (11 %) | **2494 (38 %)** |
| Composition | 1000 S + 2000 D + 1000 F + 494 T + 1500 AMI | same + 1500 single-truncated + 500 double-truncated-early |
| Early-stop best step | 12000 (score 1.700) | **9000 (score 1.800)** |

## Holdout trajectory (80-utt eval at each step)

```
  step  single  double disfl-✓ disfl-✗ trunc-✓ trunc-✗  score
  1500   80.0%   65.0%   85.0%   15.0%   60.0%   40.0%  1.100
  3000   75.0%   85.0%   70.0%   20.0%   90.0%   10.0%  1.500
  4500   80.0%   85.0%   80.0%   20.0%   70.0%   30.0%  1.350
  6000   60.0%   85.0%   70.0%   20.0%  100.0%    0.0%  1.600
  7500   90.0%   85.0%   85.0%   15.0%   80.0%   20.0%  1.500
  9000   80.0%   90.0%   90.0%   10.0%  100.0%    0.0%  1.800  ← BEST
 10500   70.0%   70.0%   90.0%   10.0%   95.0%    5.0%  1.550
 12000   75.0%   75.0%   95.0%    5.0%   90.0%   10.0%  1.600
 13500   90.0%   80.0%   90.0%   10.0%   85.0%   15.0%  1.550
 15000   70.0%   90.0%   80.0%   20.0%   90.0%   10.0%  1.600
```

Holdout: truncated correctness 100 %, misfire 0 % — perfect on
the held-out truncated examples.

## AMI conversational eval (50 utts/schema, held-out meetings)

| Metric | v3 best | **v4 best** | Δ |
|---|---|---|---|
| Single | 100 % | 98 % | −2 pp |
| **Double turn detection** | 90 % | **80 %** | **−10 pp** |
| Disfluency correct | 82 % | 86 % | +4 pp |
| Disfluency over-fire | 18 % | 14 % | −4 pp |

Small AMI regression. The model traded some turn-detection accuracy
for better disfluency robustness and (as we see below) better
streaming behavior.

## Streaming latency progression

| Metric | v2 (0 % trunc) | v3 (11 % trunc) | **v4 (38 % trunc)** |
|---|---|---|---|
| Detection rate | 100 % | 97 % | 97 % |
| **Latency P50** | −13.08 s | −9.57 s | **−7.56 s** |
| **Latency P95** | −3.41 s | −0.06 s | **+0.32 s** |
| Latency mean | −11.70 s | −8.53 s | −7.34 s |
| Latency min | −15.94 s | −15.94 s | −15.94 s |
| **Latency max** | −1.68 s | +0.32 s | **+2.15 s** |
| Median chunk infer. | 245 ms | 244 ms | 354 ms |

**The trend is real but diminishing.** Going from 0 % → 11 %
truncated moved P50 from −13.08 to −9.57 (Δ = 3.5 s). Going from
11 % → 38 % only moved P50 from −9.57 to −7.56 (Δ = 2.0 s).
**Truncated example density alone won't fix the median case.**

What did improve at the upper percentiles:
- **P95 crossed zero in v3 and stays positive in v4** (+0.32 s).
- **Max latency jumped from −1.68 s to +2.15 s** — the best
  examples now correctly wait until the end of speech.

The chunk inference slowdown (244 → 354 ms) is the model emitting
longer partial transcripts at each chunk before deciding to stop.

## Diagnosis

The model has learned two attractors:
1. "Audio looks like a complete short utterance → fire" (default
   behavior from full-utt training)
2. "Audio looks cut off mid-word → don't fire" (from truncated
   training)

But at chunk 1 of a streaming utterance, the partial audio doesn't
look "cut off mid-word" — it looks like "a complete short
utterance." So the model fires.

Pure scaling of truncated examples can't fully fix this because the
underlying ambiguity is real: the first 0.5 s of a long sentence is
acoustically indistinguishable from the entirety of a 0.5 s
utterance ("yeah").

## Recommended next experiment

**v5: trailing-silence training.** Add a new schema where the audio
is a complete utterance followed by 0.3 – 1.0 s of silence, with
markers at the end. This teaches the model:

> Fire `<END_SPEECH>` *after* I see silence following speech, not
> immediately at end-of-audio.

Combined with the existing truncated examples (which teach "no fire
mid-word"), the model should learn the canonical streaming-endpoint
rule:

> Fire iff the audio contains complete speech followed by silence.

This is a different signal than just "wait more" — it gives the
model an explicit acoustic cue (the silence after speech) to
condition on.

Estimated effort:
- Data gen: 30 min (add silence padding to existing complete
  examples)
- Training: 1 hour with early stopping
- Eval: 15 min

Goal: P50 streaming latency between −1 s and +2 s. This is the
production target — within a chunk of the true end-of-speech.

## Honest project trajectory

| | v2 (LS-only) | v3 (LS + AMI + 11 % trunc) | v4 (LS + AMI + 38 % trunc) |
|---|---|---|---|
| LibriSpeech-synth double | 90 % | 95 % | 90 % |
| AMI double | 18 % | 90 % | 80 % |
| Streaming P50 | −13.08 s | −9.57 s | −7.56 s |
| Streaming P95 | −3.41 s | −0.06 s | +0.32 s |

The project is **converging** on real-world deployment:
- AMI conversation: from broken (18 %) to working (80 %) in v3,
  small regression in v4
- Streaming P95: from −3.4 s to +0.32 s — the 95th percentile is
  now correctly waiting
- Streaming P50: still off by 7.6 s; needs the trailing-silence
  schema

## Files

- `data/semantic_endpoint_v4/data.pt` — 6494 examples, 38 % truncated
- `checkpoints/semantic_endpoint_v4_es/best.pt` — best by composite score
- `checkpoints/semantic_endpoint_v4_es/eval_log.json` — holdout trajectory
- `research/32-ami-v4-eval.json` — AMI per-example results
- `research/33-streaming-latency-v4.json` — streaming per-example results
