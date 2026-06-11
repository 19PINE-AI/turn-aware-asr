# v6 results: AMI-with-trailing-silence didn't collapse the frontier (2026-06-11)

v5 won streaming (P50 +0.32 s) but lost AMI conversation (double 76 %).
v3 won AMI (double 90 %) but lost streaming (P50 −9.57 s). v6 added
1500 examples of "AMI utterance + 0.3-1.0 s synthetic silence" to try
to teach the model: AMI-style audio CAN be followed by silence, so
the silence cue still applies on conversational speech.

**Result: v6 is a new point on the frontier, not above it.** It sits
between v3 and v5 on both axes — better AMI than v5, better streaming
than v3, but dominated by neither.

## Composition

| | v5 | **v6** |
|---|---|---|
| Total examples | 8494 | **9994** |
| Trailing-silence | 2000 (24 %) | **3500 (35 %)** |
| ↳ from AMI | 0 | **1500** |
| Truncated | 2494 (29 %) | 2494 (25 %) |
| AMI total | 2784 (33 %) | **4284 (43 %)** |
| Early-stop best step | 6000 (score 2.275) | **13500 (score 2.200)** |

## Holdout trajectory (selected steps)

```
  step    S    D  disfl-✓ disfl-✗ trunc-✓ trunc-✗ trail-✓ trail-✗  score
  3000  95% 85%    90%     0%    55%    45%    100%     0%   1.850
  6000  70% 75%    95%     5%    70%    30%    100%     0%   1.900
  9000  85% 85%    80%    10%    75%    25%    100%     0%   1.950
 10500  90% 100%   85%    15%    75%    25%    100%     0%   2.100
 13500  70% 95%    75%    15%    95%     5%    100%     0%   2.200  ← BEST
 15000  90% 100%   70%    30%    75%    25%    100%     0%   1.950
 19500 100% 90%    80%    15%    75%    25%    100%     0%   1.975  (stop)
```

Best step 13500: trail 100 % + trunc 95 % + double 95 %. Looks ideal
on holdout; the test came from real evals.

## AMI conversational eval (50 utts/schema, held-out meetings)

| Metric | v3 best | v4 best | v5 best | **v6 best** |
|---|---|---|---|---|
| Single | 100 % | 98 % | 82 % | **88 %** |
| **Double turn detection** | **90 %** | 80 % | 76 % | **84 %** |
| Disfluency correct | 82 % | 86 % | 66 % | **56 %** |
| Disfluency over-fire | 18 % | 14 % | 22 % | **40 %** |

AMI double recovered partially (76 → 84 %), but disfluency over-fire
worsened (22 → 40 %). The model now fires too eagerly on AMI
disfluency examples — probably because mid-thought silences look
like end-of-thought silences after v6's silence-emphasis training.

## Streaming latency

| Metric | v3 | v4 | v5 | **v6** |
|---|---|---|---|---|
| Detection rate | 97 % | 97 % | 100 % | **100 %** |
| **Latency P50** | −9.57 s | −7.56 s | **+0.32 s** | **−5.17 s** |
| **Latency P95** | −0.06 s | +0.32 s | **+1.38 s** | +0.43 s |
| Latency mean | −8.53 s | −7.34 s | −2.65 s | **−5.53 s** |
| Latency max | +0.32 s | +2.15 s | +1.95 s | +0.86 s |
| Median chunk infer. | 244 ms | 354 ms | 455 ms | 374 ms |

**P50 regressed from +0.32 s back to −5.17 s.** The fire-on-silence
prior collapsed when AMI examples were added (even with silence).
Most likely cause: AMI examples have short, complete-looking
utterances, and the model defaults to "this looks like a complete
short utterance → fire" before the silence padding starts.

## The Pareto frontier — three points now

| Model | AMI single | AMI double | AMI disfl-✓ | Stream P50 | Stream P95 |
|---|---|---|---|---|---|
| **v3** | 100 % | **90 %** | 82 % | −9.57 s | −0.06 s |
| **v5** | 82 % | 76 % | 66 % | **+0.32 s** | **+1.38 s** |
| **v6** | 88 % | 84 % | 56 % | −5.17 s | +0.43 s |

v6 sits between v3 and v5 on both axes. None of the three
dominates. **The original Pareto frontier hypothesis from v5
still stands** — there is a genuine trade-off between conversational
endpoint accuracy and streaming-latency P50.

## Why mixing didn't work

The hypothesis was: AMI + silence examples teach the model that
silence-after-AMI is still a valid trigger, so it keeps v5's
silence-cue behavior. The actual result: the model treats v6's
two example types asymmetrically.

- **At training time:** with AMI examples (no silence) AND AMI
  examples (with silence) both labeled with markers, the model
  finds the path-of-least-resistance feature: short audio → fire.
  The trailing silence becomes optional rather than load-bearing.

- **At streaming time:** chunk-1 of a long utterance LOOKS like
  a short audio. Model fires.

This is the same failure mode as v3 — and v6 effectively diluted
v5's "silence is required" training signal back toward v3's
"speech complete is sufficient" prior.

## What I'm taking from v6

1. **Adding AMI-with-silence wasn't enough.** To maintain v5's
   strict silence prior, we'd need to *also* show AMI examples
   where the model must NOT fire even though speech is complete
   — e.g., short conversational utterance followed by *no* silence
   and *no* markers. Currently every AMI utt has markers.

2. **The frontier is real.** Three points, all on the curve.
   This isn't an artifact of weak training — it's a fundamental
   tension between two valid endpoint policies.

3. **For a single production deployment, choose the policy:**
   - Voice assistant / push-to-talk: ship **v5**
   - Meeting transcription / conversation: ship **v3** or **v6**
     (v6 trades 6 pp double-detection for ~4 s better P50)

## Two paths forward

### Path A: accept the frontier
Ship v3 and v5 as two checkpoints for two deployment contexts.
Document the trade-off. This is honest and shippable today.

### Path B: try v7 with strict no-fire training on AMI-style audio
Add a fifth schema: "AMI-like audio with NO markers and NO trailing
silence." Examples where the model must *not* fire even though the
utterance is complete. This forces it to require the silence cue.

  - 1000 AMI single utts with audio cut at the natural end (no
    silence padding) and no markers in text. Model must NOT predict
    an end token.
  - The training signal becomes:
    * audio + silence + markers → fire
    * audio + no silence + no markers → don't fire
    * truncated audio + no markers → don't fire (existing)

This is a stronger signal than v6's "silence is one option among
many" — it makes silence necessary.

Estimated cost: 30 min data gen + ~1 hour training + 15 min eval.

## Files

- `data/semantic_endpoint_v6/data.pt` — 9994 examples
- `checkpoints/semantic_endpoint_v6_es/best.pt` — step 13500
- `checkpoints/semantic_endpoint_v6_es/eval_log.json` — holdout
- `research/40-ami-v6-eval.json` — AMI per-example
- `research/41-streaming-latency-v6.json` — streaming per-example
