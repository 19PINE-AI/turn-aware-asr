# v8 results: disjoint no_fire pool didn't collapse the frontier (2026-06-12)

v7 broke AMI single because the no_fire examples cloned the SAME
500 AMI singles already in training as fire-with-markers. v8 fixed
the data flaw: drew 1000 FRESH AMI utts from the train meetings,
excluding the 500 already in v6 (using meeting_id + text as
fingerprint).

**Result: data fix didn't fix the trade-off.** Each v8 checkpoint
sits at one attractor or the other — never both.

## Composition

| | v7 | **v8** |
|---|---|---|
| Total examples | 10994 | **10994** |
| no_fire | 1000 | **1000 (fresh AMI)** |
| Audio overlap with fire-AMI | 100 % | **0 %** |
| Early-stop best step | 3000 (score 2.45) | **12000 (score 2.70)** |

## Holdout trajectory shows oscillation

```
step    S    D   disfl-✓ trunc-✓ trail-✓  nf-✓  nf-✗  score
1500  0.40 0.75   0.55    0.90    1.00   0.70  0.30  1.975
3000  0.25 0.75   0.60    0.85    0.95   0.90  0.10  2.150
6000  0.95 0.50   1.00    0.80    1.00   0.45  0.55  1.750  ← fire-mode
9000  0.20 0.95   0.70    0.85    0.95   1.00  0.00  2.450
12000 0.10 0.95   0.80    0.95    1.00   1.00  0.00  2.700  ← BEST (no-fire-mode)
15000 1.00 0.90   0.80    0.75    1.00   0.40  0.60  1.850  ← fire-mode
16500 0.75 1.00   0.70    0.90    1.00   0.65  0.35  2.250
18000 0.30 0.70   1.00    1.00    0.95   0.90  0.10  2.550
```

The model bounces between two attractors:
- **Fire-mode** (steps 6000, 15000): single 95-100 %, nf 40-45 %
- **No-fire-mode** (steps 9000, 12000): single 10-25 %, nf 100 %

It can't hold both at once. This is the core finding.

## Two checkpoints evaluated

### v8 best (step 12000) — no-fire-mode

| Metric | Result |
|---|---|
| AMI single | 10 % |
| AMI double | 78 % |
| AMI disfl-correct | 70 % |
| Stream P50 | **−5.21 s** |
| Stream P95 | +0.36 s |

### v8 step 15000 — fire-mode

| Metric | Result |
|---|---|
| AMI single | **90 %** |
| AMI double | **88 %** |
| AMI disfl-correct | 62 % |
| Stream P50 | **−10.41 s** |
| Stream P95 | −0.40 s |

## The frontier across all eight experiments

| Model | AMI single | AMI double | Stream P50 | Stream P95 |
|---|---|---|---|---|
| v3 | **100 %** | **90 %** | −9.57 s | −0.06 s |
| v5 | 82 % | 76 % | **+0.32 s** | +1.38 s |
| v6 | 88 % | 84 % | −5.17 s | +0.43 s |
| v7 | 0 % | 64 % | **+0.32 s** | **+1.07 s** |
| v8 best (12k) | 10 % | 78 % | −5.21 s | +0.36 s |
| v8 step 15k | 90 % | 88 % | −10.41 s | −0.40 s |

No model dominates v3 on AMI AND v5 on streaming. **v8 step 15000
is closest to v3 quality on AMI, but its streaming P50 is the worst
in the table.** The model can't find a representation where AMI
single firing and streaming-silence-waiting both hold.

## The mechanism is now clear

The model has a single global threshold for "fire or not" given
audio-like-this. Training data shifts the threshold:
  - More fire-positive AMI training → fires on AMI singles AND on
    streaming chunk-1 (both look like complete short utts)
  - More no_fire AMI training → doesn't fire on AMI singles AND
    doesn't fire on streaming chunk-1 (both same)

The discriminating feature would be: "does the audio END in
silence?" — but for AMI single utts the trailing silence is ~50-200 ms
(natural micro-pause), and for chunk-1 of a streaming utt the
trailing silence is 0 ms. The model needs to perceive ~100 ms of
trailing energy difference under noisy conversational audio.
With a 0.5 s audio-tower window (the AuT encoder is window-based),
the resolution may simply not be there.

## Project conclusion: ship v3 + v5

After 8 experiments, the evidence is consistent: this architecture
cannot unify "real-conversation endpoint detection" with "streaming
chunk-aware no-fire-mid-utt" into a single checkpoint via LoRA
fine-tuning on Qwen3-ASR-0.6B.

**Production deployment:**
- **v3** (`checkpoints/semantic_endpoint_v3_es/best.pt`) for meeting
  transcription, dictation, push-to-talk — anywhere turn detection
  on real conversational audio matters
- **v5** (`checkpoints/semantic_endpoint_v5_es/best.pt`) for voice
  assistants, streaming captions — anywhere low-latency
  end-of-utterance detection matters

The dual-checkpoint frontier is a real result. It is also the
shippable result.

## What would be needed to unify

If unification is required in a follow-up project, the architecture
would need to change:
1. **Higher temporal resolution on the audio tower** — frame-level
   features (≤50 ms) instead of segment-level
2. **A separate VAD head** that explicitly models trailing-silence
   probability per frame, independent of the LM's semantic head
3. **Streaming-aware training** — interleave training examples
   that contain BOTH "incomplete chunk → no fire" and "complete
   chunk → fire" with frame-level alignment, not just at end-of-audio

Estimated cost for the architectural change: 2-4 weeks. Worth doing
only if the dual-deployment trade-off is unacceptable to the
production target.

## Files

- `data/semantic_endpoint_v8/data.pt` — 10994 examples
- `checkpoints/semantic_endpoint_v8_es/best.pt` — step 12000
- `checkpoints/semantic_endpoint_v8_es/step15000.pt`
- `research/46-ami-v8-eval.json` — AMI per-example (best)
- `research/47-streaming-latency-v8.json` — streaming (best)
- `research/48-ami-v8-step15000.json` — AMI per-example (step 15000)
- `research/49-streaming-latency-v8-step15000.json` — streaming (step 15000)
