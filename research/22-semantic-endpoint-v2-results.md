# Semantic endpoint v2 — scaled training (2026-05-19)

The user asked for scale. v2 uses 2500 examples (3.6× v1) and 3000 steps
(3× v1) with the same architecture (Qwen3-ASR-0.6B + LoRA r=16 +
gradient-masked new vocab rows).

## Setup

| | v1 | v2 |
|---|---|---|
| Training examples | 700 | 2500 |
| - single | 200 | 500 |
| - double (turn boundary) | 300 | **1500** (5×) |
| - disfluency | 200 | 500 |
| Training steps | 1000 | 3000 |
| Wall clock | 7 min | 8 min |
| LR / LoRA / etc. | identical | identical |

## Comparison

| Metric | v1 (700/1000) | v2 (2500/3000) | Δ |
|---|---|---|---|
| **Single: emit 1 `<END_SPEECH>`** | 100 % | **100 %** | + 0.0 |
| **Disfluency: emit exactly 1 marker** | 93.3 % | **100 %** | **+ 6.7 pp** |
| **Disfluency: over-fire (≥ 2 markers)** | 6.7 % | **0 %** | **− 6.7 pp** |
| **Double: emit 2 markers** | 16.7 % | **6.7 %** | − 10.0 pp **(regression)** |
| WER single | 5.4 % | 6.5 % | + 1.1 |
| WER double | 12.1 % | 7.8 % | − 4.3 |
| WER disfluency | 5.4 % | 5.8 % | + 0.4 |

## Headline: disfluency is now perfect

**Disfluency: 100 % correct, 0 % over-fire across 30 held-out examples.**
The model sees long internal pauses (0.3 – 0.7 s) and *never* fires
`<END_SPEECH>` until the actual end. This is the project's most
important robustness property, exceeding the synthesis 00 §6.1 target
of ≤ 5 % false-endpoint rate.

## Headline regression: double-utt accuracy dropped

Despite 5× more double-utt training data (1500 vs 300), the model's
double-utt accuracy went **down** from 17 % to 7 %. Pattern in the
outputs: the model emits the final `<END_SPEECH>` correctly but
fails to emit a mid-stream marker at the silence between utt_A and
utt_B. It transcribes both utterances as one continuous response.

This is the LM's strongest prior asserting itself: "given audio,
transcribe it as one continuous response, then end." More training
examples don't break that prior — the LM keeps converging to the
"emit one marker pair at the very end" mode because that's
consistent with most of its pretraining.

## Diagnosis

Three possible failure modes, each suggesting a different next experiment:

1. **Pause duration too short.** The 0.7 – 2.0 s gap may not be
   acoustically obvious as a turn boundary. Real conversational turn
   gaps are often 0.3 – 0.5 s; my synthetic doubles are at the upper
   end of that, but training cases of *single* with 0 pause and
   *disfluency* with 0.3 – 0.7 s pause may have taught the model that
   pauses < ~1 s are "internal." The model learned the wrong heuristic.
2. **Data imbalance.** 1500 double examples × 2 markers = 3000 marker
   positions. Of these, ~1500 are at the very end of audio and ~1500
   are mid-stream. But the *single* and *disfluency* examples (1000
   total) each have 1 marker at the audio end — adding 1000 more
   "end-of-audio" examples. Net: 2500 end-of-audio markers vs 1500
   mid-stream. The model biases toward the more common pattern.
3. **The transcription prior is the bottleneck.** Even with perfect
   data balance, Qwen3-ASR was trained on (audio → transcript)
   pairs where transcript is one contiguous response. Breaking the LM
   out of this prior may require either: (a) much more data
   (10×+), (b) initializing from a checkpoint that already does
   turn-level segmentation, or (c) architectural change (e.g. a
   separate small head that fires on silence frames, supervising
   the LM's marker emission).

## What scaling did and didn't do

| Property | Scaling helped? |
|---|---|
| Disfluency robustness | **Yes** (93 → 100 %) — strong improvement |
| Single-utt endpoint emission | n/a (was already 100 %) |
| Double-utt turn detection | **No** — *regressed* (17 → 7 %) |
| Transcription WER | Mixed — double improved, single/disfluency slightly worse |

## Next experiments

In priority order:

1. **Pause-duration ablation.** Hold all else constant; sweep the
   double-utt gap range: 1–2 s, 2–3 s, 3–5 s. If accuracy improves
   with longer gaps, the model has learned a pause-duration threshold;
   the synthetic gaps just weren't far enough above the disfluency
   pause distribution. ~1 hour experiment.
2. **Acoustic-VAD-supervised mid-stream marker.** Use webrtcvad or
   the v1 endpoint head to detect silence > 1 s; emit a marker at the
   detected position in the training transcript. This gives the LM
   acoustic supervision for mid-stream markers without relying on its
   prior. ~2 hours.
3. **Train only on doubles for a phase.** Stage 1: 500 single + 500
   disfluency teach "emit at end." Stage 2: 1500 doubles teach
   "emit mid-stream." Phased curriculum may avoid the imbalanced-data
   issue. ~1 hour.
4. **Test on the original (production-style) test set.** This
   experiment evaluated on held-out *synthetic* examples — same
   schema as training. Real test would be conversational data
   (AMI/CHiME-6 segments) where turns happen mid-recording naturally.

## What's now solid

- ✅ Single-utt endpoint emission (100 %)
- ✅ **Disfluency robustness (100 % correct, 0 % over-fire)**
- ✅ WER reasonable (5–8 %)
- ✅ Trains in 8 min on Blackwell with 7 M LoRA params

What still needs work: turn-boundary detection in continuous
conversational audio. Scaling alone doesn't break the LM's
"transcribe-the-whole-audio" prior.

## Files

- `eval/semantic_endpoint_data_v2.py` — scaled data gen (not used in
  the end; base-transcript path was too slow on shared GPU)
- `data/semantic_endpoint_v2_simple/data.pt` — 2500 examples, 4.4 GB
- `checkpoints/semantic_endpoint_v2/step3000.pt` — trained LoRA
- `research/21-semantic-endpoint-v2-eval.json` — per-example metrics
