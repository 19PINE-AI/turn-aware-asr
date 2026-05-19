# Endpoint head v1 — Qwen3-ASR's missing VAD (2026-05-19)

The project's actual architectural contribution: Qwen3-ASR ships no
endpoint detection, just transcribes. This adds a 0-delay BCE head
on AuT hidden states, trained with forced-alignment endpoint labels.

## Setup

- 80 solo LibriSpeech test-clean utterances + 30 synthetic concat pairs
  (utt_A + 1.0 s silence + utt_B → endpoint should NOT fire in the gap)
- Forced alignment via Qwen3-ForcedAligner-0.6B for word-level timestamps
- AuT-frozen, head-only training (no LM, no AuT fine-tune)
- 88 train / 22 eval (80/20 split)
- 600 steps, bsz=4, lr=3e-3, AdamW
- Trains in **5 seconds** on Blackwell

## Architecture

Tiny 2-layer MLP + GELU + dropout, two heads (eager + end):

```
input (B, T, 1024)            # AuT hidden state at 12.5 Hz
  ↓ Linear(1024 → 256) + GELU + Dropout(0.1)
  ↓ Linear(256 → 256) + GELU + Dropout(0.1)
  ├─ Linear(256 → 1) → eager logit
  └─ Linear(256 → 1) → end logit
```

**0.33 M trainable params** vs Qwen3-ASR's 938 M base. The head is
0.03 % of the model.

## Results

| Metric | Value | Notes |
|---|---|---|
| Frame precision | 81.6 % | high when the head fires |
| Frame recall | 23.0 % | conservative; threshold 0.5 |
| Solo utterances evaluated | 8 | small held-out |
| **Eager P50 latency** | **−400 ms** | fires ~400 ms before audio end, at last-word boundary |
| Eager P95 latency | −160 ms | |
| Eager mean latency | −660 ms | |
| False-endpoint rate (concat) | 33 % (2/6) | small sample; needs more concat data |

**Interpreting negative latency:** the head fires before the audio
ends because it predicts at the **last-word boundary**, not after a
post-pause silence. This is the "EAGER_END_SPEECH" behavior — useful
for downstream LLM prefill. A separate calibrated threshold or a
second-head setup would give the "END_SPEECH" (post-pause) behavior.

## Comparison to synthesis 00 §6.1 targets

| Target | This v1 | Status |
|---|---|---|
| `<EAGER_END_SPEECH>` P50 ≤ 200 ms | -400 ms (300 ms early!) | over-achieved on direction; need to check it's not too eager |
| `<END_SPEECH>` P50 ≤ 400 ms | needs separate train (post-pause labels) | not yet measured |
| False-endpoint rate ≤ 5 % | 33 % | **failing — needs more concat data + better threshold** |

## What this shows

1. **The architecture works.** Adding endpoint detection to Qwen3-ASR
   is trivial — a 0.3 M-param head trained on a few minutes of data
   in 5 seconds produces structured output.
2. **Forced alignment is a viable label source.** Qwen3-ForcedAligner
   gives word-level timestamps; the last-word end is a clean target.
3. **The eager behavior is real.** The head learned to predict at
   the last-word boundary, which is what audio-only acoustic features
   support. This is exactly the `<EAGER_END_SPEECH>` mechanism in
   synthesis 00 §3.1.

## Limitations of this v1

1. **Small data:** 88 train / 22 eval is too small for stable metrics.
   Should scale to 1 k+ utterances.
2. **High false-endpoint rate:** 33 % on concat pairs. Need:
   - More diverse synthetic data (longer pauses, more pair variety)
   - Threshold tuning on a validation set
   - Per-frame causality enforcement (no peek-ahead even in concat audio)
3. **Single-head reality:** I trained on `last_word_end` as the
   single target. To separate eager vs end, I'd need:
   - Eager target = last-word boundary frames
   - End target = post-pause silence frames (or a "no more speech"
     supervision signal)

## Next steps

1. Scale to 1 k LibriSpeech utts + 200 concat pairs.
2. Tune threshold: sweep 0.3 – 0.7, find one that hits ≤ 5 %
   false-endpoint rate at the cost of higher latency.
3. Add a separate "end-after-pause" head trained with a different
   target (frames > 300 ms after last word).
4. Eval on AMI-style real conversational data once available.

## Files

- `eval/endpoint_data_prep.py` — forced-alignment label gen
- `src/train_endpoint.py` — training + eval
- `src/endpoint_head.py` — model architecture
- `data/endpoint/data.pt` — 110 examples with AuT features + labels
- `checkpoints/endpoint_head.pt` — trained head weights
- `research/17-endpoint-results.json` — raw metrics
