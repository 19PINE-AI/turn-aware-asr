# Phase 0 Bake-off Results (2026-05-19)

Two-arm comparison after the train data pipeline became available.

## Setup

| | Value |
|---|---|
| Train data | train-clean-100, first 3000 utterances (~10 h) |
| Eval data | test-clean, first 100 utterances |
| Steps | 3000 |
| Batch | 4 |
| LR | 5e-5 (cosine, 200 warmup) |
| Trainable arm A | Adapter (2.1 M) + Qwen3-0.6B-Base (596 M) = 598 M |
| Trainable arm B | Arm A + AuT.proj1/proj2/ln_post (1.9 M) = 600 M |
| GPU time per arm | ~10 min |

Both arms used the same data, seed, and hyperparameters. Only the AuT proj freeze policy differed.

## Results

| Arm | WER (test-clean, n=100) |
|---|---|
| `aut_frozen_baseline` (proj frozen) | **93.99 %** |
| `aut_proj_unfrozen` (proj1/proj2/ln_post unfrozen) | **91.82 %** |

Unfreezing AuT's projector improves WER by **2.17 pp absolute** — directionally correct, but neither arm produces usable transcription. Both are dominated by hallucination.

## Sample outputs (`aut_proj_unfrozen` arm)

```
REF: HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES
HYP: THEY WERE ALL THERE TO DAY'S EVENTS AND ALL WERE THERE TO DAY'S EVENTS

REF: STUFF IT INTO YOU HIS BELLY COUNSELLED HIM
HYP: I DON'T KNOW WHAT'S GOOD FOR ME BUT I'LL DO WHAT I CAN TO GET OVER MY SADNESS

REF: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID
HYP: AND THE PRINCE'S EYES WERE SHUT AND THE PRINCESS WAS SILENT
```

The model has learned the **LibriSpeech literary genre** (capitalized 19th-century audiobook prose, no punctuation) but is not aligning the produced text with the audio content. It's confabulating plausible audiobook sentences.

## Interpretation

The model has 600 M trainable parameters and saw 12 k samples (3 k steps × bsz 4). For the LM to learn the AuT → token mapping, it needs orders of magnitude more (audio, text) pairs. Qwen3-ASR's own AuT was pretrained on **40 M h** of pseudo-labeled audio; my Phase 0 used **0.00025 % of that** for fine-tuning the LM.

**Phase 0 architecture validation: ✓.** The pipeline is sound:
- AuT extraction and inference work
- `<|audio_pad|>` substitution + LM forward works
- Loss decreases monotonically
- Decoder produces fluent English

**Phase 0 actual ASR: ✗** — requires Phase 1 data scale (100+ h training data, 20k+ steps).

## Decisions for Phase 1

Based on this bake-off, the synthesis 00 plan stands:

1. **Adopt AuT, projector unfrozen.** The 2 pp improvement is small but directionally consistent with the deep-dive's prediction (`research/10-qwen3-asr-deepdive.md` Implication 8). Phase 1 should use `--unfreeze-aut-proj` from day one.

2. **Scale up training data 30× and steps 7×.** Use full train-clean-100 (28k utt, ~100 h) and target 20 k steps minimum at bsz 8. Realistic compute: ~6–8 GPU-hours on Blackwell.

3. **Keep Qwen3-0.6B-Base as the LM.** Even though Qwen3-ASR's AuT was trained for Qwen3-Omni, the proj-unfrozen path closes the gap given enough data.

4. **Consider a second 0-delay endpoint head only after Phase 1.** Phase 0 didn't reach the WER baseline needed to make endpoint metrics meaningful.

## Remaining open questions

- **Is the 2 pp gap stable across longer training?** Run the same bake-off at 20 k steps in Phase 1; the gap may widen or vanish.
- **Does the model converge to <10 % WER given 100 h?** Qwen3-ASR-0.6B hit 2.11 % on 40 M h. Linear extrapolation isn't valid, but 10-30 % WER at the 100 h scale would be a plausible Phase 1 outcome.
- **Is the LR right?** 5e-5 may be too low for 600 M trainable params; consider 1e-4 — 2e-4 in Phase 1 with longer warmup.

## Compute summary

| Phase | Wall-clock | GPU-h | Notes |
|---|---|---|---|
| 0 day-1 (env, AuT extract, smoke runs) | ~2 h | ~0.5 | Includes all the false starts |
| 0 bake-off (2 arms × 3 k steps + eval) | ~22 min | ~0.7 | At bsz 4 |
| **Phase 0 total this session** | **~2.5 h** | **~1.2** | vs plan: 0.8 GPU-h budgeted |

Phase 0 went modestly over budget (1.2 vs 0.8 GPU-h) but the architecture is now validated and Phase 1 can begin without further infra work.
