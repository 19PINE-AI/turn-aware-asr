# Phase 1 — Failure Analysis and Pivot Decision (2026-05-19)

Phase 1 ran the full 30 k-step recipe and failed the gate badly.

## Result

| Run | Steps | LR | Data | Test-clean WER |
|---|---|---|---|---|
| Phase 0 bake-off (`aut_proj_unfrozen`) | 3 000 | 5e-5 | 3 k utt (10 h) | **91.82 %** |
| Phase 1 retry @ step 2 500 | 2 500 | 1e-4 | 10 k utt (35 h) | 116.4 % |
| Phase 1 final @ step 30 000 | 30 000 | 1e-4 | 10 k utt (35 h) | **99.64 %** |

Phase 1 gate target was WER ≤ 30 %. **Failed by ~3×.**

## What the model actually does

```
REF: HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES
HYP: HIS ROOM WAS AS BLACK AS PITCH WITH THE THICK DARKNESS FOR THE SHUTTERS WERE
     CLOSE FASTENED THROUGH FEAR OF ROBBERS AND SO I KNEW THAT HE CO

REF: STUFF IT INTO YOU HIS BELLY COUNSELLED HIM
HYP: WAS ENGAGED IN CONVERSATION WITH GENERAL TILNEY        # from Northanger Abbey
```

The model produces *fluent* English in audiobook style with **memorized phrases from the LibriSpeech books themselves** (e.g., "GENERAL TILNEY", "FETNAH" from Arabian Nights). The audio embeddings affect *which* fiction passage is generated — passing random noise vs zeros vs real audio all produce different but equally hallucinatory outputs — but the audio is being used as a **stylistic seed**, not as a transcription target.

This is a classic modality-bridging failure: the LM prior dominates the audio signal because (a) the LM has many orders of magnitude more parameters than the audio bridge, (b) the training objective (CE on text) rewards *any* high-LM-probability text completion, and (c) the audio embedding can't carry enough signal to discriminate among LM-plausible continuations at this data scale.

## Why the higher LR made it worse

Phase 1 used **lr = 1e-4** vs the Phase 0 bake-off's **lr = 5e-5**, plus 10 × more training steps and 2 × the batch. The combined effect is that the LM weights drifted further into the "I generate LibriSpeech-style fiction" basin of attraction:

| | LM ignores audio | LM transcribes audio |
|---|---|---|
| Loss | low (fluent fiction is high-likelihood under any LM) | low (matches reference) |
| Gradient pull | weak | strong but only with informative audio signal |

At small bridge capacity and high LR, the first mode dominates training.

## Diagnostic: is the audio embedding informative?

I tested with `real audio` vs `random mel` vs `zeros mel` at inference:
- All three produced different outputs ⇒ the model **does** use audio.
- None of them transcribed the reference ⇒ the audio's effect is stylistic, not content-driven.

So the audio path works architecturally; it just doesn't carry enough useful signal under this training regime to override the LM prior.

## Root cause attribution

| Hypothesis | Evidence | Verdict |
|---|---|---|
| Training bug | Phase 0 bake-off (same code) hit 91.8 % | ✗ |
| LR too high | step-2500 WER was 116 % (worse than start) at lr=1e-4 | ✓ contributes |
| Too few audio data | Qwen3-ASR-0.6B's AuT used 40 M h pretrain; we have 35 h fine-tune | ✓ root cause |
| AuT → Qwen3-Base domain shift unbridgeable at this scale | proj-unfrozen only gained 2 pp in bake-off; the rest of the gap remains | ✓ contributes |
| Architecture wrong | model produces fluent English on real audio, just wrong content | partially ✗ — pipeline works, training signal is the problem |

The honest answer: the **AuT → Qwen3-0.6B-Base** adaptation works in principle but needs **orders of magnitude more (audio, text) pairs** to overcome the LM's hallucination prior, OR a different training objective that doesn't reward LM-plausible-but-wrong text.

## Pivot options

| Path | Cost | Expected outcome |
|---|---|---|
| A. Retry with bake-off recipe (lr=5e-5, 3 k steps) | ~7 min | Reproduces 91 % WER; same regime |
| B. CTC head on AuT output (no LM) | ~3 h impl + train | Real ASR but no context-prefix; abandons the Kyutai-style plan |
| C. Switch backbone to Qwen3-Omni-MoE | ~4 h | AuT was designed for this LM; closes the domain shift |
| D. Use Qwen3-ASR-0.6B's full model as the LM (transformers PR pending) | wait for HF | True validation; matches Qwen3-ASR's 2.11 % baseline |
| E. Scale to 1000+ h training data | several days | Standard solution to LM-prior dominance; aligns with Phase 1 plan in `streaming-vad-asr-plan.md` §5 |

The plan's original Phase 1 budget (5–13 GPU-days for 15 k h of data) was *correct*; my Phase 0 sanity scale (35 h) is two orders of magnitude under what Phase 1 actually requires. **Phase 1 at the originally-planned scale was never expected to work in 1 calendar day on a shared GPU.**

## Decision

**Stop autonomous progression and report the bottleneck.**

Continuing Phase 2 / 3 / 4 / 5 against a 99.6 % WER base is pointless — every later phase's gate depends on basic ASR working. The architectural pieces (AuT extraction, two-head decoder, randomized window, endpoint head, context prefix, GSPO scaffold) are all built, tested at the unit level, and committed. They will work *once* Phase 1 produces a usable ASR base.

The plan's original 5–13 GPU-day Phase 1 budget on the original 15 k+ h corpus remains the right path. That's not a 4-hour task — it's a multi-day commitment. Recommending Bo make that commitment with eyes open rather than chase fixes that won't change the underlying compute/data shortfall.

## What is still useful from this session

- `src/aut_encoder.py`: standalone AuT loader, verified roundtrip on Blackwell
- `src/projector.py`, `src/endpoint_head.py`, `src/streaming_model.py`: all functional
- `src/train_phase0.py`: full training loop with bf16, grad-ckpt, periodic saves, divergence watchdog
- `src/train_phase2.py`: randomized-window training (untested but ready)
- `eval/simulator.py`, `eval/metrics.py`, `eval/gate.py`, `eval/run_librispeech.py`, `eval/run_streaming.py`
- `scripts/`: download, prepare, bake-off, Phase 1 launcher
- 13 research documents in `research/`

All in https://github.com/bojieli/streaming-vad-asr.

## Recommendation for next session

1. Pre-tokenize the full ASR foundation corpus (LibriSpeech + MLS-en + People's Speech CC-BY + CommonVoice + Emilia-YODAS = ~39 k h per `research/04-dataset-licenses.md`).
2. Switch to a streaming dataset loader so 100+ GB of mel features can be served without loading into RAM all at once.
3. Run Phase 1 at the originally-planned scale: 3 epochs × ~40 k h, expected ~12–15 GPU-days FP8 per `research/05-compute-sanity.md`.
4. Re-evaluate the architectural choices (CTC vs AR LM, Qwen3-Base vs Qwen3-Omni) after Phase 1 reaches a sub-10 % WER baseline.
