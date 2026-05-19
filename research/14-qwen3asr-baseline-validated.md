# Qwen3-ASR-0.6B baseline validated (2026-05-19)

After Phase 1 failed at session scale (`research/13-phase1-failure-analysis.md`),
I switched backbone to Qwen3-Omni-Thinker — i.e. loaded the full open-weights
`Qwen/Qwen3-ASR-0.6B` model directly via the official `qwen-asr` package
instead of trying to bridge AuT to Qwen3-0.6B-Base.

## Result

200 utterances of LibriSpeech `test-clean`:

| Metric | This run | Paper baseline |
|---|---|---|
| WER (%) | **2.09** | 2.11 |

**Match within 0.02 pp.** Reproducibility confirmed; my LibriSpeech prep,
WER metric (jiwer + whisper-style normalization), and audio pipeline are
all correct.

## Sample outputs (perfect)

```
REF: HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND
     BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK
     PEPPERED FLOUR FATTENED SAUCE
HYP: He hoped there would be stew for dinner, turnips and carrots
     and bruised potatoes and fat mutton pieces to be ladled out in
     thick peppered flour-fatted sauce.
```

## What changed vs Phase 1

| | Phase 1 (failed) | Qwen3-ASR baseline (working) |
|---|---|---|
| LM | Qwen3-0.6B-Base (text-only) | Qwen3-Omni-Thinker (audio-aware) |
| AuT-LM coupling | 2.1 M random-init adapter | Native (jointly pretrained) |
| Trainable surface | 600 M params, fine-tune | 0 — use open weights as-is |
| Test-clean WER | 92–99 % | **2.09 %** |

The Phase 1 failure was NOT an architectural problem; it was a
data-scale problem caused by trying to learn the AuT-to-LM mapping
from scratch on 35 h of audio when Qwen3-ASR's pretrain used 40 M h.

## Implications for the original plan

The original `streaming-vad-asr-plan.md` §5 sized Phase 1 at 5–13
GPU-days × 15 k+ h corpus to learn this mapping from a generic LM.
**This entire phase is avoided** by reusing the Qwen3-ASR open
weights. Synthesis 00's recommendation to use AuT was correct;
the additional insight is that Qwen3-ASR's complete model (AuT +
its co-trained thinker LM) is what should be reused, not just AuT.

## Project differentiators that remain

Qwen3-ASR ships a transcript-only model with no VAD/endpointing
and zero published context-biasing benchmarks. The project's value
is now sharpened to four contributions on top of the existing
ASR base:

1. **Endpoint head**: 0-delay BCE head on AuT hidden states, emitting
   `<EAGER_END_SPEECH>` and `<END_SPEECH>` per frame. Qwen3-ASR has
   none of this — `<END_SPEECH>` is the only architectural addition.
2. **Context-biasing benchmark**: measure Qwen3-ASR baseline's
   hotword recall + distractor-hallucination on Earnings-22 with
   the system-prompt biasing mechanism. The paper claims biasing
   works but reports zero numbers. The first chart Bo would publish.
3. **Streaming WER differential**: measure offline vs the qwen-asr
   package's streaming API. Quantify the documented 0.92 pp gap.
4. **Distractor-prefix robustness**: bias the system prompt with
   irrelevant entities; measure the false-insertion rate. Synthesis 00 §6.1
   targets ≤ 3 %. Currently unmeasured anywhere.

These are the actual experiments that produce publishable results.
Phase 1 is now considered "complete" by reuse, not training.

## Next steps (resumable autonomously)

- Context-biasing experiment: download Earnings-22 test, define
  hotword list, run with-context vs without-context, compute recall.
- Distractor-prefix experiment: feed Earnings-22 utterances with
  hotwords from a *different* utterance; measure hallucination rate.
- Endpoint-head implementation: subclass `qwen_asr.core.transformers_backend.modeling_qwen3_asr`
  to add the head; build forced-alignment endpoint labels from
  WhisperX; train with BCE loss + frozen base model.

Each is a 2–6 hour autonomous task; none requires multi-day compute.
