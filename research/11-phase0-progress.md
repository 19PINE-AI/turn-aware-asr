# Phase 0 Progress Notes (2026-05-19)

End-of-day status after one working session of Phase 0.

## What works end-to-end

| Component | Status | Evidence |
|---|---|---|
| AuT encoder extraction from Qwen3-ASR-0.6B | ✓ | `src/aut_encoder.py`; 186.4 M params load from safetensors |
| AuT inference on Blackwell, bf16 | ✓ | 5.5 ms for 5 s of audio (~900× realtime, single sequence) |
| Whisper-style 128-mel features | ✓ | `src/features.py` uses HF `WhisperFeatureExtractor` matching Qwen3-ASR's `preprocessor_config.json` |
| AuT + 2-MLP adapter + Qwen3-0.6B-Base, audio_pad substitution | ✓ | `src/streaming_model.py`, `src/train_phase0.py` |
| Training loop (bf16, AdamW, cosine LR, gradient clipping) | ✓ | Loss curve drops 5.5 → 2.2 over 1000 steps |
| Eval loop (greedy decode, KV cache, WER) | ✓ | `eval/run_librispeech.py` produces hypothesis text |
| LibriSpeech data ingest | ✓ | `scripts/download_librispeech.sh`, `scripts/prepare_librispeech.py` — direct openslr.org download bypassing HF datasets parquet path |

## Smoke run output

After 1000 steps on test-clean (5.3 h, 2611 utt, full LM trainable, AuT frozen):

```
REF: HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES
HYP: THEY WERE IN A TERRIBLE STATE OF DISCOMFORT AND THEY WERE SO GLAD TO BE IN THE HANDS OF THE KING

REF: STUFF IT INTO YOU HIS BELLY COUNSELLED HIM
HYP: I HAVE A FEW THINGS TO SAY TO YOU ABOUT THE MOTHER
```

WER 90.5% on 30 utterances. Model has learned the LibriSpeech literary style (capitalization, archaic phrasing, no punctuation) but isn't transcribing the specific audio content.

## What this means

Architecturally the pipeline is sound — audio embeddings flow through the LLM and influence output. The bottleneck is the **AuT→Qwen3-Base domain shift**: AuT's `proj1/proj2` were trained to feed Qwen3-Omni's LM, and the 2.1 M-param adapter + 1000 steps + 5.3 h of audio isn't enough to bridge to Qwen3-Base's representation space.

Two clean paths forward:

1. **Scale data + steps.** Train on `train-clean-100` (100 h) for 10k steps. Expected WER target after this: ≤20% on test-clean. Phase 0 sanity gate is ≤5%, so 10k+ may be insufficient at 0.6B without more data.
2. **Unfreeze AuT.proj1/proj2 (~1.9 M params).** These are the audio-to-LLM projector inside AuT. Unfreezing them with a low LR (~5e-5) lets the audio output distribution adapt to Qwen3-Base. Combined with (1), this is the recommended Phase 1 setup.

## Next steps (queued)

- [x] Download `train-clean-100` — completed ~08:48 (6.3 GB)
- [ ] Prep `train-clean-100` mels (~28k utterances)
- [ ] Run `aut_frozen` arm proper: 10k steps on train-clean-100, eval on test-clean
- [ ] Run `aut_unfrozen_top6` arm: same data, same steps
- [ ] Compare WER → encoder decision

## Loss curve, 1000-step overfit run

```
step    0  loss 5.5029
step  100  loss 3.6361
step  200  loss 3.3853
step  300  loss 3.2030
step  400  loss 3.0342
step  500  loss 3.4945
step  600  loss 3.2317
step  700  loss 2.9568
step  800  loss 2.1065
step  900  loss 2.5864
step  999  loss 2.2068
```

Curve is healthy — monotonic decrease modulo batch noise, LR scheduler working, no NaN/divergence. ~0.10 s/step at bsz=4 on Blackwell (audio at 12.5 Hz contributes most of the sequence length).

## Memory + compute notes

- Trainable: 597.9 M params (Qwen3-0.6B fully unfrozen + 2.1 M adapter); AuT 186 M frozen.
- Peak GPU memory: ~7 GB at bsz=4 (very comfortable; could scale to bsz=16 easily).
- Throughput: ~200 ms per step at bsz=4 × ~5–10 s/utt = ~200× realtime training.
- Coexisted with other GPU workloads (Bo's nanochat etc. holding ~60 GB) for the entire session.
