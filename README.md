# streaming-vad-asr

Streaming context-aware VAD + ASR — a unified Kyutai-style delayed-streams model that replaces the cascaded VAD → ASR pipeline. Built on Qwen3-0.6B with the AuT audio encoder extracted from Qwen3-ASR-0.6B.

**Status:** Backbone validated. Qwen3-ASR-0.6B reproduces the paper's 2.11% LibriSpeech test-clean WER (we measured 2.09% on 200 utts). Phase 1 fine-tuning was abandoned in favor of using the open-weights model directly — see `research/13-phase1-failure-analysis.md` and `research/14-qwen3asr-baseline-validated.md` for the pivot. Project differentiators (endpoint head, context-biasing benchmarks) are layered on top of this base.

## Key design choices

| Choice | Rationale |
|---|---|
| AuT encoder (Apache 2.0, extractable from `Qwen/Qwen3-ASR-0.6B`) | 40 M h pretrain; 12.5 Hz continuous output; SOTA WER |
| Qwen3-0.6B backbone | 0.6B → 1.7B stretch; LLM-init transfers cleanly |
| `<\|audio_pad\|>` placeholder + 2-MLP projector | No vocab extension; matches Qwen3-ASR pattern |
| Randomized 1–8 s attention window during training | Single checkpoint serves streaming + offline |
| Decoupled 0-delay acoustic endpoint head | The actual differentiator — Qwen3-ASR has no VAD/endpointing |
| Hotword recall + distractor-hallucination as lead chart | Qwen3-ASR claims context biasing but reports zero numbers |

## Hardware

Single RTX Pro 6000 Blackwell, 96 GB GDDR7. Targets ≤ 25 ms wall-clock per 240 ms streaming tick.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
pip install -r requirements.txt
# Optional, GPU-bound:
pip install transformer_engine[torch] flash-attn --no-build-isolation
```

## Layout

```
src/             # model, data loader, training loop
eval/            # streaming simulator, metrics, gate predicates
scripts/         # extract_aut.py, download_data.py, etc.
data/            # pre-tokenized audio (gitignored)
research/        # plan, synthesis, deep-dives — read 00-synthesis.md first
```

## License

Code: Apache 2.0. AuT encoder weights extracted from `Qwen/Qwen3-ASR-0.6B` retain their original Apache 2.0 license.
