# Phase 0 — Day-by-Day Engineering Plan

Week 1 of the project. Goal: working training+eval pipeline, three parallel sanity runs (**AuT-frozen vs AuT-trainable vs Mimi-frozen**), and the eval harness from `research/07-eval-harness-spec.md` ready to gate Phase 1.

**Revision 2026-05-19 (post Qwen3-ASR deep-dive):** primary encoder arm shifted from Mimi to AuT (extracted from `Qwen/Qwen3-ASR-0.6B`). Mimi remains a fallback arm. See `research/10-qwen3-asr-deepdive.md` and `research/00-synthesis.md` TL;DR.

Hardware: single RTX Pro 6000 Blackwell, 96 GB. Software: PyTorch 2.4+, Transformer Engine, FlashAttention-3, `transformers` for Qwen3-ASR safetensor loading, Mimi inference from `kyutai-labs/moshi`.

---

## Day 1 (Mon) — Environment + base inference

**Morning (4 h):**
- [ ] Install: PyTorch 2.4 + CUDA 12.6, Transformer Engine 1.11+, FlashAttention-3, `moshi[mimi]`, `transformers>=4.46` (Qwen3 support), `datasets`, `librosa`, `jiwer`, `WhisperX` (for forced alignment).
- [ ] Verify FlashAttention-3 attention kernel on Blackwell sm_120 — run their unit tests. Some FA-3 builds default to sm_90 (Hopper); confirm the wheel matches.
- [ ] Validate Mimi codec: encode 10 LibriSpeech clips, decode, listen. Confirm 12.5 Hz frame rate, 32 codebooks/frame output shape.

**Afternoon (4 h):**
- [ ] Load Qwen3-0.6B from HF (`Qwen/Qwen3-0.6B-Base`). Verify forward pass on a text prompt matches the HF reference.
- [ ] Confirm `torch.compile` works on Blackwell with the Qwen3 architecture (recent fix; may need PyTorch nightly).
- [ ] **Decision point:** measure tokens/sec at bs=1, seq=2048, fp8-gemm + bf16-attention via TE. Should be ≥ 800 tok/s decode. If lower, debug the kernel selection.

**Deliverable:** Mimi-encoded audio + Qwen3-0.6B inference both run; tokens/sec measurement recorded.

---

## Day 2 (Tue) — AuT encoder extraction + projector + data loader

**Morning (4 h):**
- [ ] Download `Qwen/Qwen3-ASR-0.6B` from HF. Inspect safetensors — locate `thinker.audio_tower.*` weights (the AuT encoder, ~180 M params).
- [ ] Build a standalone `AuTEncoder` nn.Module wrapping these weights: 3× Conv2D subsampler (128 mel bins → 16, 8× temporal downsample) + 18-layer Transformer (d=896, h=14, FFN=3584) + output proj to 1024. Sinusoidal PE, non-causal windowed attention (~104 tokens ≈ 8 s).
- [ ] Verify roundtrip: feed a LibriSpeech-clean clip through, get ~12.5 Hz continuous embeddings out, compare against Qwen3-ASR-0.6B's own pipeline (decode 1 clip; should reproduce a known WER ballpark).
- [ ] Projector: 2-layer MLP, `1024 → 1024 (GELU, bias) → 1024 (bias)`. Initialize MLP weights with small Gaussian (σ=0.02).
- [ ] Add control tokens to text vocab: `<NO_SPEECH>`, `<START_SPEECH>`, `<EAGER_END_SPEECH>`, `<END_SPEECH>`, `<PROFILE>`, `</PROFILE>`, `<HOTWORDS>`, `</HOTWORDS>`, `<HISTORY>`, `</HISTORY>`. Ten new entries. Do NOT extend vocab for audio (use `<|audio_pad|>` placeholder + projector output instead).

**Alternate path (Mimi arm, run in parallel):**
- [ ] Mimi setup: separate `nn.Embedding(65536, hidden)` for the 32 codebooks × 2048 entries; init as mean of Qwen3 text embeddings + σ=0.02 Gaussian. This is the original-plan path, kept as a fallback bake-off arm.

**Afternoon (4 h):**
- [ ] Build the delayed-streams data loader. Input: (audio, transcript). Output: two interleaved streams aligned per frame, with text-stream delayed by configurable number of frames.
- [ ] **Critical:** the delay must be a *training-time data layout*, not a model-time shift. The model sees `(audio_t, text_t-K)` pairs and predicts `text_t-K+1` autoregressively. Easier to debug, easier to vary at inference.
- [ ] Unit test: roundtrip a 5 s LibriSpeech clip through the loader with delay=0, 6 (≈0.5s), 12 (≈1.0s). Visualize alignment.

**Deliverable:** training-ready batches at bs=8, seq=10k, with delays exposed as a hyperparameter.

---

## Day 3 (Wed) — Two-head decoder + training loop

**Morning (4 h):**
- [ ] Implement two-head decoder per `06-architecture-review.md` Issue 3:
  - Stream-position type embedded as a tag (audio vs text) so the model knows which it's currently emitting.
  - At each position, **only the text head computes logits**; audio positions emit zero loss (skip the LM head entirely).
  - This reduces memory by ~30% vs the masked-loss approach in the plan.
- [ ] Implement loss weighting: `<NO_SPEECH>` at 0.1× weight (Issue 1), regular text at 1.0×, control tokens at 2.0×.

**Afternoon (4 h):**
- [ ] Training loop: TE-autocast (BF16 attention + MXFP8 GEMMs), AdamW (β=0.9/0.95, wd=0.1), cosine LR with 1500 warmup steps, gradient checkpointing on. Use `torch.distributed.tensor.parallel` set up for single-process, multi-stream pipelining.
- [ ] Logging: wandb run with loss curves split by token type (`<NO_SPEECH>`, text, control). Cumulative tokens/sec.
- [ ] Save+resume from checkpoint — restart from intentionally-killed process to validate.

**Deliverable:** training loop runs, loss decreases on 50 steps of LibriSpeech-clean.

---

## Day 4 (Thu) — Eval harness build (per `07-eval-harness-spec.md`)

**Morning (4 h):**
- [ ] Streaming simulator + LibriSpeech-clean WER metric. Validate against published Whisper-base baseline (~5.5%).
- [ ] WhisperX forced alignment for AMI dev set. Cache alignments to disk.

**Afternoon (4 h):**
- [ ] Endpoint metric: P50/P95 latency, premature/late false-endpoint rates.
- [ ] Per-tick wall time + memory metrics.
- [ ] JSON output schema; gate-predicate runner (`eval/gate.py --phase N`).

**Deliverable:** `python eval/run.py --model qwen3-0.6b-base --datasets librispeech-clean-v1` produces a valid JSON report.

---

## Day 5 (Fri) — Sanity runs + go/no-go

**Morning (4 h):**
- [ ] Launch **three parallel** sanity training runs on LibriSpeech-clean 200 h, 1 epoch each, ~6 GPU-h each:
  - **Run A:** Qwen3-0.6B + **AuT-frozen** + 2-MLP projector. Only projector + LM trains.
  - **Run B:** Qwen3-0.6B + **AuT** with top 6 layers trainable + 2-MLP projector.
  - **Run C:** Qwen3-0.6B + **Mimi-frozen** + embedding extension. (Original-plan arm.)
- [ ] Optional **Run D** (if any of A/B/C disappoints): Qwen3-0.6B + frozen Whisper-small + MLP adapter.

**Afternoon (4 h):**
- [ ] Evaluate all runs on LibriSpeech-clean test (5.4 h). Record:
  - WER (final, offline-mode 8 s window)
  - WER (streaming, 1 s window)
  - Per-tick wall time at inference
  - Trainable params count + GPU-h used
- [ ] **Decision gates:**
  - If `min(A, B) ≤ C - 1 pp`: ship AuT. (Pick A vs B by tiebreak: faster GPU-h wins if WER within 0.5 pp.)
  - If `C ≤ min(A, B) - 1 pp`: ship Mimi. (Unexpected outcome — AuT pretrain not transferring.)
  - Otherwise (within 1 pp): pick by streaming WER. Tiebreak: AuT (fewer trainable params, smaller memory).
- [ ] **Sanity floor:** any arm with WER > 5 % on LS clean fails the smoke test. (AuT-frozen on Qwen3-ASR-0.6B's own benchmark hits 2.11 %; even on 200 h fine-tune we should be well under 5 %.)

**Deliverable:** decision documented in `research/11-phase0-results.md`. Phase 1 commencement approved or deferred.

---

## Risk mitigation in Phase 0

| Risk (from `08-risk-register.md`) | Phase 0 mitigation |
|---|---|
| R1 (endpoint latency vs delay) | Day 2-3: data loader exposes delay as hyperparameter, two-head decoder enables future decoupled endpoint head |
| R2 (prefix hallucination) | Day 3: control tokens for `<PROFILE>`, `<HOTWORDS>`, `<HISTORY>` already in vocab; Phase 4 doesn't need re-tokenization |
| R4 (Mimi vs Whisper) | Day 5: parallel sanity runs explicitly compare |
| R5 (hardware fault) | Day 3: checkpoint every 1000 steps; resume tested |
| R7 (Qwen3 init advantage) | Day 5: sanity-run WER curves provide an early signal vs published Whisper baselines |

---

## Out of scope for Phase 0 (deferred to Phase 1 start)

- Pre-tokenizing the full 15k h curated corpus to disk (depends on dataset license findings, ~24 h ingestion).
- Implementing the Phase 4 distractor-prefix data generator (will design in Phase 3).
- FP8 attention enablement (Phase 2 perf optimization).
- DPO preference data construction (Phase 5).

---

## Day-1 setup commands (reference)

```bash
# Python env
conda create -n svadasr python=3.11 -y && conda activate svadasr

# Core
pip install torch==2.4.* --index-url https://download.pytorch.org/whl/cu126
pip install transformer_engine[torch] flash-attn==2.6.3 --no-build-isolation
pip install transformers>=4.46 datasets jiwer librosa soundfile wandb

# Mimi
pip install moshi  # provides mimi.MimiModel.from_pretrained("kyutai/mimi")

# Alignment
pip install whisperx

# Sanity test
python -c "import torch; print(torch.cuda.get_device_name(0))"  # expect RTX PRO 6000
python -c "from transformer_engine.pytorch import fp8_autocast; print('TE OK')"
```

Total Phase 0 wall clock: 5 working days. GPU usage: ~21 GPU-hours (Day 1: 1 h benchmarks, Day 2: 1 h AuT roundtrip, Day 3: 1 h debug, Day 5: 18 h sanity runs across 3 arms).
