# Compute Budget Sanity Check

Hardware: 1× RTX Pro 6000 Blackwell Workstation, 96 GB GDDR7, 1.79 TB/s. Verified tensor peaks: FP32 125, BF16/FP16 tensor 252 dense (504 2:4-sparse), FP8 ~504 dense (~1008 sparse), FP4 ~1008 (~2017 sparse). The "4000 AI TOPS" headline = FP4 sparse; for training the real numbers are 252 (BF16) / 504 (FP8) TFLOPS dense.

---

## Claim 1: Phase 1 = 3 epochs × 15k h ≈ 45B tokens, ~5 GPU-days — ⚠ optimistic by ~2-3×

**Token count under delayed-streams tokenization.**
Mimi at 12.5 Hz × 32 codebooks = 400 audio tokens / sec.
Per hour: 3600 × 400 = 1.44 M audio tokens. Text stream at ~3-5 tok/sec adds ~1 %.

- Per epoch: 15 000 h × 1.44 M ≈ **21.6 B tokens**.
- Three epochs: **~65 B tokens** (audio-token-dominated).

The plan's "45B" undercounts by ~30 %; the prompt's "6.5 trillion" is off by 100× (it confuses per-epoch with per-second). Forward+backward FLOPs use the full sequence regardless of which positions contribute to the loss, so the loss-mask does not reduce compute.

**Steps per epoch.**
Effective batch = 64 × 30 s = 1920 audio-seconds = **32 minutes of audio / step** (not 57.6 min — that figure double-counted). 15 000 h × 60 / 32 ≈ **28 100 steps / epoch**, ~84 k steps total.

**FLOPs.**
6 N per token (bf16 fwd + bwd) × 0.6e9 × 65e9 = **2.34 × 10²⁰ FLOPs**.

**Wall-clock at realistic MFU on Blackwell.**
Small dense models with grad checkpointing typically hit 30-45 % MFU.

| Precision | Peak (TFLOPS) | Sustained @ 40 % MFU | Wall-clock for 2.34e20 |
|---|---|---|---|
| BF16 dense | 252 | 100 | **~27 days** |
| FP8 dense (TE) | 504 | 200 | **~13.5 days** |
| FP8 with sparsity (unlikely in training) | 1008 | 400 | ~7 days |

**Verdict: ⚠ tight to unrealistic.** 5 GPU-days needs FP8 + ~60 % MFU + reduced epochs, simultaneously. A defensible Phase 1 budget is **10-15 GPU-days in FP8** or **20-25 in pure bf16**.

**What to adjust.** Pick one or combine: (a) drop to 1 epoch (5 d at FP8); (b) use only 5-8 codebooks for audio conditioning instead of 32 (3-4× speedup, also closer to Moshi practice); (c) use a frozen Whisper encoder so audio enters the LM as ~50 Hz of continuous frames rather than 400 tok/s — this is the single biggest lever; (d) commit to FP8 + FlashAttn-3 from Phase 0 and budget ~12 days.

---

## Claim 2: ≤25 ms per 240 ms tick — ✓ comfortable

**Workload per tick.** Append 96 audio tokens to KV cache (prefill), then decode ≤3 text tokens.

- Prefill 96 tokens, forward only (2 N FLOPs/token): 96 × 1.2e9 = **1.15 × 10¹¹ FLOPs**.
  At 200 TFLOPS sustained (FP8, prefill is compute-bound): **0.6 ms**.
- Decode 1-3 tokens (memory-bandwidth-bound): weights ≈ 1.2 GB bf16 or 0.6 GB FP8. With 1.79 TB/s → **0.34 ms/token bf16, 0.17 ms FP8**.
- Reference: Qwen3-0.6B on H20 with SGLang hits 414 tok/s bf16 (2.4 ms/tok) and 458 tok/s FP8 (2.2 ms/tok) at short context. Blackwell has ~2× the bandwidth and FP8 throughput of H20, so 1-2 ms/token is realistic.
- Plus framework overhead (CUDA graph launch, sampling, Python): ~3-5 ms.

**Total: ~6-10 ms / tick.** Leaves headroom for the 2-3k token context prefix in KV (extra bandwidth cost ~1-2 ms).

**Verdict: ✓ comfortable.** Even at bf16 without FP8 the budget holds.

---

## Claim 3: Memory for bs=64 × 30 s × 0.6 B + ~10 k ctx, bf16 + checkpointing — ✓ very comfortable

Per-sequence audio tokens at 400 tok/s × 30 s = 12 k tokens. With text-stream interleaving, ~12-13 k positions per sequence; matches the plan's "~10k" once delays and sub-codebook tricks are applied.

- Weights bf16: 0.6 B × 2 = **1.2 GB**
- Gradients bf16: **1.2 GB**
- AdamW state (fp32 m, v + master copy): 12 B/param = **7.2 GB**
- Activations with grad-ckpt, FlashAttn-3, seq=12 k, bs=64, hidden=1024, 28 layers:
  full unckpt ≈ 64 × 12 k × 1024 × 2 × 28 = ~44 GB; with √L checkpointing ≈ **6-9 GB**
- Workspace / KV during fwd / NCCL buffers: **2-3 GB**

**Total: ~18-22 GB.** Fits in 96 GB with ~4× headroom.

**Verdict: ✓ very comfortable.** Options unlocked:
- Drop gradient checkpointing → activations ~44 GB, total ~55 GB, still fits and saves ~25 % wall-clock.
- Or grow batch from 64 → 128 (to 30-40 GB activations) and halve step count, which would actually bring Phase 1 closer to the 5-day target.

---

## Bottom line

| Claim | Verdict | Action |
|---|---|---|
| Phase 1 ≈ 5 GPU-days | ⚠ optimistic 2-3× | Re-budget to 12-15 d FP8, or shrink audio token rate, or drop epochs |
| Per-tick ≤ 25 ms | ✓ comfortable | None — likely 5-10 ms in practice |
| Memory fit | ✓ very comfortable | Consider dropping grad-ckpt or doubling batch to reclaim wall-clock |

The realistic Phase-1-through-5 GPU budget is **~25-35 GPU-days** at FP8, not 15. Calendar slack in §7 (4-6 weeks for 15 GPU-days) absorbs this, but the per-phase day counts should each roughly double.

Sources:
- [RTX Pro 6000 Blackwell datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf)
- [WareDB RTX Pro 6000 spec table](https://waredb.com/processor/nvidia-rtx-pro-6000-blackwell)
- [Qwen3 speed benchmark (SGLang on H20)](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)
- [Artificial Analysis — Qwen3-0.6B](https://artificialanalysis.ai/models/qwen3-0.6b-instruct)
