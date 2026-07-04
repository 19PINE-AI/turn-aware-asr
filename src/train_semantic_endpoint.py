"""Fine-tune Qwen3-ASR thinker LM to emit semantic endpoint tokens.

Strategy:
  - Load Qwen3-ASR-0.6B with the official package.
  - Extend tokenizer with EAGER_END_SPEECH and END_SPEECH.
  - Resize input embeddings AND lm_head; init new rows from existing mean + noise.
  - LoRA on thinker text-model attention modules (q_proj, k_proj, v_proj, o_proj).
  - Train only: LoRA deltas + new-token embedding rows + new-token lm_head rows.
  - Frozen: AuT, original thinker body weights, projector.
  - Standard chat format; loss only on assistant text (everything after <|im_start|>assistant\\n).

This is meant to add a *new capability* (semantic endpoint emission) to the
already-SOTA base, without regressing the transcription. Total trainable
surface: ~5-10 M params depending on LoRA rank.

Usage:
    python -m src.train_semantic_endpoint \\
        --data data/semantic_endpoint/data.pt --steps 2000
"""

from __future__ import annotations
import argparse
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


END_TOK = "<END_SPEECH>"
EAGER_TOK = "<EAGER_END_SPEECH>"

ASSISTANT_PREFIX_TOKENS = ["<|im_start|>", "assistant", "\n"]


def load_base_model(cache_dir: str = "data/qwen3-asr-0.6b-pkg"):
    """Load Qwen3-ASR-0.6B via the qwen-asr package and return (model, tokenizer, processor)."""
    import json as _json
    from qwen_asr import Qwen3ASRModel
    wrapper = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        cache_dir=cache_dir,
        max_inference_batch_size=4,
        max_new_tokens=256,
    )
    model = wrapper.model            # Qwen3ASRForConditionalGeneration
    processor = wrapper.processor    # Qwen3ASRProcessor
    tokenizer = processor.tokenizer
    # The chat template is shipped as a separate file; install it on the
    # tokenizer so apply_chat_template works.
    chat_template_path = Path("data/qwen3-asr-0.6b/chat_template.json")
    if chat_template_path.exists():
        with open(chat_template_path) as f:
            tokenizer.chat_template = _json.load(f)["chat_template"]
    return model, tokenizer, processor


def extend_tokenizer_and_model(model, tokenizer, processor) -> tuple[list[int], int, int]:
    """Add EAGER_END_SPEECH and END_SPEECH; init their embedding + lm_head rows.

    The Qwen3-Omni tokenizer reserves slots up to 151936 for future special
    tokens; when we add EAGER_END_SPEECH and END_SPEECH they go into the
    reserved range (e.g. 151705, 151706), which means their rows in
    embed_tokens / lm_head already exist (likely zero or random). We need to
    re-init those *specific* rows, not blindly extend the vocab.

    Returns (new_token_ids, eager_id, end_id). new_token_ids is the list of
    just-added IDs (gradient masking should preserve gradients for these
    rows and zero everything else).
    """
    thinker = model.thinker

    tokenizer.add_special_tokens(
        {"additional_special_tokens": [EAGER_TOK, END_TOK]}
    )
    eager_id = tokenizer.convert_tokens_to_ids(EAGER_TOK)
    end_id = tokenizer.convert_tokens_to_ids(END_TOK)
    new_ids = [eager_id, end_id]

    embed = thinker.model.embed_tokens
    lm_head = thinker.lm_head
    # Re-initialize ONLY these specific rows
    with torch.no_grad():
        mean_e = embed.weight.mean(dim=0)
        mean_h = lm_head.weight.mean(dim=0)
        std = 0.02
        for tid in new_ids:
            embed.weight[tid] = mean_e + torch.randn_like(mean_e) * std
            lm_head.weight[tid] = mean_h + torch.randn_like(mean_h) * std

    logger.info(
        "Tokenizer extended: vocab=%d, eager=%d, end=%d (new rows initialized)",
        len(tokenizer), eager_id, end_id,
    )
    return new_ids, eager_id, end_id


def apply_lora(thinker, rank: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Apply LoRA to thinker.model attention modules.

    We do NOT freeze new-token embedding rows / lm_head rows; those will be
    trained via gradient zeroing for old indices (see train loop).
    """
    from peft import LoraConfig, get_peft_model

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        modules_to_save=None,  # we'll handle embedding/lm_head separately
    )
    peft_thinker = get_peft_model(thinker.model, cfg)
    thinker.model = peft_thinker.base_model.model  # unwrap
    # Reset all params to frozen, then unfreeze only the LoRA tensors
    return peft_thinker  # return the wrapped object for completeness


def freeze_except_lora_and_new_rows(model, new_token_ids: list[int]):
    """Freeze AuT and original LM body. Unfreeze: LoRA deltas + only the
    specific rows in embed_tokens / lm_head corresponding to new token IDs.

    Because embed_tokens / lm_head are stored as single tensors, we use a
    gradient hook to mask out gradients on rows we don't want to train.
    """
    thinker = model.thinker
    # Default: freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze LoRA params on the text model
    for name, p in thinker.model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True

    # Allow grad on full embed_tokens / lm_head, then mask in hook
    thinker.model.embed_tokens.weight.requires_grad = True
    thinker.lm_head.weight.requires_grad = True

    new_ids = sorted(set(new_token_ids))

    def keep_only_new_rows_hook(grad: torch.Tensor):
        # grad: (vocab, hidden). Build a mask that's 1 for new_ids, 0 elsewhere.
        mask = torch.zeros(grad.shape[0], 1, device=grad.device, dtype=grad.dtype)
        for tid in new_ids:
            mask[tid] = 1
        return grad * mask

    thinker.model.embed_tokens.weight.register_hook(keep_only_new_rows_hook)
    thinker.lm_head.weight.register_hook(keep_only_new_rows_hook)


def build_inputs(processor, examples: list[dict], device, dtype):
    """Use the Qwen3-ASR processor to build inputs from (audio, text) pairs.

    For each example, we build the prompt twice via apply_chat_template:
      1. system + user (with empty assistant placeholder) → "prefix" length
      2. system + user + actual assistant content → "full" length
    Then labels[:prefix_len] = -100, labels[prefix_len:] = ids[prefix_len:].

    This avoids needing to find the assistant boundary by token-matching.
    """
    audios = [np.asarray(e["audio"], dtype=np.float32) for e in examples]
    tokenizer = processor.tokenizer

    text_prompts = []
    prefix_lens = []
    for e in examples:
        # Prefix: system + user, add_generation_prompt=True (adds the
        # "<|im_start|>assistant\n" header but no content).
        msgs_prefix = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio"}]},
        ]
        prefix_str = tokenizer.apply_chat_template(
            msgs_prefix, tokenize=False, add_generation_prompt=True,
        )
        prefix_ids = tokenizer(prefix_str, add_special_tokens=False).input_ids
        # Full: prefix + the assistant content (we append manually because
        # apply_chat_template with role='assistant' may behave differently).
        full_str = (
            prefix_str
            + f"language English<asr_text>{e['text']}"
            + "<|im_end|>"
        )
        text_prompts.append(full_str)
        prefix_lens.append(len(prefix_ids))

    inputs = processor(text=text_prompts, audio=audios, return_tensors="pt", padding=True)
    inputs = {
        k: v.to(device=device, dtype=dtype) if torch.is_floating_point(v) else v.to(device=device)
        for k, v in inputs.items()
    }

    # NOTE: the processor *expands* <|audio_pad|> in input_ids based on the
    # audio length. So input_ids[i] is LONGER than text_prompts[i]'s tokenized
    # length. The prefix in input_ids is at the same logical position, but
    # offset by the audio_pad expansion. We need to find where assistant starts
    # in the expanded input_ids.
    #
    # Cleanest: locate the LAST occurrence of "<|im_end|>\n<|im_start|>" + a
    # specific marker. Easier: tokenize the FULL prefix-string (including the
    # generation prompt) without audio expansion, count its length pre-expansion,
    # then add `extra_audio_tokens = (expanded_length - original_length)` worth
    # of <|audio_pad|> to find the assistant start.
    input_ids = inputs["input_ids"]
    B, L = input_ids.shape
    labels = input_ids.clone()
    labels.fill_(-100)
    for i in range(B):
        # The processor inserted (T_aud - 1) copies of <|audio_pad|> at the
        # position of the original single <|audio_pad|>. So:
        #   expanded prefix_len = original prefix_len + (T_aud - 1)
        # where T_aud = number of audio_pad tokens in this row's input_ids.
        audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        T_aud = (input_ids[i] == audio_pad_id).sum().item()
        expanded_prefix_len = prefix_lens[i] + max(0, T_aud - 1)
        if expanded_prefix_len < L:
            labels[i, expanded_prefix_len:] = input_ids[i, expanded_prefix_len:]
        # Also mask padding tokens at the end
        attn = inputs.get("attention_mask")
        if attn is not None:
            labels[i, attn[i] == 0] = -100
    inputs["labels"] = labels
    return inputs


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/semantic_endpoint/data.pt")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--bsz", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--checkpoint-dir", default="checkpoints/semantic_endpoint")
    # Early stopping
    p.add_argument("--eval-every", type=int, default=0,
                    help="If > 0, run held-out eval every N steps; save best.pt by double-utt accuracy")
    p.add_argument("--eval-holdout-size", type=int, default=60,
                    help="N held-out examples (balanced across schemas) reserved from training")
    p.add_argument("--score-spec", choices=["legacy", "v9"], default="legacy",
                    help="v9: expected marker count per example = END_TOK count in its "
                         "target text (exact match per schema); score = mean of per-schema "
                         "exact rates. Works for any schema set.")
    p.add_argument("--resume", action="store_true",
                    help="resume from the latest step*.pt in --checkpoint-dir "
                         "(trainable weights + optimizer + step + best_score). For "
                         "surviving OOM kills on a shared box via an auto-restart wrapper.")
    p.add_argument("--early-stop-patience", type=int, default=4,
                    help="Stop training if no improvement after N evals")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading data %s…", args.data)
    examples = torch.load(args.data, weights_only=False)
    # Filter out examples whose audio is > 12 s (memory-bound on shared GPU)
    MAX_DUR_S = 12.0
    examples = [e for e in examples if len(e["audio"]) <= int(MAX_DUR_S * 16000)]
    logger.info("Loaded %d examples (filtered to ≤ %.0f s audio)", len(examples), MAX_DUR_S)

    # Reserve a balanced held-out set for early stopping. Pick reproducibly
    # so the train set never sees these.
    holdout_examples: list = []
    if args.eval_every > 0:
        import random as _r
        from collections import defaultdict as _dd
        _rng = _r.Random(12345)
        by_schema: dict[str, list] = _dd(list)
        for e in examples:
            by_schema[e["schema"]].append(e)
        for k in by_schema:
            _rng.shuffle(by_schema[k])
        n_schemas = max(1, len(by_schema))
        per_schema = max(1, args.eval_holdout_size // n_schemas)
        for k, pool in by_schema.items():
            holdout_examples.extend(pool[:per_schema])
        holdout_keys = {id(e) for e in holdout_examples}
        examples = [e for e in examples if id(e) not in holdout_keys]
        counts = {k: sum(1 for e in holdout_examples if e["schema"] == k)
                   for k in by_schema}
        logger.info("Reserved %d holdout examples %s; train set now %d",
                     len(holdout_examples), counts, len(examples))

    logger.info("Loading Qwen3-ASR-0.6B base…")
    model, tokenizer, processor = load_base_model()
    model = model.to(device).bfloat16()

    new_ids, eager_id, end_id = extend_tokenizer_and_model(model, tokenizer, processor)

    apply_lora(model.thinker, rank=args.lora_r, alpha=args.lora_alpha)
    freeze_except_lora_and_new_rows(model, new_ids)

    # Enable gradient checkpointing to fit on a shared GPU
    model.thinker.model.gradient_checkpointing_enable()
    # Make sure use_cache is off during training (checkpointing requires it)
    if hasattr(model.thinker.model.config, "use_cache"):
        model.thinker.model.config.use_cache = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Trainable %.2f M / Total %.2f M", n_trainable, n_total)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )

    rng = np.random.default_rng(0)
    indices = np.arange(len(examples))

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _run_holdout_eval(model, eager_id_, end_id_) -> dict:
        """Greedy-decode the holdout set and compute marker metrics.

        Schema-conditioned metrics:
          - single: emit ≥ 1 end marker
          - double: emit ≥ 2 end markers
          - disfluency: emit exactly 1 end marker (no over-fire)
          - truncated: emit ZERO markers (audio is incomplete)
          - trailing_silence: emit ≥ 1 marker (audio is complete + silence)
          - no_fire: emit ZERO markers (complete utt, no trailing silence)

        Returns dict with per-schema accuracies.
        """
        model.eval()
        if hasattr(model.thinker.model.config, "use_cache"):
            model.thinker.model.config.use_cache = True
        s_hit = d_hit = f_correct = f_overfire = t_correct = t_misfire = 0
        ts_hit = ts_under = 0
        nf_correct = nf_misfire = 0
        s_n = d_n = f_n = t_n = ts_n = nf_n = 0
        # v9 spec: exact / over / under vs the target's own marker count
        v9_counts: dict[str, list[int]] = {}   # schema -> [n, exact, over, under]
        with torch.no_grad():
            for e in holdout_examples:
                audio = np.asarray(e["audio"], dtype=np.float32)
                msgs = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": [{"type": "audio"}]},
                ]
                prompt = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(text=[prompt], audio=[audio], return_tensors="pt", padding=True)
                inputs = {
                    k: v.to(device).bfloat16() if torch.is_floating_point(v) else v.to(device)
                    for k, v in inputs.items()
                }
                try:
                    out = model.generate(**inputs, max_new_tokens=128)
                except Exception as ex:
                    logger.warning("Holdout gen failed: %s", ex)
                    continue
                gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
                n_end = sum(1 for t in gen_ids if t == end_id_)
                sch = e["schema"]
                if args.score_spec == "v9":
                    want = e["text"].count(END_TOK)
                    c = v9_counts.setdefault(sch, [0, 0, 0, 0])
                    c[0] += 1
                    c[1] += int(n_end == want)
                    c[2] += int(n_end > want)
                    c[3] += int(n_end < want)
                    continue
                if sch == "single":
                    s_n += 1
                    if n_end >= 1: s_hit += 1
                elif sch == "double":
                    d_n += 1
                    if n_end >= 2: d_hit += 1
                elif sch == "disfluency":
                    f_n += 1
                    if n_end == 1: f_correct += 1
                    if n_end >= 2: f_overfire += 1
                elif sch == "truncated":
                    t_n += 1
                    if n_end == 0: t_correct += 1
                    if n_end >= 1: t_misfire += 1
                elif sch == "trailing_silence":
                    ts_n += 1
                    if n_end >= 1: ts_hit += 1
                    else: ts_under += 1
                elif sch == "no_fire":
                    nf_n += 1
                    if n_end == 0: nf_correct += 1
                    if n_end >= 1: nf_misfire += 1
        model.train()
        if hasattr(model.thinker.model.config, "use_cache"):
            model.thinker.model.config.use_cache = False
        if args.score_spec == "v9":
            out_m: dict[str, float] = {}
            for sch, (n, exact, over, under) in sorted(v9_counts.items()):
                out_m[f"{sch}_exact"] = exact / max(1, n)
                out_m[f"{sch}_over"] = over / max(1, n)
                out_m[f"{sch}_under"] = under / max(1, n)
            return out_m
        return {
            "single": s_hit / max(1, s_n),
            "double": d_hit / max(1, d_n),
            "disfl_correct": f_correct / max(1, f_n),
            "disfl_overfire": f_overfire / max(1, f_n),
            "trunc_correct": t_correct / max(1, t_n),
            "trunc_misfire": t_misfire / max(1, t_n),
            "trail_correct": ts_hit / max(1, ts_n),
            "trail_underfire": ts_under / max(1, ts_n),
            "no_fire_correct": nf_correct / max(1, nf_n),
            "no_fire_misfire": nf_misfire / max(1, nf_n),
        }

    best_score = -1.0
    evals_since_best = 0
    eval_log = []
    start_step = 0

    # Resume: on a shared box the OOM killer can take the process mid-run
    # (see research notes — global OOM targets the user session). Frequent
    # step*.pt saves + this resume make a kill cost only a few hundred steps.
    if args.resume:
        ckpts = sorted(Path(args.checkpoint_dir).glob("step*.pt"),
                       key=lambda p: int(p.stem[4:]))
        if ckpts:
            latest = ckpts[-1]
            rk = torch.load(latest, weights_only=False, map_location="cpu")
            model.load_state_dict(rk["trainable"], strict=False)
            if rk.get("optimizer") is not None:
                try:
                    optim.load_state_dict(rk["optimizer"])
                except Exception as e:
                    logger.warning("optimizer state not restored (%s); continuing", e)
            start_step = int(rk.get("step", 0))
            best_score = float(rk.get("best_score", -1.0))
            elog = Path(args.checkpoint_dir) / "eval_log.json"
            if elog.exists():
                import json as _json
                eval_log = _json.loads(elog.read_text())
            logger.info("RESUMED from %s at step %d (best_score %.3f)",
                         latest, start_step, best_score)
        else:
            logger.info("--resume set but no step*.pt found; starting fresh")

    def _save_step(step, extra_best=False):
        trainable = {n: p.detach().cpu()
                     for n, p in model.named_parameters() if p.requires_grad}
        payload = {
            "step": step, "trainable": trainable,
            "tokenizer_vocab_size": len(tokenizer),
            "new_ids": new_ids, "eager_id": eager_id, "end_id": end_id,
            "best_score": best_score, "optimizer": optim.state_dict(),
        }
        tmp = Path(args.checkpoint_dir) / f"step{step}.pt.tmp"
        torch.save(payload, tmp)
        tmp.rename(Path(args.checkpoint_dir) / f"step{step}.pt")
        # keep only the latest 2 step*.pt to bound disk (optimizer state is large)
        olds = sorted(Path(args.checkpoint_dir).glob("step*.pt"),
                      key=lambda p: int(p.stem[4:]))[:-2]
        for p in olds:
            p.unlink(missing_ok=True)

    model.train()
    cursor = 0
    for step in range(start_step, args.steps):
        if cursor + args.bsz > len(indices):
            rng.shuffle(indices)
            cursor = 0
        batch_idx = indices[cursor : cursor + args.bsz].tolist()
        cursor += args.bsz
        batch = [examples[i] for i in batch_idx]

        try:
            inputs = build_inputs(processor, batch, device, torch.bfloat16)
        except Exception as e:
            logger.warning("Skipping batch (input build failed): %s", e)
            continue

        lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        for g in optim.param_groups:
            g["lr"] = lr

        t0 = time.perf_counter()
        # Forward goes through the wrapper's thinker
        out = model.thinker(**inputs)
        loss = out.loss
        if loss is None or not torch.isfinite(loss):
            logger.warning("step %d loss=%s — skipping", step, loss)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optim.step()
        optim.zero_grad()
        dt = time.perf_counter() - t0

        if step % args.log_every == 0:
            logger.info("step %4d  loss %.4f  lr %.2e  %.2f s/step  free %.0f MB",
                         step, loss.item(), lr, dt,
                         torch.cuda.mem_get_info()[0] / 1e6)

        if step > 0 and step % args.save_every == 0:
            _save_step(step)
            logger.info("Saved step%d.pt (resumable)", step)

        # Periodic held-out eval + early stopping
        if args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
            t0_eval = time.perf_counter()
            m = _run_holdout_eval(model, eager_id, end_id)
            dt_eval = time.perf_counter() - t0_eval
            if args.score_spec == "v9":
                exact_keys = [k for k in m if k.endswith("_exact")]
                score = sum(m[k] for k in exact_keys) / max(1, len(exact_keys))
                eval_log.append({"step": step, "score": score, **m})
                logger.info("EVAL step %5d  %s  score=%.3f (%.1fs)", step,
                             " ".join(f"{k[:-6]}={m[k]:.2f}" for k in sorted(exact_keys)),
                             score, dt_eval)
                if score > best_score + 1e-6:
                    best_score = score
                    evals_since_best = 0
                    best_path = Path(args.checkpoint_dir) / "best.pt"
                    trainable = {n: p.detach().cpu()
                                 for n, p in model.named_parameters() if p.requires_grad}
                    torch.save({
                        "step": step, "trainable": trainable,
                        "tokenizer_vocab_size": len(tokenizer),
                        "new_ids": new_ids, "eager_id": eager_id, "end_id": end_id,
                    }, best_path)
                    logger.info("New best (v9 score %.3f) -> %s", score, best_path)
                else:
                    evals_since_best += 1
                    if evals_since_best >= args.early_stop_patience:
                        logger.info("Early stop at step %d (no improvement in %d evals)",
                                     step, evals_since_best)
                        break
                import json as _json
                (Path(args.checkpoint_dir) / "eval_log.json").write_text(
                    _json.dumps(eval_log, indent=2))
                continue
            # Composite score: reward turn detection + disfluency correctness +
            # truncated correctness + trailing-silence emission;
            # penalize disfluency over-fire, truncated mis-fire,
            # and trailing-silence under-fire. The trailing-silence schema
            # is added in v5 to fix the streaming P50 latency.
            score = (
                m["double"]
                + 0.5 * m["disfl_correct"]
                + 0.5 * m["trunc_correct"]
                + 0.5 * m["trail_correct"]
                + 0.5 * m["no_fire_correct"]
                - 0.5 * m["disfl_overfire"]
                - 0.5 * m["trunc_misfire"]
                - 0.5 * m["trail_underfire"]
                - 0.5 * m["no_fire_misfire"]
            )
            eval_log.append({"step": step, "score": score, **m})
            logger.info("EVAL step %5d  S=%.2f D=%.2f disfl-✓=%.2f disfl-✗=%.2f "
                         "trunc-✓=%.2f trunc-✗=%.2f trail-✓=%.2f trail-✗=%.2f "
                         "nf-✓=%.2f nf-✗=%.2f score=%.3f (%.1fs)",
                         step, m["single"], m["double"], m["disfl_correct"],
                         m["disfl_overfire"], m["trunc_correct"], m["trunc_misfire"],
                         m["trail_correct"], m["trail_underfire"],
                         m["no_fire_correct"], m["no_fire_misfire"],
                         score, dt_eval)
            if score > best_score + 1e-6:
                best_score = score
                evals_since_best = 0
                best_path = Path(args.checkpoint_dir) / "best.pt"
                trainable = {n: p.detach().cpu()
                             for n, p in model.named_parameters() if p.requires_grad}
                torch.save({
                    "step": step,
                    "trainable": trainable,
                    "tokenizer_vocab_size": len(tokenizer),
                    "new_ids": new_ids,
                    "eager_id": eager_id,
                    "end_id": end_id,
                    "metrics": m,
                    "score": score,
                }, best_path)
                logger.info("  ↳ new BEST: score %.3f saved to %s", score, best_path)
            else:
                evals_since_best += 1
                logger.info("  ↳ no improvement (%d / %d)",
                             evals_since_best, args.early_stop_patience)
                if evals_since_best >= args.early_stop_patience:
                    logger.info("Early stopping at step %d (no improvement in %d evals)",
                                 step, evals_since_best)
                    break

    # Save eval log if we ran any
    if eval_log:
        eval_log_path = Path(args.checkpoint_dir) / "eval_log.json"
        import json as _json
        eval_log_path.write_text(_json.dumps(eval_log, indent=2))
        logger.info("Wrote eval log to %s (best score %.3f)", eval_log_path, best_score)

    # Final save (with the LATEST weights, not necessarily the best)
    ckpt_path = Path(args.checkpoint_dir) / f"step{args.steps}.pt"
    trainable = {n: p.detach().cpu()
                 for n, p in model.named_parameters() if p.requires_grad}
    torch.save({
        "step": args.steps,
        "trainable": trainable,
        "tokenizer_vocab_size": len(tokenizer),
        "new_ids": new_ids,
        "eager_id": eager_id,
        "end_id": end_id,
    }, ckpt_path)
    logger.info("Saved final %s", ckpt_path)


if __name__ == "__main__":
    main()
