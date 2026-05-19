"""AuT audio encoder extracted from Qwen3-ASR-0.6B.

Strategy: load the full Qwen3-ASR checkpoint via HF transformers, extract the
``thinker.audio_tower`` submodule, and save it as a standalone module that can
be loaded without pulling the entire ASR pipeline.

The encoder produces 12.5 Hz continuous embeddings at d_model = 1024 (after
the encoder's output projection). Input: 128-dim log-mel spectrogram at
100 Hz frame rate (10 ms hop, 25 ms window, 16 kHz mono audio).

Reference: research/10-qwen3-asr-deepdive.md §1-§2.
"""

from __future__ import annotations
from pathlib import Path
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def extract_aut(
    qwen3_asr_path: str = "Qwen/Qwen3-ASR-0.6B",
    save_to: str | Path | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Pull AuT out of a Qwen3-ASR checkpoint.

    Returns the audio_tower module in eval mode. If save_to is provided, also
    saves the state_dict to that path so future loads can skip pulling the
    full ASR checkpoint.
    """
    from transformers import AutoModelForCausalLM

    logger.info("Loading Qwen3-ASR checkpoint from %s", qwen3_asr_path)
    full = AutoModelForCausalLM.from_pretrained(
        qwen3_asr_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # The actual attribute path depends on the published HF class structure.
    # antirez's MODEL.md notes weights live at thinker.audio_tower.*; verify
    # at first run, log the discovered path.
    candidates = [
        "thinker.audio_tower",
        "audio_tower",
        "model.audio_tower",
        "model.thinker.audio_tower",
    ]
    audio_tower = None
    for path in candidates:
        try:
            obj = full
            for part in path.split("."):
                obj = getattr(obj, part)
            audio_tower = obj
            logger.info("Found AuT at attribute path: %s", path)
            break
        except AttributeError:
            continue
    if audio_tower is None:
        # Fallback: scan for any module with "audio_tower" in its name.
        for name, module in full.named_modules():
            if "audio_tower" in name and "." not in name.replace(".", "", name.count(".") - 1):
                logger.warning("Falling back to module path: %s", name)
                audio_tower = module
                break
    if audio_tower is None:
        raise RuntimeError(
            "Could not locate AuT submodule in the Qwen3-ASR checkpoint. "
            "Inspect `full.named_modules()` and update extract_aut()."
        )

    audio_tower = audio_tower.eval()

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        torch.save(audio_tower.state_dict(), save_to)
        logger.info("Saved AuT state_dict to %s (%.1f MB)",
                    save_to, save_to.stat().st_size / 1e6)

    # Release the rest of the checkpoint
    del full
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return audio_tower


def freeze_aut(aut: nn.Module, unfreeze_top_n_layers: int = 0) -> nn.Module:
    """Freeze the encoder, optionally unfreezing the top N transformer blocks.

    Used for Phase 0 Run A (n=0, fully frozen) vs Run B (n=6, top-6 trainable).
    """
    for p in aut.parameters():
        p.requires_grad = False
    if unfreeze_top_n_layers > 0:
        # Locate the transformer blocks. Standard HF naming uses .layers or .blocks.
        blocks = None
        for attr in ("layers", "blocks", "encoder.layers", "encoder.layer"):
            try:
                obj = aut
                for part in attr.split("."):
                    obj = getattr(obj, part)
                if isinstance(obj, nn.ModuleList):
                    blocks = obj
                    break
            except AttributeError:
                continue
        if blocks is None:
            raise RuntimeError(
                "Could not find transformer blocks for selective unfreezing. "
                "Inspect `aut.named_modules()` and update freeze_aut()."
            )
        for blk in blocks[-unfreeze_top_n_layers:]:
            for p in blk.parameters():
                p.requires_grad = True
    return aut


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--out", default="data/aut_encoder.pt")
    args = p.parse_args()
    extract_aut(args.source, save_to=args.out)
