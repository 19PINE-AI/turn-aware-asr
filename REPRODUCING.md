# Reproducing the paper

Paper: [arXiv:2609.04225](https://arxiv.org/abs/2609.04225).

## What is available

This repository publishes the training and evaluation code, experiment scripts,
recorded result JSONs, paper source, and interactive website source. It does not
include pretrained adapter checkpoints, dataset audio, generated training pools,
or the website's generated audio/data payload. There is currently no checkpoint
download linked from this repository. Historical notes that call a checkpoint
“released” identify the selected experimental configuration, not a hosted download.

The selected unified configuration is **v18, rank 32, step 9000**. Its selection
record is [`research/123-v18-release-decision.md`](research/123-v18-release-decision.md).
The headline 0.97 recall / 0.39 s / 0.3 false fires per speech-minute result refers
to the **pure causal endpointing** configuration; do not attribute it to v18.
Use the paper and recorded replay results for matched comparisons. Numbered
research notes preserve the experiment history and can describe superseded models.

## Environment

Use the setup commands in [README.md](README.md). The existing research environment
uses Python 3.11, PyTorch 2.8.0, qwen-asr 0.0.6, Transformers 4.57.6, PEFT 0.19.1,
and Accelerate 1.12.0. These are recorded environment versions, not a claim that
all experiments have been rerun from a fresh installation. Training was performed
on a single RTX Pro 6000 Blackwell with 96 GB memory.

Additional tools depend on the experiment:

- Dictation synthesis: `pip install edge-tts`, plus system `ffmpeg`.
- Entity extraction: `pip install anthropic google-generativeai`; the helper reads
  `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` from the environment and caches outputs.
  See [`eval/llm_entities.py`](eval/llm_entities.py) before generating entities.
- External turn detectors use separate environments documented in
  [`eval/external/README.md`](eval/external/README.md).
- Website: Node.js 18 or later, `npm ci` in `website/`, and `ffmpeg` for audio export.

## Data and recipe entry points

Obtain the underlying datasets under their respective licenses. The scripts
expect AMI IHM parquet data at `data/ami/ihm/`, LibriSpeech at
`data/librispeech_raw/LibriSpeech/`, and Free Spoken Digit Dataset recordings at
`data/fsdd/recordings/`. Context-biasing evaluation also uses Earnings-22.
The historical meeting split is expected at
`data/semantic_endpoint_v3/meeting_split.json`; preserve the training/test split
when comparing to recorded results.

| Stage | Entry point | Inputs / output |
| --- | --- | --- |
| Causal endpoint pool | `eval/build_v9_training_data.py` | LibriSpeech and AMI; writes the v9 pool and transcript cache |
| Dictation and context schemas | `eval/build_v10_training_data.py` | v9 pool, FSDD, and synthesized speech; configurable hold duplication and distractor fraction |
| Unified v18 pool | `scripts/build_v18_pool.sh` | Requires the prepared v12 pool and v9 transcript cache; adds ASR replay and counterfactual context |
| Unified training | `scripts/train_v18.sh` | Reads `data/semantic_endpoint_v18/data.pt`; writes rank-32 snapshots |
| Replay evaluation | `eval/streaming_replay_eval.py` | A checkpoint, AMI audio, and meeting split |
| Merge for inference | `scripts/merge_endpoint_lora.py` | A trained snapshot and an output directory |

These are research entry points with intermediate-file dependencies, not a single
fresh-clone reproduction command. Inspect each script's `--help` and defaults
before running it. The v12 pool and cached annotations must be prepared before
running the v18 builder; neither is bundled here.

Once the required audio, split, and trained v18 snapshot are present, run the
100-stretch replay evaluation from the repository root:

```bash
.venv/bin/python -m eval.streaming_replay_eval \
  --checkpoint checkpoints/semantic_endpoint_v18_es/snap_step9000.pt \
  --ami-dir data/ami/ihm \
  --split data/semantic_endpoint_v3/meeting_split.json \
  --energy-gate --n-stretches 100 --seed 0 \
  --out eval/results/v18-replay-100.json
```

Use a new output
path to preserve the recorded paper results. See `scripts/eval_v18_release.sh`
for the development and confirmation-horizon configurations.

## Inspecting results without training

The live [trajectory explorer](https://01.me/research/turn-aware-asr/#explorer)
plays the recorded benchmark audio and shows recorded hypotheses and decisions.
Tracked JSON files under `research/` retain the numeric outputs. For a local copy,
follow [`website/README.md`](website/README.md); a complete explorer requires the
local source audio and data export, not only `npm run build`.
