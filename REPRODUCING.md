# Setup, training, and evaluation

[README](README.md) · [Data format](docs/data-format.md) · [Code map](docs/code-map.md)

There are two ways to use the code: **train with your own labeled audio**, or
**reconstruct the paper's data and experiments**. The first path needs only a
training pool in the documented format. The second also needs the original
corpora, meeting splits, and intermediate data caches.

## Install the training environment

Run from a Linux machine with Python 3.11 and an NVIDIA GPU. The paper used one
RTX Pro 6000 Blackwell with 96 GB memory; that is the experiment hardware, not a
measured minimum requirement.

```bash
git clone https://github.com/19PINE-AI/turn-aware-asr.git
cd turn-aware-asr
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0
python -m pip install -r requirements.txt
```

These commands select CUDA 12.8 wheels; use a PyTorch build compatible with your
GPU and driver. The recorded environment used qwen-asr 0.0.6, Transformers 4.57.6,
PEFT 0.19.1, and Accelerate 1.12.0. This is not a fully locked environment or a
claim that all experiments have been rerun from a fresh installation.
The first model load downloads `Qwen/Qwen3-ASR-0.6B` from Hugging Face.

Check that the command-line entry points load without starting training:

```bash
python -m src.train_semantic_endpoint --help
python -m eval.streaming_replay_eval --help
```

## Train on your own data

First prepare `data/custom/data.pt` using the [data-format guide](docs/data-format.md).
Include both completed turns and examples where the model must keep listening.
The illustrative two-record file in that guide demonstrates serialization only;
use a substantially larger pool for this command.

From the repository root, with the environment activated:

```bash
python -m src.train_semantic_endpoint \
  --data data/custom/data.pt \
  --checkpoint-dir checkpoints/custom \
  --steps 2000 --bsz 2 \
  --lora-r 32 --lora-alpha 64 \
  --no-wd-on-embeddings \
  --score-spec v9 --eval-every 500 --eval-holdout-size 60
```

This is a starting configuration for your data, not the paper's full training
schedule. Adjust batch size to available GPU memory. Ensure the pool contains
enough records per schema for both training and the reserved holdout.

| Option | What it controls |
| --- | --- |
| `--data` | Saved list of training examples |
| `--checkpoint-dir` | Directory for weights, training state, and evaluation logs |
| `--lora-r`, `--lora-alpha` | Adapter rank and scaling; the unified paper model uses 32 and 64 |
| `--no-wd-on-embeddings` | Prevents weight decay from changing the original embedding/head rows |
| `--eval-every`, `--eval-holdout-size` | Frequency and size of the schema-balanced training diagnostic |
| `--score-spec v9` | Scores exact end-marker counts against the target for each schema |
| `--snapshot-evals` | Keeps snapshots at evaluation steps for later comparison |
| `--resume` | Resumes from the latest `step*.pt` checkpoint in the output directory |

The model predicts transcript tokens and turn markers with next-token loss.
The training diagnostic counts markers; it does not measure streaming latency.

### Checkpoints and inference

Training writes `step*.pt` resume checkpoints, `best.pt` when held-out evaluation
improves, and final weights as `step<requested-steps>.pt` at completion. The final
save contains weights and metadata, not optimizer state. With `--snapshot-evals`, it also writes
`snap_stepN.pt`. These are project-specific PyTorch checkpoints containing
trainable tensors and marker metadata, not standalone Hugging Face adapters.
The base Qwen model is still required.

To merge a checkpoint into a model directory:

```bash
python -m scripts.merge_endpoint_lora \
  --checkpoint checkpoints/custom/best.pt \
  --out-dir checkpoints/merged/custom
```

The merge helper infers rank from the saved tensors and assumes **alpha = 2 ×
rank**, as used above and in the paper's selected configuration. If you train
with another scaling, adapt the loader before merging or replaying that model.
For streaming behavior, start with the decoder in
[`eval/streaming_replay_eval.py`](eval/streaming_replay_eval.py).
This repository is a research implementation; it does not provide a microphone
application or a production service quickstart.

## Reproduce the paper

### Choose the configuration

| Configuration | Purpose | Where to look |
| --- | --- | --- |
| Pure causal endpointing (`v9`) | The headline 0.97 recall / 0.39 s / 0.3 false fires per minute result | [Recipe](research/59-v9-recipe-design.md), [results](research/62-v9-results.md) |
| Unified (`v18`, rank 32, step 9000) | Endpointing plus dictation and context grounding | [Selection record](research/123-v18-release-decision.md), [training script](scripts/train_v18.sh) |

Version names are historical experiment identifiers, not package releases.
Numbered research notes record decisions at the time and may describe superseded
configurations. In those notes, “released” means the selected experimental model:
**no pretrained checkpoint download is currently linked from this repository**.

### Prepare the data

Obtain each corpus under its own terms and prepare the paths consumed by the
builders. Raw audio, generated pools, annotation caches, and model checkpoints
are not tracked in Git.

| Input | Expected location | Used for |
| --- | --- | --- |
| AMI IHM parquet files | `data/ami/ihm/` | Conversational training and continuous replay |
| LibriSpeech audio | `data/librispeech_raw/LibriSpeech/` | Transcription examples and replay in the training mix |
| Free Spoken Digit Dataset recordings | `data/fsdd/recordings/` | Dictated digit sequences |
| Historical meeting split | `data/semantic_endpoint_v3/meeting_split.json` | Held-out replay selection |
| Earnings-22 audio and metadata | `data/earnings22/` | Context-biasing evaluation |

The loaders expect the layouts and fields in their source, not arbitrary WAV
folders. Keep source meetings separated between training and test. For exact
comparisons, preserve the experiment's split, seeds, and construction settings.

Some builders need additional tools:

- Dictation synthesis: `pip install edge-tts` and system `ffmpeg`.
- Entity extraction: `pip install anthropic google-generativeai`; the
  [helper](eval/llm_entities.py) reads `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` and
  caches annotations. Uncached extraction sends transcripts to those services.
- External detectors: use their [separate environment instructions](eval/external/README.md).

### Build pools and train

| Stage | Entry point | Dependencies |
| --- | --- | --- |
| Causal endpointing pool | `python -m eval.build_v9_training_data` | LibriSpeech and AMI; produces the v9 pool and transcript cache |
| Dictation/context schemas | `python -m eval.build_v10_training_data` | v9 pool, digit recordings, synthesized speech; options control variants |
| Unified v18 pool | `bash scripts/build_v18_pool.sh` | Prepared v12 pool and v9 transcript cache; entity extraction/cache for context examples |
| Unified training | `bash scripts/train_v18.sh` | `data/semantic_endpoint_v18/data.pt` |

These historical scripts depend on intermediate files; the table is a map of
entry points, not a complete fresh-clone reproduction pipeline. In particular,
the v12 pool and cached annotations must be reconstructed before the v18 build.
Use `--help` on the Python builders and inspect the shell scripts for their exact
paths and settings. The [code map](docs/code-map.md) links the corresponding
implementation and experiment records.

### Evaluate continuous streaming

Once AMI data, the meeting split, and a checkpoint are available:

```bash
python -m eval.streaming_replay_eval \
  --checkpoint checkpoints/semantic_endpoint_v18_es/snap_step9000.pt \
  --ami-dir data/ami/ihm \
  --split data/semantic_endpoint_v3/meeting_split.json \
  --energy-gate --n-stretches 100 --seed 0 \
  --out eval/results/v18-replay-100.json
```

Replace `--checkpoint` to evaluate your own model. Replay feeds continuous audio
in 0.5 s chunks by default, including silence after speech. The output JSON
contains `summary` metrics and `per_stretch` records. Read boundary recall,
latency, and false fires together: high recall alone can hide repeated fires.
The evaluator also records costs such as firing before a speaker resumes.

Use a new output path to preserve the recorded results. The
[paper evaluation script](scripts/eval_v18_release.sh) shows the development and
confirmation-horizon variants. For external comparisons, keep the same audio,
meeting split, chunk size, and seeds.

## Explore results without a GPU

The [live website](https://01.me/research/turn-aware-asr/#explorer) plays recorded
audio alongside model decisions; it does not run inference in the browser.
See the [research index](research/README.md) for result files and the
[website guide](website/README.md) to build a local copy.
