# Code and experiment map

[README](../README.md) · [Training guide](../REPRODUCING.md) · [Data format](data-format.md)

The repository retains earlier experiments alongside the paper's final approach.
For the published method, begin with the semantic-endpoint trainer below.
Files named `phase0`, `phase1`, and `endpoint_head` belong to earlier approaches;
they are not prerequisites for training the LoRA model.

## Follow one training example

| Step | File | What to read |
| --- | --- | --- |
| Construct audio/text pairs | [Causal data builder](../eval/build_v9_training_data.py) | `ex`, `is_complete`, and the schema construction in `main` |
| Add dictation/context examples | [Schema builder](../eval/build_v10_training_data.py) | Digit examples, context prompts, and counterfactual options |
| Add ordinary ASR examples | [ASR replay builder](../eval/build_v15_asr_replay.py) | How unmarked transcript examples enter the unified pool |
| Add conflicting context | [Natural biasing builder](../eval/build_natural_biasing.py) | Relevant and distractor entity hints |
| Train the model | [Semantic-endpoint trainer](../src/train_semantic_endpoint.py) | `build_inputs`, `apply_lora`, `freeze_except_lora_and_new_rows`, and `main` |
| Replay continuous audio | [Streaming evaluator](../eval/streaming_replay_eval.py) | `build_stretches`, `StreamDecoder`, `score_fires`, and `aggregate` |
| Export merged weights | [Merge helper](../scripts/merge_endpoint_lora.py) | Loading trainable tensors and merging LoRA into the base model |

The stored training format is shared across pool versions. Names such as `v9`,
`v12`, and `v18` distinguish experiment compositions. They do not denote different
file formats. Read the [field reference](data-format.md) before making custom data.

## Find a directory

| Directory | Purpose |
| --- | --- |
| [`src/`](../src/) | Models and trainers; start with `train_semantic_endpoint.py` |
| [`eval/`](../eval/) | Data builders, inference/evaluation, metrics, and probes |
| [`eval/external/`](../eval/external/README.md) | External turn detectors using the shared replay protocol |
| [`scripts/`](../scripts/) | Exact experiment launch commands and checkpoint export |
| [`research/`](../research/README.md) | Historical notes, recorded JSON results, and logs |
| [`paper/`](../paper/) | Published paper's LaTeX source and figures; start with `main.tex` |
| [`website/`](../website/README.md) | React site, interactive figures, and recorded trajectory explorer |
| `data/`, `checkpoints/` | Local inputs and generated artifacts, excluded from Git |

For the scientific narrative, read the [paper](https://arxiv.org/abs/2609.04225).
For a practical workflow, read the [training guide](../REPRODUCING.md).
For experiment history, use the [research index](../research/README.md).
