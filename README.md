# The Trade-off Was in the Labels: Causal Supervision for Turn-Aware Streaming ASR

**Bojie Li (Pine AI) · Noah Shi (University of Washington)**

[Paper: arXiv:2609.04225](https://arxiv.org/abs/2609.04225) · [PDF](https://arxiv.org/pdf/2609.04225) · [Interactive website](https://01.me/research/turn-aware-asr/)

The code, training recipe, and benchmark accompanying the paper. A small LoRA adapter on **Qwen3-ASR-0.6B**, trained in hours on one GPU, adds semantic end-of-turn detection, dictation handling, and context-grounded transcription to streaming ASR.

## Abstract

A voice agent must decide, moment to moment, whether the user has finished; silence rarely settles it: a caller reading a phone number pauses mid-digits, a one-word "Stop!" ends a turn, a long question carries pauses longer than real turn-gaps. A voice-activity detector plus a silence timeout (the deployed default) cannot separate these, because within-turn pauses routinely exceed between-turn gaps; what distinguishes them is whether the words so far form a complete thought: what a recognizer computes to produce a transcript. We present the first open training recipe and benchmark for turn-aware streaming ASR: a small LoRA adapter on Qwen3-ASR-0.6B, trained in hours on one GPU, that transcribes, detects end-of-turn from meaning and silence, handles dictation, and grounds transcription in context. On a deployment-matched benchmark it reaches 0.97 boundary recall at 0.39 s median latency with 0.3 false fires per speech-minute, replicated on a fresh test set; no silence timeout reaches this point. The recipe rests on one principle: every streaming-decision label must be computable from input up to the decision point. Offline corpora violate it, encoding the future; such clairvoyant labels manufactured oscillation and a phantom recall-versus-precision trade-off, exposed when one appended second of silence raised a "broken" model's end-of-turn recall from 0.10 to 1.00. The same leak recurred with context: an always-matching biasing prefix became a copied shortcut (40% intrusion), and counterfactuals disagreeing with the audio cut this to 0.8% while keeping most of a +28.9 pp entity-recall benefit.

## Method and results

Every streaming-decision label must be computable from input available **up to the decision point**. Causal minimal pairs teach when to hold or fire; counterfactual context examples teach the model to use hints while respecting the audio.

- **Endpointing:** the pure endpointing model reaches 0.97 boundary recall, 0.39 s median latency, and 0.3 false fires per speech-minute on the deployment-matched benchmark, with confirmation on a fresh test set. No silence timeout reaches this operating point.
- **Label diagnosis:** appending one second of silence raises a previously “broken” model’s end-of-turn recall from 0.10 to 1.00, exposing future-dependent labels and an artificial recall-versus-precision trade-off.
- **Context grounding:** counterfactual examples reduce context intrusion from 40% to 0.8% while retaining most of a +28.9 percentage-point entity-recall benefit.
- **Unified model:** one adapter combines transcription, turn detection, dictation, and context biasing. Its breadth has a measured precision cost; the endpointing figures above describe the pure endpointing model, not every configuration.

The checkpoints are research prototypes trained on approximately 20,000 synthesized examples. Endpointing evidence is limited to single-speaker, single-channel English from held-out AMI meetings; see the paper for generalization, completeness-heuristic, and unified-model limitations.

## Setup

```bash
git clone https://github.com/19PINE-AI/turn-aware-asr.git
cd turn-aware-asr
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
pip install -r requirements.txt
```

See [REPRODUCING.md](REPRODUCING.md) for environment versions, dataset prerequisites, recipe entry points, and checkpoint availability.

## Repository layout

| Path | Contents |
| --- | --- |
| [`src/`](src/) | Model components and training code, including `train_semantic_endpoint.py` |
| [`eval/`](eval/) | Data construction, streaming replay, metrics, and external-system evaluation |
| [`scripts/`](scripts/) | Training, evaluation, and serving launch scripts |
| [`research/`](research/) | Experiment records and recorded evaluation results |
| [`paper/`](paper/) | Paper source and figures |
| [`website/`](website/) | Interactive paper website and trajectory explorer |
| `data/` | Local datasets and generated training pools (gitignored) |

The experiment scripts document their data and checkpoint paths; datasets and model checkpoints are not included in this Git repository. Website build and data-export instructions are in [`website/README.md`](website/README.md).

## Citation

```bibtex
@misc{li2026tradeofflabels,
  title = {The Trade-off Was in the Labels: Causal Supervision for Turn-Aware Streaming ASR},
  author = {Bojie Li and Noah Shi},
  year = {2026},
  eprint = {2609.04225},
  archivePrefix = {arXiv},
  primaryClass = {eess.AS},
  doi = {10.48550/arXiv.2609.04225},
  url = {https://arxiv.org/abs/2609.04225}
}
```

## License

Code: [Apache 2.0](LICENSE). AuT encoder weights extracted from `Qwen/Qwen3-ASR-0.6B` retain their original Apache 2.0 license.
