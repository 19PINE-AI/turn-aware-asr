# Turn-Aware Streaming ASR

**Teach a speech recognizer to transcribe and decide when the speaker has finished.**

Code and experiments for **The Trade-off Was in the Labels: Causal Supervision for Turn-Aware Streaming ASR** · Bojie Li and Noah Shi.

[Paper · arXiv:2609.04225](https://arxiv.org/abs/2609.04225) · [PDF](https://arxiv.org/pdf/2609.04225) · [Interactive demos and results](https://01.me/research/turn-aware-asr/) · [Training guide](REPRODUCING.md)

## What does the model learn?

A pause does not always mean a turn is over. Someone dictating a phone number may pause between digits; someone finishing a sentence may expect an immediate response.

We fine-tune **Qwen3-ASR-0.6B** with a small **LoRA adapter** to:

- **Transcribe** the speech heard so far.
- **End a turn** when the words form a complete thought and sufficient silence has been heard.
- **Keep listening** through unfinished thoughts and pauses within dictated numbers.
- **Use context** such as names or vocabulary hints while following what the audio actually says.

Training predicts the next transcript token, including special turn-ending tokens. The audio encoder and original model weights stay frozen; training updates LoRA weights and the new token rows. The unified paper configuration uses LoRA rank 32 and approximately 20,000 constructed examples.

**The labeling rule:** a decision must depend only on audio and context available at that point. Do not label a turn as finished merely because an offline audio file ends there.

## What does the training data look like?

The trainer reads a **PyTorch `.pt` file containing a list of dictionaries**. Each example has a **mono, 16 kHz waveform** and a **target transcript**:

```python
{
    "audio": waveform,          # 1-D numpy float32 array, sampled at 16,000 Hz
    "text": "Please call me tomorrow. <EAGER_END_SPEECH><END_SPEECH>",
    "ctx": "",                  # optional context, placed in the system prompt
    "schema": "single_sil",     # example category, used for held-out evaluation
    "source": "custom",         # provenance metadata
    "audio_end_s": len(waveform) / 16000,
}
```

Here, the waveform must include the completed sentence **and observed trailing silence**. The target marker pair is `<EAGER_END_SPEECH><END_SPEECH>`; **omit it to teach the model to keep listening**. The trainer adds the Qwen chat template and ASR wrapper itself.

| Audio available at the decision point | Target behavior |
| --- | --- |
| Complete sentence, no trailing silence yet | Transcribe; no turn marker |
| Same sentence with observed trailing silence | Transcribe, then emit the marker pair |
| Unfinished phrase followed by a pause | Transcribe; keep listening |
| Speech with a conflicting name in the context | Transcribe the spoken name, not the hint |

The causal endpointing recipe uses at least 0.3 s of observed silence for its positive examples. Silence duration alone does not determine the label. See the [data-format guide](docs/data-format.md) for required fields, a runnable save example, and labeling examples.

## Where should I start?

| I want to… | Start here |
| --- | --- |
| Understand the result or listen to examples | [Interactive website](https://01.me/research/turn-aware-asr/) |
| Train on my own labeled audio | [Setup and training](REPRODUCING.md#train-on-your-own-data), then [data format](docs/data-format.md) |
| Reproduce the paper's experiments | [Paper recipe and evaluation](REPRODUCING.md#reproduce-the-paper) |
| Find the relevant implementation | [Code and experiment map](docs/code-map.md) |
| Edit the paper website | [Website guide](website/README.md) |

**Available now:** training/evaluation code, recorded results, paper source, and website source. **Not bundled:** model checkpoints, dataset audio, generated training pools, or the website's generated audio/data. There is currently no checkpoint download linked from this repository; using a trained model requires your own checkpoint.

## Main results

| Result | Paper finding |
| --- | --- |
| Pure endpointing model | 0.97 boundary recall, 0.39 s median latency, 0.3 false fires per speech-minute; confirmed on a fresh test set |
| Context grounding | Counterfactual examples reduce context intrusion from 40% to 0.8%, retaining most of a +28.9 percentage-point entity-recall benefit |
| Label diagnosis | Adding one second of silence raises a previously “broken” model's turn-ending recall from 0.10 to 1.00 |

The endpointing row describes the **pure endpointing model**, not the broader unified model. These are research prototypes; endpointing evidence covers single-speaker, single-channel English from held-out AMI meetings. See the [paper](https://arxiv.org/abs/2609.04225) for configuration-specific results and limitations.

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
