# Training data format

[README](../README.md) · [Training guide](../REPRODUCING.md) · [Code map](code-map.md)

## File and fields

Pass `--data path/to/data.pt` to `python -m src.train_semantic_endpoint`.
The file contains a Python `list[dict]` saved with `torch.save`. It stores waveforms,
not audio filenames, JSON records, spectrograms, or precomputed embeddings.

| Field | Type | Required? | Meaning |
| --- | --- | --- | --- |
| `audio` | 1-D NumPy `float32` array | Yes | Mono waveform at **16,000 Hz**; include the silence relevant to the label |
| `text` | `str` | Yes | Transcript with zero or more turn-marker pairs at the appropriate positions |
| `ctx` | `str` | No; defaults to `""` | Context supplied in the system prompt, such as a vocabulary hint |
| `schema` | `str` | When held-out evaluation is enabled | Category used to balance the holdout and report per-category results |
| `source` | `str` | No | Provenance label used by builders, such as `ls`, `ami`, or `custom` |
| `audio_end_s` | `float` | No | Duration in seconds; builders set this to `len(audio) / 16000` |

Builders may add metadata such as `meeting_id`, `gap_s`, or `trunc_frac`.
The trainer reads `audio`, `text`, `ctx`, and (for evaluation) `schema`.
It assumes 16 kHz: adding a `sample_rate` field does not resample the waveform.
It currently **filters out examples longer than 12 seconds**.

## Target text and turn markers

The recipe writes the adjacent pair `<EAGER_END_SPEECH><END_SPEECH>` at each
labeled turn ending. The streaming replay evaluator detects `<END_SPEECH>`.
Despite the two names, these examples teach them together, not as two separately
timed decisions. There is no explicit “hold” token: holding means no turn marker.

In the following examples, `M` abbreviates that pair; do not put a literal `M`
in a training target.

| Example | Audio | `text` target |
| --- | --- | --- |
| Complete, no silence | “Please call me tomorrow.” up to the speech ending | `Please call me tomorrow.` |
| Complete with silence | Same audio followed by observed silence | `Please call me tomorrow. M` |
| Pause within a thought | “Please call” + pause + “me tomorrow.” + final silence | `Please call me tomorrow. M` |
| Two completed turns | “Please call me tomorrow.” + gap + “Thank you.” + final silence | `Please call me tomorrow. M Thank you. M` |
| Silence only | No speech | Empty string |

These are labeling illustrations, not an instruction to synthesize the transcript
from the example wording. Each transcript must match its waveform. Internal
markers belong after the corresponding words. Do not append a marker to every
file, and do not create a marker at a pause simply because later audio reveals
that the speaker never resumed.

The causal builder pairs the same utterance with and without a silence tail.
It also contrasts complete and incomplete phrases, teaches silence-only inputs,
and adds dictation and context examples. For context counterfactuals, changing
the hint must not change what the waveform says: a misleading hint still gets
the audio-grounded transcript. Context must be available before the decision.

## Save a small example file

Assume `completed.wav` contains the sentence below **followed by at least 0.3 s
of silence**, at 16 kHz. This illustrates the storage format; two examples are
not enough to train a useful model.

```python
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

audio, sample_rate = sf.read("completed.wav", dtype="float32")
assert sample_rate == 16000, "Resample audio to 16 kHz before saving."
assert audio.ndim == 1, "Use mono audio."
assert 0 < len(audio) <= 12 * sample_rate
assert np.isfinite(audio).all()

examples = [{
    "audio": audio,
    "text": "Please call me tomorrow. <EAGER_END_SPEECH><END_SPEECH>",
    "ctx": "",
    "schema": "single_sil",
    "source": "custom",
    "audio_end_s": len(audio) / sample_rate,
}]

# An illustrative no-speech example: empty transcript and no markers.
examples.append({
    "audio": np.zeros(sample_rate, dtype=np.float32),
    "text": "",
    "schema": "silence_only",
    "source": "custom",
    "audio_end_s": 1.0,
})
Path("data/custom").mkdir(parents=True, exist_ok=True)
torch.save(examples, "data/custom/data.pt")
```

Use a larger, balanced pool with both marker and no-marker examples for actual
training. Split source recordings or speakers before constructing related
examples so near-duplicates do not cross your training and evaluation boundary.

## What the trainer does with an example

1. Puts `ctx` into a system message and `audio` into the user audio input.
2. Builds the assistant target as `language English<asr_text>{text}<|im_end|>`.
3. Computes next-token loss on the assistant target; prompt and padding tokens
   are masked from the loss.
4. Updates the attention LoRA parameters and new marker-token rows.

Store only the transcript and markers in `text`. Do not add the language prefix,
`<asr_text>`, chat-role delimiters, or `<|im_end|>` yourself. The current wrapper
hard-codes English. The model receives audio via the Qwen processor; you do not
need to build acoustic features manually.

With `--eval-every > 0`, the trainer reserves a schema-balanced holdout. With
`--score-spec v9`, it checks whether the predicted number of end markers matches
the target count, reporting accuracy per schema. This is a training diagnostic;
use continuous streaming replay to measure turn latency and false fires.

Source of truth: [`build_inputs` and the trainer](../src/train_semantic_endpoint.py),
[causal pool builder](../eval/build_v9_training_data.py), and
[dictation/context builder](../eval/build_v10_training_data.py).
