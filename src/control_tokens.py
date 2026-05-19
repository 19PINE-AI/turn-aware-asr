"""Control-token vocabulary additions for the streaming VAD+ASR model.

These tokens are added on top of the Qwen3-Omni tokenizer (151,936 entries +
`<|audio_pad|>` family). They are emitted exclusively on the text stream.
"""

from dataclasses import dataclass


CONTROL_TOKENS = [
    "<NO_SPEECH>",
    "<START_SPEECH>",
    "<EAGER_END_SPEECH>",
    "<END_SPEECH>",
    "<PROFILE>",
    "</PROFILE>",
    "<HOTWORDS>",
    "</HOTWORDS>",
    "<HISTORY>",
    "</HISTORY>",
]


# Loss weights applied at training time on the text-stream cross-entropy.
# <NO_SPEECH> dominates the stream; down-weight so it doesn't dilute the
# gradient on content tokens. Control tokens up-weighted so the model learns
# the rare transitions.
LOSS_WEIGHTS = {
    "<NO_SPEECH>": 0.1,
    "<START_SPEECH>": 2.0,
    "<EAGER_END_SPEECH>": 2.0,
    "<END_SPEECH>": 2.0,
    "<PROFILE>": 1.0,
    "</PROFILE>": 1.0,
    "<HOTWORDS>": 1.0,
    "</HOTWORDS>": 1.0,
    "<HISTORY>": 1.0,
    "</HISTORY>": 1.0,
}


@dataclass(frozen=True)
class ControlIds:
    """Filled in once the tokenizer has been extended."""
    no_speech: int
    start_speech: int
    eager_end_speech: int
    end_speech: int
    profile_open: int
    profile_close: int
    hotwords_open: int
    hotwords_close: int
    history_open: int
    history_close: int


def add_control_tokens(tokenizer) -> ControlIds:
    """Extend a HF tokenizer with the control tokens; return their ids.

    Idempotent: safe to call on a tokenizer that already has the tokens.
    """
    added = tokenizer.add_special_tokens({"additional_special_tokens": CONTROL_TOKENS})
    if added == 0:
        # Already present.
        pass
    ids = tokenizer.convert_tokens_to_ids(CONTROL_TOKENS)
    return ControlIds(
        no_speech=ids[0],
        start_speech=ids[1],
        eager_end_speech=ids[2],
        end_speech=ids[3],
        profile_open=ids[4],
        profile_close=ids[5],
        hotwords_open=ids[6],
        hotwords_close=ids[7],
        history_open=ids[8],
        history_close=ids[9],
    )
