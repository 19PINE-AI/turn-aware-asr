"""Streaming simulator (per research/07-eval-harness-spec.md).

Replays audio in 240 ms chunks against a model with a `step()` interface,
recording emitted events with simulated timestamps. Wall-clock per tick is
also recorded but does not affect the simulator's internal clock —
keeping the eval reproducible regardless of model speed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol
import time
import math


TICK_MS = 240
SAMPLE_RATE = 16000
SAMPLES_PER_TICK = TICK_MS * SAMPLE_RATE // 1000  # 3840 samples


class StreamingASRProtocol(Protocol):
    def reset_session(self) -> None: ...
    def ingest_context(self, context: str) -> None: ...
    def step(self, chunk: Any) -> list[str]: ...


@dataclass
class Event:
    sim_time_ms: int
    event_type: str
    payload: Any = None


@dataclass
class TickStat:
    sim_time_ms: int
    wall_ms: float


@dataclass
class StreamResult:
    events: list[Event] = field(default_factory=list)
    tick_stats: list[TickStat] = field(default_factory=list)

    def transcript(self) -> str:
        return " ".join(e.payload for e in self.events if e.event_type == "TEXT")

    def first_end_speech_ms(self) -> int | None:
        for e in self.events:
            if e.event_type == "END":
                return e.sim_time_ms
        return None

    def first_eager_end_ms(self) -> int | None:
        for e in self.events:
            if e.event_type == "EAGER_END":
                return e.sim_time_ms
        return None


def chunk_audio(audio, tick_samples: int = SAMPLES_PER_TICK):
    """Yield fixed-size tick chunks; right-pad the last one with zeros."""
    n = len(audio)
    for i in range(0, n, tick_samples):
        chunk = audio[i : i + tick_samples]
        if len(chunk) < tick_samples:
            import numpy as np
            chunk = np.concatenate([chunk, np.zeros(tick_samples - len(chunk), dtype=chunk.dtype)])
        yield chunk


def run_stream(
    model: StreamingASRProtocol,
    audio,
    context: str = "",
) -> StreamResult:
    result = StreamResult()
    model.reset_session()
    if context:
        model.ingest_context(context)
        result.events.append(Event(0, "CONTEXT_INGESTED", len(context)))

    for i, chunk in enumerate(chunk_audio(audio)):
        sim_time_ms = (i + 1) * TICK_MS
        t0 = time.perf_counter_ns()
        tokens = model.step(chunk)
        wall_ms = (time.perf_counter_ns() - t0) / 1e6
        result.tick_stats.append(TickStat(sim_time_ms, wall_ms))
        for tok in tokens:
            if tok == "<START_SPEECH>":
                result.events.append(Event(sim_time_ms, "START"))
            elif tok == "<END_SPEECH>":
                result.events.append(Event(sim_time_ms, "END"))
            elif tok == "<EAGER_END_SPEECH>":
                result.events.append(Event(sim_time_ms, "EAGER_END"))
            elif tok == "<NO_SPEECH>":
                continue
            else:
                result.events.append(Event(sim_time_ms, "TEXT", tok))

    return result
