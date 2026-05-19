# Evaluation Harness Spec (Phase 0 deliverable)

The plan (§6.2) calls for a streaming eval harness in Phase 0 but doesn't spec it. This is that spec. The harness is on the critical path: every phase's go/no-go gate depends on it.

---

## Design principles

1. **One JSON per run.** All metrics in a single file, schema-versioned. Re-runs append, never overwrite.
2. **Replay-based, not wall-clock-based.** Audio is read from disk and fed in 240 ms chunks; "real time" is simulated against an injected clock. This makes evals reproducible on any hardware and decouples eval speed from model speed.
3. **Reference timing is forced-alignment, not human label.** Human turn-boundary labels disagree at >40% IAA for ±200 ms windows; force-aligned transcript boundaries are the operational ground truth.
4. **Each eval set is a fixed directory tree.** No on-the-fly filtering. If the set changes, it gets a new version tag.

---

## Eval set inventory (fixed in Phase 0)

| Tag | Source | Hours | Purpose |
|---|---|---|---|
| `librispeech-clean-v1` | LibriSpeech test-clean | 5.4 | baseline WER |
| `librispeech-other-v1` | LibriSpeech test-other | 5.3 | noisy WER |
| `tedlium-v3-test-v1` | TED-LIUM v3 test | 2.6 | proper-noun WER |
| `earnings22-test-v1` | Earnings-22 holdout subset | ~5 | domain WER + context biasing |
| `ami-eval-v1` | AMI dev+eval, segmented | ~12 | endpoint latency on real conversation |
| `disfluency-v1` | **synthesized custom** | ~2 | false-endpoint rate |
| `hotword-stress-v1` | **synthesized custom** | ~3 | context biasing recall + hallucination |

Synthesized sets are generated **once**, hashed, and treated as fixed thereafter. Generation script lives in `eval/gen/`.

### `disfluency-v1` construction

200 clips, each 5–30 s. Each contains exactly one disfluency injected at a controlled point:
- 40 clips with mid-sentence "uhh" / "umm" / breathing (300-2000 ms)
- 40 clips with self-correction ("I was going to — actually, ...")
- 40 clips with mid-sentence silent pauses (400-1200 ms)
- 40 clips with throat clear / cough
- 40 clips with mid-sentence "let me think" filler phrase

For each clip, the *true* endpoint is the end of the actual utterance. Any `<END_SPEECH>` emitted before that point is a false endpoint.

### `hotword-stress-v1` construction

300 clips from TED-LIUM and Earnings-22 with at least one proper noun or domain term. Each clip is run **three times**:
1. With relevant context prefix (target entities present).
2. With no context prefix.
3. With distractor prefix (entities from a different clip).

Metrics:
- **Recall@relevant**: entity-level recall with run 1.
- **Recall@none**: recall without context (baseline ceiling).
- **Hallucination@distractor**: rate at which run 3 inserts distractor entities not present in audio.

---

## Streaming simulator

```python
class StreamingSimulator:
    def __init__(self, model, tick_ms=240):
        self.model = model
        self.tick_ms = tick_ms

    def run(self, audio: np.ndarray, sample_rate: int, context: str = ""):
        events = []  # list of (sim_time_ms, event_type, payload)
        chunks = self._chunk(audio, self.tick_ms, sample_rate)

        self.model.reset_session()
        if context:
            self.model.ingest_context(context)
            events.append((0, "CONTEXT_INGESTED", len(context)))

        for i, chunk in enumerate(chunks):
            sim_time_ms = (i + 1) * self.tick_ms
            t0 = time.perf_counter_ns()
            tokens = self.model.step(chunk)
            wall_ms = (time.perf_counter_ns() - t0) / 1e6
            for tok in tokens:
                if tok == "<START_SPEECH>":
                    events.append((sim_time_ms, "START", None))
                elif tok == "<END_SPEECH>":
                    events.append((sim_time_ms, "END", None))
                elif tok != "<NO_SPEECH>":
                    events.append((sim_time_ms, "TEXT", tok))
            events.append((sim_time_ms, "TICK_WALL_MS", wall_ms))

        return events
```

Wall-clock per tick is recorded but does not affect simulator timing. This means the harness runs faster than real-time on any hardware; eval throughput is gated only by the model's batch decode speed.

---

## Metrics

### WER

Standard whisper-norm + jiwer. Compute three flavors per dataset:
- **Final WER** — text accumulated until each `<END_SPEECH>`, scored against reference.
- **Streaming WER@1.0s** — text available 1.0 s before final, scored against final-aligned reference (measures partial accuracy).
- **Per-entity recall/hallucination** — token-level NER pass over both hypothesis and reference, compute recall (hyp ∩ ref) / ref and hallucination (hyp \ ref) / hyp for each entity type.

### Endpoint latency

For each utterance:
- `endpoint_event_ms` = sim time of the model's `<END_SPEECH>`.
- `endpoint_truth_ms` = sim time of end-of-last-word in forced alignment.
- `latency_ms = endpoint_event_ms - endpoint_truth_ms` (can be negative for premature firing).

Report P50, P95, and the false-endpoint rate:
- **Premature endpoint**: `latency_ms < -100` (fired more than 100 ms before truth).
- **Late endpoint**: `latency_ms > 1500` (fired more than 1.5 s after truth).

### Per-tick wall time

Distribution of `TICK_WALL_MS` events. Gate: P95 ≤ 25 ms.

### Memory per session

`torch.cuda.max_memory_allocated()` after warmup, measured per-session in a 16-session batched run. Gate: ≤ 500 MB per session.

---

## Output schema

`eval/results/{commit_sha}_{timestamp}.json`:

```json
{
  "schema_version": 1,
  "commit": "abc123",
  "model": "qwen3-0.6b-streaming-phase3",
  "hardware": "rtx-pro-6000-blackwell",
  "datasets": {
    "librispeech-clean-v1": {
      "wer_final": 7.4,
      "wer_streaming_1s": 8.1,
      "per_entity_recall": {"PERSON": 0.92, "ORG": 0.81, ...},
      "endpoint_latency_p50_ms": 380,
      "endpoint_latency_p95_ms": 720,
      "false_endpoint_premature_rate": 0.03,
      "false_endpoint_late_rate": 0.01,
      "tick_wall_ms_p50": 14.2,
      "tick_wall_ms_p95": 22.7,
      "memory_per_session_mb": 410,
      "n_utterances": 2620
    },
    "disfluency-v1": { ... },
    ...
  }
}
```

---

## Gate enforcement

Each phase's go/no-go gate is encoded as a JSON predicate over the schema. Example for Phase 3:

```json
{
  "phase": 3,
  "predicates": [
    {"dataset": "ami-eval-v1", "metric": "endpoint_latency_p50_ms", "op": "<=", "value": 400},
    {"dataset": "disfluency-v1", "metric": "false_endpoint_premature_rate", "op": "<=", "value": 0.05},
    {"dataset": "librispeech-clean-v1", "metric": "wer_final", "op": "<=", "value": 10.0}
  ]
}
```

Run with `python eval/gate.py --phase 3 --results eval/results/latest.json`. CI-style: exits 0 / 1.

---

## Build order in Phase 0

1. Day 1: simulator scaffold + LibriSpeech-clean WER metric. Validate against published Whisper baselines.
2. Day 2: forced alignment pipeline (WhisperX) for endpoint truths on AMI.
3. Day 3: endpoint metric + disfluency-v1 generation.
4. Day 4: hotword-stress-v1 generation + per-entity metric.
5. Day 5: gate predicate runner + dashboard plot (P50/P95/WER over time).

5 days total; one engineer. Critical that this is done **before** Phase 1 starts.
