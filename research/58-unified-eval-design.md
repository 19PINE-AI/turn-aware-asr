# Unified streaming-replay endpoint eval — design (2026-07-04)

Replaces both `ami_conversational_eval.py` (offline marker counting) and
`streaming_latency.py` (synthetic LS, from-scratch re-decode) as the
production metric. Motivated by two label/eval contradictions found in the
v1–v8 record:

**Contradiction 1 — trailing silence.** The offline AMI eval cuts clips at
the forced-alignment end of speech (~50–200 ms tail) and demands a fire;
the streaming eval demands NO fire on any prefix that ends in speech
without silence. Same acoustic condition, opposite required outputs. In
deployment the condition "audio ends with no trailing silence" never
occurs — silence keeps arriving after the speaker stops. The offline
protocol tests a capability no deployment needs, and v5–v8 were trained
and scored against both labels at once (v8's `no_fire` pool is complete
AMI utterances; its fire pool is other complete AMI utterances — 50/50
label noise, hence the fire-mode/no-fire-mode oscillation).

**Contradiction 2 — non-causal disfluency labels.** Measured on the v5
eval set: "disfluency" examples (must not fire at the internal same-speaker
pause) have median gap 1.39 s (88 % ≥ 1 s); "double" examples (must fire at
the internal gap) have median gap 0.83 s. The distributions overlap almost
completely. At streaming decision time — mid-pause — the two cases are
distinguishable only by who speaks next, i.e. by future audio. A causal
(streaming) policy cannot satisfy both label sets; an offline decoder can,
because it sees the whole clip. Part of the historical "disfluency
over-fire" penalty demanded clairvoyance.

## Design principle: causal labelability

Every ground-truth label must be decidable from audio up to (shortly after)
the label point. Labels that depend on later audio are removed from the
target and moved into *descriptive* metrics (reported, not penalized as
model error).

## Test material

Continuous single-channel stretches from held-out AMI meetings
(md5(meeting_id) % 3 == 2 — same rule as training split).

Primary stream type: **one speaker's ihm channel, continuous timeline**
(voice-assistant / dictation shape: one mic = one user). For a chosen
speaker and a 30–60 s window of the meeting timeline: place that speaker's
utterance clips at their true offsets; fill everything between with
silence. Keep the meeting's full multi-speaker segmentation as *metadata*
for boundary classification (below).

Stretch selection: 25 stretches × 30–60 s from ≥ 10 distinct eval meetings,
each containing ≥ 2 of the target speaker's utterances and ≥ 1
same-speaker quick resume if available (so continuation cases are
represented). A secondary mixed-channel variant (all speakers summed —
meeting-transcription shape) is out of scope for the first cut.

## Ground truth: turn-final vs continuation boundaries

For each utterance end `e` of the target speaker, with `g` = gap until that
speaker's next utterance on the timeline:

- **Turn-final boundary** (model SHOULD fire): `g ≥ T_turn` (default 2.0 s),
  or a different speaker's utterance begins before the speaker resumes.
  Deployment meaning: the floor passed or the user stopped; an endpoint is
  correct and required.
- **Continuation point** (fire is acceptable-but-costly, measured
  descriptively): `0.3 s ≤ g < T_turn` and same speaker resumes with no
  interleaving speaker. Causally ambiguous at decision time. A fire here
  segments one thought into two utterances — count it in `resume_after_fire`
  rate, NOT as a false fire.
- **Internal pause** (model must NOT fire): any pause fully inside an
  utterance (between forced-aligned words), and the region < 0.3 s after
  any utterance end. Firing here is a **false fire** — it truncates speech.

This replaces the impossible "never fire at a same-speaker pause of any
length" target with: never fire while speech is ongoing or within 300 ms of
it; always fire once a turn-final pause is established; quick-resume
segmentation is a cost knob, not an error.

## Protocol

Streaming replay at chunk size 0.5 s (same as historical evals; later also
0.25 s on the vLLM path):

1. Feed chunks in order. At each chunk boundary run one incremental decode.
2. **Committed-prefix decoding** (matches qwen-asr's official vLLM
   streaming protocol and deployment semantics): prompt = chat prefix +
   committed transcript with last K=5 tokens rolled back; model continues.
   A `--from-scratch` flag preserves the v3–v8 protocol for comparability.
3. Fire event: first `<END_SPEECH>` in the current decode. Timestamp =
   audio time of the current chunk end (audio-clock latency, hardware
   independent). Wall-clock compute per chunk recorded separately.
4. After a fire: flush the segment (transcript up to the marker), reset
   committed prefix, continue the stream. Models utterance-by-utterance
   consumption; enables multiple fires per stream.

## Metrics (one JSON per run)

| Metric | Definition | Target |
|---|---|---|
| Boundary recall | fires within [b − 0.25 s, b + 1.5 s] of turn-final boundary b | ≥ 90 % |
| False-fire rate | fires in internal-pause/ongoing-speech regions, per minute of speech | ≤ 1/min |
| Latency P50 / P95 | fire_time − b over recalled boundaries | ≤ 0.5 s / ≤ 1.2 s |
| resume_after_fire | fires at continuation points / # continuation points | report (knob) |
| Streaming WER | concatenated flushed transcripts vs reference words of the stream | ≤ offline WER + 1 pp |
| Chunk compute | median / P95 wall ms per decode step | report |

Recall tolerance rationale: −0.25 s allows semantic-eager fire on utterance
tails; +1.5 s = 0.3 s silence-confirm + 0.5 s chunk quantization + decode
budget, beyond which a voice assistant feels stuck.

## Why this cannot recreate the old contradiction

Every stream contains real post-speech silence (the timeline continues), so
"fire at end-of-speech" and "wait for silence" are the same behavior here.
Offline single-shot decoding on silence-free clips no longer appears
anywhere in training or eval.

## Files

- `eval/streaming_replay_eval.py` (to be written) — builder + replay driver
- Applies to: v3/v5/v8 (re-baselined), endpoint head v2 (frame path), v9,
  and the composed policy (LM ∧ head).
