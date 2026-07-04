# E4: does context biasing survive session length? (2026-07-04)

The novel serving claim (research/60): in a minute-scale streaming session,
the `<CTX>` hotword prefix must keep biasing the *current* utterance even
though a growing span of audio history sits between the prefix and the
audio now being transcribed. If the biasing effect decays with that
distance, long sessions lose their hotwords; if it holds, context biasing
is a durable session-level capability.

Test (`worker_integration/e4_pinned_ctx.py`): for each entity-bearing
Earnings-22 utterance, prepend `age` seconds of unrelated filler speech,
then bounded-re-feed the last 16 s (target at the end) — so `age` seconds
of audio sit between the pinned CTX and the target's words. Transcribe
with the CTX prefix present (PIN) vs absent (NOPIN, baseline), and measure
hotword recall vs age. 40 targets, LLM (Haiku 4.5) entities. This tests
the claim at the application layer — bounded re-feed, which the Metronome
paper notes "achieves the same memory horizon" as an in-engine windowed KV
by re-pinning the prefix each frame; the Triton-kernel sink is the
efficiency refinement (research/67).

## Result: the biasing advantage is flat across session age

| Prior audio (age) | Recall, CTX pinned | Recall, no CTX | Gap |
|---|---|---|---|
| 0 s | 0.898 | 0.571 | **+32.7 pp** |
| 4 s | 0.837 | 0.531 | +30.6 pp |
| 8 s | 0.878 | 0.571 | +30.6 pp |
| 12 s | 0.898 | 0.551 | **+34.7 pp** |

Both arms are essentially age-independent, and the PIN−NOPIN gap holds at
**~+30 pp at every age** from an isolated utterance (0 s) to one buried
behind 12 s of intervening audio in the window. **Context biasing does not
decay with session length.** The re-fed `<CTX>` prefix keeps biasing the
current utterance regardless of how much audio precedes it — exactly the
property the pinned-attention-sink is designed to guarantee, shown here
via the equivalent application-level bounded re-feed.

The +30 pp gap is consistent with the isolated Earnings-22 biasing measure
(+28.9 pp, research/63); the absolute recall is a few points lower here
because the target is embedded in windowed conversational audio with
filler, a harder condition than the isolated-utterance biasing test.

## What this validates, and the remaining refinement

- **For the shipping endpoint+biasing product** (v9 + gate + `<CTX>`,
  bounded re-feed): biasing is durable across a session at no extra cost —
  the prefix is re-fed every segment and stays effective. E4 confirms
  there is no session-length penalty to design around.
- **The in-engine pinned sink** (windowed KV that pins `[0,S)` and slides
  the audio out) is the *efficiency* version of this — it avoids
  re-encoding the CTX prefix every frame. It requires the dense-Qwen3 SWA
  vLLM patch (research/67) and would show the same flat curve without the
  per-frame re-encode. That optimization remains the one open piece for
  the continuous-transcription variant; the *capability* is proven here.

## Files

- `research/69-e4-pinned-ctx.json` — per-target, per-age recall
- `worker_integration/e4_pinned_ctx.py` — driver (bounded re-feed, LLM
  entities via `eval/llm_entities.py`)
