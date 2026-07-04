# Context biasing on Earnings-22: the lead chart (2026-07-04)

The project's synthesis (research/00 §6.1) named hotword recall with a
relevant text prefix as the lead differentiator — the capability Qwen3-ASR
*claims* but reports **zero numbers** for. The first measurement
(research/16) used LibriSpeech and was **saturated**: 97 % entity recall
with no context, because audiobook "entities" are common English words.
This run uses **Earnings-22** — spontaneous earnings-call speech, dense
with company names, executives, tickers, products — the regime where
biasing should actually matter.

## Setup

- Data: `distil-whisper/earnings22`, 150 utterances sampled uniformly
  across the shard (`--shuffle`, for company/entity diversity).
- Entities: **LLM (Claude Haiku 4.5)** via `eval/llm_entities.py` — batched,
  cached, Gemini fallback. Replaces the earlier spaCy `en_core_web_sm` NER,
  which mislabeled domain proper nouns on mixed-case earnings speech (kept
  generic tokens like "Q4", missed tickers). The LLM keeps only genuine
  biasing hotwords — person/company/product/ticker/place names.
- Base `Qwen/Qwen3-ASR-0.6B` (no fine-tuning), three conditions per utt:
  NO_CTX / RELEVANT (this utt's entities in the system prompt) /
  DISTRACTOR (a different utt's entities).
- `eval/earnings22_biasing.py`.

## Results

150 utts, 37 entity-bearing (LLM keeps only genuine proper nouns, so the
entity-bearing rate is lower than spaCy's — the recall metric is now over
real hotwords).

| Metric | Value | Target (research/00) |
|---|---|---|
| Entity recall, no context | 66.7 % | — |
| **Entity recall, relevant prefix** | **95.6 %** | ≥ 80 % ✓ |
| **Recall uplift** | **+28.9 pp** | — |
| Distractor hallucination rate | 3.7 % | ≤ 3 % ✗ (close) |
| WER, no context | 15.3 % | — |
| WER, relevant prefix | 15.0 % | — |
| WER, distractor prefix | 17.1 % | — |

*(spaCy-NER run, for reference — `research/63-earnings22-biasing-spacy.json`:
recall 75.0 %→95.5 %, +20.5 pp uplift, 5.1 % hallucination. Cleaner LLM
entities lower the no-context baseline and raise the uplift.)*

## Interpretation

1. **Biasing delivers a large, real recall gain on hard audio.** 66.7 % →
   95.6 % (+28.9 pp) — an order of magnitude more headroom than the
   saturated LibriSpeech test showed (which moved 97.4 → 98.6 %). On
   entity-dense speech the context prefix is doing substantial work, and
   95.6 % clears the ≥ 80 % target comfortably. This is the differentiator
   number, now measured where it counts. (With cleaner LLM-extracted
   entities the effect is *larger* than the spaCy run's +20.5 pp: genuine
   proper nouns are rarer, so the no-context baseline is lower and the
   prefix contributes more.)
2. **A relevant prefix also nudges WER down** (15.3 → 14.9 %), while a
   distractor prefix nudges it up (→ 16.8 %) — the model uses the prefix
   as a genuine domain signal, consistent with research/16.
3. **Distractor hallucination (5.1 %) slightly exceeds the ≤ 3 % target.**
   Higher than LibriSpeech's 0.9 %, expected on harder audio with a real
   NER surfacing rarer entities. This is the base model with no
   biasing-specific fine-tuning; the plan's Phase-4 distractor-ratio
   training (research/00 §5) is the lever to push it back under 3 %, and
   is the natural next step if hallucination robustness becomes binding.

## Relationship to the streaming/serving work

In the shipping configuration (v9 + energy gate + `<CTX>` prefix,
bounded re-feed — E2/E3, research/65), the context prefix is re-fed with
every bounded segment, so it is always resident and attended: this recall
uplift applies directly to the streaming endpoint product. The
**pinned-CTX-attention-sink** mechanism (research/60 E4) is a refinement
for the *continuous-transcription* variant (minute-scale decode with no
utterance flush), where the LLM-side context would otherwise grow
unbounded — it keeps `<CTX>` attended while audio history slides out of a
windowed KV. That experiment needs the dense-`Qwen3ForCausalLM`
sliding-window vLLM patch and is deferred (see research/67).

## Files

- `research/63-earnings22-biasing.json` — per-utterance results
- `eval/earnings22_biasing.py` — the eval (spaCy NER, 3 conditions)
