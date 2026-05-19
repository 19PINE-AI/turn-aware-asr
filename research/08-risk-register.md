# Risk Register

Quantified risks for the streaming VAD+ASR project. Each entry: **likelihood** (L/M/H), **impact** (L/M/H), **mitigation already in plan**, **additional mitigation recommended**, and a **trigger** for the mitigation (a measurable signal that the risk is materializing).

Likelihood × impact scoring: H×H = critical; H×M or M×H = high; M×M or H×L or L×H = medium; everything else low. Only critical/high entries below — low-priority risks omitted.

---

## CRITICAL

### R1 — Endpoint latency target (≤400 ms P50) is geometrically inconsistent with the 0.5-1.0 s delay hyperparameter

- **Likelihood:** H (it's arithmetic, not probability)
- **Impact:** H (Phase 3 fails its eval gate)
- **In plan:** none — the inconsistency is silent
- **Additional mitigation:** decoupled endpoint head at 0-delay (acoustic-only) + transcription head at 0.5-1.0 s delay. See `research/06-architecture-review.md` Issue 2. Note: Kyutai STT's two production models ship at 0.5 s and 2.5 s delay; their semantic VAD inherits the delay.
- **Trigger:** Phase 0 sanity model shows endpoint P50 > 600 ms even on clean LibriSpeech.

### R2 — Prefix hallucination not bounded by training

- **Likelihood:** M (current 80/20 split is a guess, not a result)
- **Impact:** H (the central differentiator of the model becomes a liability)
- **In plan:** 20% distractor prefix in Phase 4; ≤3% target.
- **Additional mitigation:** (a) ablate distractor ratios — train three runs at 10%, 20%, 35% in Phase 4 and pick by held-out hallucination rate. (b) Add a confidence-gated copy mechanism: model must "consume" a hotword from prefix only when its acoustic posterior over that span exceeds threshold τ. (c) Per-section distractor differentiation: 20% for `<HOTWORDS>`, 0% for `<HISTORY>`, 5% for `<PROFILE>` — see `research/06-architecture-review.md` Issue 4.
- **Trigger:** `hallucination@distractor` in Phase 4 eval > 5%.

---

## HIGH

### R3 — Disfluent speech triggers premature endpoint

- **Likelihood:** H (every existing acoustic VAD has this failure mode)
- **Impact:** M (lowers UX quality, eval gate is ≤5% false-endpoint)
- **In plan:** synthesize hesitations in Phase 3; DPO refinement in Phase 5.
- **Additional mitigation:** Training mix must include real disfluent speech, not only synthesized. Add **Switchboard** (LDC paid) or **TalkBank/CallHome** as a Phase 3 corpus addition. Synthesizing disfluencies via TTS produces artifacts (clipped formants, unrealistic pause durations) the model overfits to.
- **Trigger:** false-endpoint rate on `disfluency-v1` synthetic set ≤5% but on AMI real-data subset >10%. Gap > 5 pp signals the synthetic data is leaking.

### R4 — Mimi tokenization not ASR-optimal

- **Likelihood:** M
- **Impact:** H (would force re-running Phase 1 with a different audio frontend, costing 13+ GPU-days)
- **In plan:** flagged as Phase 0 open question.
- **Additional mitigation:** run **two** Phase 0 sanity models in parallel: 0.6B + Mimi-32-codebooks, and 0.6B + frozen-Whisper-small encoder. Same data (200 h LibriSpeech-clean), same steps. Compare LibriSpeech-clean WER at the smoke-test point. ~6 GPU-h × 2 = 12 GPU-h total. Worth burning to avoid committing to the wrong encoder. Possibly add a third: Mimi-8-codebooks (faster, less compute, may sacrifice accuracy).
- **Trigger:** the better of the two beats the other by >2% absolute WER. If they're within 1%, default to Mimi (consistency with Kyutai's published recipe).

### R5 — Single-Blackwell training run halts on hardware fault

- **Likelihood:** M (consumer-grade card, no ECC redundancy by default; long jobs see SDC events)
- **Impact:** H (loses days of compute)
- **In plan:** none.
- **Additional mitigation:** checkpoint **every 1000 steps** to local SSD + every 5000 steps to a cloud bucket (S3/GCS); validate the last checkpoint by reloading and running one eval batch before deleting prior. Add a watchdog that restarts the process from latest valid checkpoint on segfault/CUDA error. ~5% wall-clock overhead, ~$0 cloud cost given checkpoint size (~10 GB).
- **Trigger:** any unexplained loss spike or NaN. Loss spike → roll back to prior checkpoint, halve LR for the affected window.

### R6 — Spotify Podcast Dataset access has been pulled

- **Likelihood:** H (verified separately by dataset license audit agent — pending)
- **Impact:** M (5k hours of conversational data, but substitutes exist)
- **In plan:** listed as 5,000+ hours of conversational training data in §4.2.
- **Additional mitigation:** assume not available; replace with podcast subset of People's Speech (already CC-BY-SA), MLS-en podcasts, and GigaSpeech podcast portion (~3k h overlap, distinct provenance).
- **Trigger:** N/A — already a near-certainty per industry reports.

---

## MEDIUM

### R7 — Qwen3-0.6B init advantage is small

- **Likelihood:** M
- **Impact:** M (would extend Phase 1 by ~30% but not break the project)
- **In plan:** fallback to train-from-scratch listed.
- **Additional mitigation:** measure loss at step 1000 and step 5000 vs a baseline run from randomly-initialized weights. If gap < 0.05 nats, init is providing little value — possibly worth using a different small LLM init (Pythia-410M, OLMo-1B) that is closer to ASR-aligned in training distribution.
- **Trigger:** Phase 1 LibriSpeech-clean WER at end of epoch 1 > 15%. Indicates init is not helping more than from-scratch would.

### R8 — Mimi codec is rate-limited at 12.5 Hz vs. UX expectations

- **Likelihood:** L (Kyutai ships this in production; 12.5 Hz is sufficient)
- **Impact:** L (only matters if Bo wants sub-200 ms perceived latency, which he doesn't)
- **In plan:** open question on 25 Hz custom codec.
- **Additional mitigation:** none — at 240 ms tick (3 Mimi frames), the audio-side latency floor is already 240 ms regardless of codec rate.

### R9 — DPO preferences for endpoint timing are noisy

- **Likelihood:** M (humans disagree on ±200 ms boundaries at >40% rate)
- **Impact:** M (Phase 5 may not improve over Phase 3)
- **In plan:** none — preferences vaguely specified.
- **Additional mitigation:** operationalize preferences against forced-alignment ground truth; avoid LLM-judge. See `research/06-architecture-review.md` Issue 5.
- **Trigger:** Phase 5 DPO does not improve endpoint P50 latency by ≥10% relative over Phase 4.

---

## Summary

Total critical risks: 2. Total high risks: 4. Total medium risks: 3.

The two critical risks (R1, R2) are *architectural*, not training-time. Both can and should be mitigated **before Phase 1 starts**. R1 in particular changes the model's head structure; redesigning it after Phase 1 means re-running Phase 1.

The four high risks (R3-R6) are each addressable inside their respective phase, but only if measured. Adding the listed triggers to the eval harness JSON output is sufficient — they're all already-computed metrics, just newly emphasized as decision criteria.
