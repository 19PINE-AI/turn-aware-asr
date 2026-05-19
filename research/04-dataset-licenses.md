# Dataset License Audit — Streaming VAD+ASR Open-Weights Release

**Audit date:** 2026-05-19. Scope: plan sections 4.1–4.5.

## License table

| Dataset | License | Train (comm.)? | Open weights? | Obligations | Notes |
|---|---|---|---|---|---|
| LibriSpeech | CC BY 4.0 | Yes | Yes | Attribute Panayotov et al. + LibriVox | LibriVox source is public domain |
| MLS (en) | CC BY 4.0 | Yes | Yes | Attribute Pratap et al. + LibriVox | Clean |
| GigaSpeech M+L | Apache-2.0 wrapper; audio Terms of Access = **non-commercial research only**; models "may be eligible" for commercial use but user verifies upstream copyright | Ambiguous | **Risky** — SpeechColab does not own YouTube/podcast/audiobook source | Sign ToA; attribution | No "v3" as of 2026-05; GigaSpeech 2 is a separate low-resource multilingual corpus, not a re-license |
| People's Speech | Mixed CC BY 2.0/2.5/3.0/4.0 + some CC BY-SA | Yes (MLCommons explicit) | Yes for CC-BY portion; SA portion triggers share-alike exposure | Per-clip attribution; SA if any SA clip in train set | HF schema exposes per-row license — filter to CC-BY only |
| Common Voice 17 (en) | CC0 1.0 | Yes | Yes | None required | Cleanest license |
| AMI | CC BY 4.0 | Yes | Yes | Attribute AMI Consortium | Re-licensed by Edinburgh |
| ICSI | CC BY 4.0 | Yes | Yes | Attribute ICSI | Re-licensed alongside AMI |
| CHiME-6 / CHiME-5 source | CC BY-SA 4.0 (re-issued 2024-01-01) | Yes | Yes, **SA may attach to weights** | Attribution + ShareAlike on derivatives | Unresolved: are weights a "derivative work"? Conservative = yes |
| Spotify Podcast Dataset | Spotify research-only; no redistribution; non-commercial | No | No | N/A | **Access discontinued Dec 2023.** Unusable. |
| TED-LIUM v3 | CC BY-NC-ND 3.0 (TED Conferences LLC retains talk copyright) | **No (NC)** | **No (ND)** | Attribution, NC, no derivatives | Hard blocker for training |
| Earnings-22 | CC BY-SA 4.0 (HF Revai/earnings22) | Yes | Yes with SA exposure | Attribute Del Rio et al. / Rev.com; SA | SA ambiguity as above |
| Earnings-21 | CC BY-SA 4.0 (HF Revai/earnings21) | Yes | Yes with SA exposure | Same | Plan uses as eval only |
| Fisher Corpus (LDC2004/2005 S13/T19) | LDC User Agreement (~$7k/part for non-members) | **Conditional** — licensee org's approved research only; commercial = separate LDC license | **No** — non-members may not redistribute/publish | Per-org seat; cite LDC IDs | Incompatible with standard LDC terms |
| Emilia (101k h, non-YODAS) | CC BY-NC 4.0 | **No (NC)** | **No** | Non-commercial | Hard blocker |
| Emilia-YODAS (114k h) | CC BY 4.0 | Yes | Yes | Attribute Amphion + YouTube uploaders | Amphion chose CC BY; upstream YouTube copyright caveat remains |

## BLOCKERS — prevent open-weights release

1. **TED-LIUM v3** — NC-ND. Drop from training; eval-only is fine if eval reporting is research-noncommercial.
2. **Spotify Podcast Dataset** — closed Dec 2023, NC, no redistribution. Drop entirely.
3. **Fisher Corpus** — LDC bars redistribution and restricts use to licensee org. Incompatible with open weights. Defer permanently.
4. **Emilia (non-YODAS)** — CC BY-NC 4.0. Use Emilia-YODAS only.
5. **GigaSpeech M+L** — ToA restricts audio to non-commercial; SpeechColab disclaims model-license responsibility. Soft-blocked. Replace with MLS-en + People's Speech CC-BY + Emilia-YODAS.

## Soft warnings

- **CC BY-SA 4.0 inputs** (People's Speech SA portion, Earnings-21/22, CHiME-6): unresolved whether weights are a "derivative work". Options: (a) restrict SA to <5% of training tokens (de minimis), (b) release weights under CC BY-SA, (c) legal sign-off. This project: keep CHiME-6 and Earnings-22 (small volumes); filter People's Speech to CC-BY-only via the HF per-row license field.
- **YouTube-sourced** (Emilia-YODAS): upstream uploader copyright remains despite the curator relabel. Risk non-zero but accepted by precedent.

## RECOMMENDED CURATED CORPUS (open-weights safe)

| Dataset | License | Hours used |
|---|---|---|
| LibriSpeech | CC BY 4.0 | 1,000 |
| MLS English | CC BY 4.0 | 10,000 (subset of 44.5k) |
| People's Speech (CC-BY-only filter) | CC BY 4.0 | 15,000 (subset of 30k) |
| Common Voice 17 English | CC0 1.0 | 3,000 |
| AMI | CC BY 4.0 | 100 |
| ICSI | CC BY 4.0 | 70 |
| Emilia-YODAS (English curated) | CC BY 4.0 | up to 10,000 |
| **Clean-license total** | | **~39,000 h** |

Optional with SA disclosure (small volume):

| CHiME-6 | CC BY-SA 4.0 | 50 |
| Earnings-22 train | CC BY-SA 4.0 | 119 |

**Excluded from training:** TED-LIUM v3, Spotify Podcasts, Fisher, Emilia non-YODAS, GigaSpeech M+L.

**Eval-only (does not produce a derivative work):** LibriSpeech test-clean/other, TED-LIUM v3 test, Earnings-22 test, AMI eval — fine to keep.

## Model-card attribution string

> Trained on LibriSpeech, MLS (en), the CC-BY portion of MLCommons People's Speech, Common Voice 17, AMI, ICSI, and Emilia-YODAS — CC BY 4.0 except Common Voice (CC0 1.0). Per-dataset citations included. Release weights under Apache-2.0 or CC BY 4.0; both satisfy upstream attribution.

## Action items

1. 4.1: drop GigaSpeech; replace with Emilia-YODAS + expanded MLS-en + CC-BY-only People's Speech.
2. 4.2: drop Spotify and Fisher from training; mark TED-LIUM v3 eval-only.
3. 4.5: TED-LIUM v3 / Earnings-22 test sets stay (eval is not a derivative work).
4. Add a "data licenses" appendix to the model card.
5. If CC-BY-SA inputs (CHiME-6, Earnings-22 train) are kept, get legal sign-off; else drop — cost <0.5% of curated hours.
