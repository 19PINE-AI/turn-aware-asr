# Dictation generalization: zero-shot transfer to untrained enumeration shapes

**Date:** 2026-07-06. **Model:** release v15@6000 (`unified_release.pt`), trained on
10-digit phone numbers (3-3-4) ONLY. **Harness:** eval/dictation_probe_eval.py, same
streaming stack. Probes: eval/build_generalization_probes.py (FSDD held-out george/lucas
for digits; edge-tts held-out voices for addresses). Delta-based scoring (per research/85):
on-time = a fire in [last_end-0.30, last_end+1.75]; genuine-mid = a fire < last_end-0.30.

| shape | n | reported recall | δ on-time recall | genuine mid-seq prem | median earliest fire | digit acc |
|---|---|---|---|---|---|---|
| address (3-4 house digits + apt, in words) | 60 | 0.933 | **1.000** | 0.150 | +0.44 s | 0.997 |
| phone13 (13-digit, 3-3-3-4) | 60 | 0.117 | 0.117 | 0.200 | -1.23 s | 1.000 |
| card (16-digit, 4-4-4-4) | 100 | 0.000 | **0.000** | 1.000 | **-3.04 s** | 0.996 |

Zero late fires in all three (consistent with research/85; the delta method is not just
inflating recall — card stays at 0).

## Finding
Transfer is **bounded by the trained pattern length**, not by digits-vs-words:
- **Short enumerations transfer.** Street addresses (3-4 house digits embedded in words)
  are handled cleanly: on-time recall 1.00, only 0.15 extra early fires, digit acc 0.997.
  The model holds through the short number and fires at the end.
- **Longer-than-trained enumerations do NOT transfer.** On 16-digit card numbers the model
  fires ~3 s **before** the sequence ends (100% genuine mid-sequence, recall 0) — it hits its
  learned "≈10 digits = complete" point and interrupts. 13-digit is the transition: ~20%
  premature, ~12% correct, the rest never fire (holds, waiting for a pattern that never
  completes to its 10-digit template).
- Transcription is unaffected (digit acc ≈1.0 everywhere) — this is purely an **endpointing /
  completeness** failure, exactly the mechanism the paper attributes to pattern-specific schemas.

## Paper impact
Turns the Limitations assertion "street addresses, serial numbers, and card numbers are
untrained" into a measured transfer number, and **strengthens** the very next sentence ("a
learned completeness judge remains the general lever beyond pattern-specific schemas"): the
10-digit schema demonstrably does not extrapolate to longer enumerations. Suggested rewrite:
"The dictation schema transfers to shorter enumerations (street addresses: 1.00 on-time recall
zero-shot) but not to longer-than-trained ones (16-digit card numbers: the model fires at its
learned ~10-digit completeness point, ~3 s early, recall 0) — a learned completeness judge is
the general lever beyond pattern-specific schemas."

Artifacts: research/93-gen-{card,phone13,addr}-release.json; probes data/probes_{card,phone13,addr}.
