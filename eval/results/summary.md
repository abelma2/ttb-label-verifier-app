# Label verifier — evaluation summary

**Defects caught (per-check): 9 of 11 fixtures.** 0 confirmed a known unchecked item, 2 were calibration/strictness findings, 0 unexpected.

**Control (clean) labels:** the compliance check passed on 2/2 controls this run. IMPORTANT: brand-name matching AND the warning-bold gate are non-deterministic run-to-run (a clean control can flip pass/fail) — see gaps.md #5. The deterministic checks (exact wording, S/G capitalization, ABV numeric, appellation) are the reliable signal.

**Mandatory checks supported:** 13 yes (incl. import-conditional), 13 partial, 27 no, of 53 checklist items across all three beverage types.

**Speed:** slowest label `PROOF-CONSISTENCY` at 11.7s; **13 of 14 labels exceeded the ~5s bar** — every full front+back read is ~6-9s, so the 2-image submission is over Sarah's 5s target.

## What it gets right
- Government warning: exact wording, ALL-CAPS header, "Surgeon General" capitalization, bold (lowercase S/G -> FAIL).
- ABV numeric mismatch vs the application (40% vs 20% -> FAIL).
- ABV notation: the bare "ABV" abbreviation is rejected (NOTATION-ABV -> FAIL; 27 CFR 5.65/7.65).
- Proof vs ABV consistency: proof != 2x ABV is rejected (PROOF-CONSISTENCY, 50 proof on 20% ABV -> FAIL; 27 CFR 5.65).
- Wine appellation-of-origin when a varietal/vintage requires it (cross-field rule -> FAIL).
- Net contents: same volume in a different unit/format -> NEEDS_REVIEW; a materially different volume -> FAIL (unit-aware, PR-B).
- Name/address: punctuation/relationship-prefix differences normalized (fewer false reviews); a short subset read that drops the producer name -> NEEDS_REVIEW. (KNOWN GAP: a producer-name word substitution can still pass the fuzzy score.)

## Top gaps for follow-up (see gaps.md)
- **Brand fuzzy cutoff too loose** — BRAND-FUZZY ("JON'S" vs "JOHN'S", one letter) PASSED instead of going to review.
- **Single-character warning edits** (missing comma) go to needs_review, not fail (near-miss wording policy).
- Spelling, same-field-of-vision, separate-and-apart, formula numbers: not verified.

_Deviations from the brief: the pre-populated `coverage_matrix_starter.csv` and a `.env` were not present, so the matrix was built from the three TTB checklists and the key was loaded from `.streamlit/secrets.toml`. The `.docx` was a real binary (extracted via its XML). Extraction uses the production `gpt-5.4-mini` reasoning model (temperature is not tunable for it). Fixtures are scored per-check because Gemini-rendered errored faces introduce unrelated read variance vs the real clean faces._