# Gaps & out-of-scope items — verifier coverage findings

From `eval/coverage_matrix.csv` + the fixture run (per-check scored). **Report-first: production code was NOT modified to hide these.**

## 1. gap-no-check — mandatory items the verifier does NOT check

- **Designation spelled correctly** (spirits, 27 CFR 5.165) — Explicit checklist line. Fuzzy match TOLERATES a dropped/changed letter, so a misspelled designation can silently PASS. No spelling/dictionary check.
- **Formula / statement of composition / formula number** (spirits, 27 CFR 5.141 / 5.165) — Rum "with natural flavors added" likely requires an approved formula. formula_number is application-side and NOT modeled; statement-of-composition vs formula not compared.
- **Designation spelled correctly** (malt, 27 CFR 7.63) — Fuzzy match tolerates misspelling; no spelling check.
- **Fanciful name matches application ""Fanciful Name"" field** (malt, 27 CFR part 7 subpart I) — Fanciful name "Honey Huckleberry Pie" is not a modeled field; no separate fanciful-name match. Brand vs fanciful is conflated in brand_name.
- **Formula / statement of composition / formula number** (malt, 27 CFR 7.63 / part 7 subpart I) — Honey/huckleberry flavor likely requires a formula; formula_number application-side, not modeled.
- **Alcohol content - required by trigger** (malt, 27 CFR 7.65) — ABV is mandatory for malt ONLY when alcohol from added flavors/non-beverage ingredients or a state requires it. The verifier treats a MISSING malt ABV as PASS (optional) and does NOT know the flavor trigger -> a flavored malt with NO ABV would be wrongly PASSED. (Our sample shows 5%, so present.)
- **Designation spelled correctly (e.g. ""Chardonay"")** (wine, 27 CFR 4.34) — FIXTURE GAP: a misspelled varietal ("Chardonay") should FAIL but no such fixture exists; fuzzy match would likely PASS/REVIEW it anyway (no spelling check).

## 2. Confirmed by a fixture this run (defect PASSED its exercised check)

- (none)

## 3. Calibration / strictness findings

- `GW-REWORD`: PARTIAL (flagged needs_review, not fail; near-miss wording policy) (`government_warning`=needs_review, expected FAIL).
- `GW-COMMA`: PARTIAL (flagged needs_review, not fail; near-miss wording policy) (`government_warning`=needs_review, expected FAIL).

## 4. Unexpected misses (investigate)

- (none — every other fixture scored as expected on its exercised check)

## 5. Non-determinism (a top finding) — brand reads AND warning-bold vary run-to-run

The extractor (a vision model) is not deterministic. Two checks flip across runs, so a SINGLE run is not authoritative for them (the deterministic checks below ARE reliable):
- **Warning bold/clean** — a compliant all-caps warning sometimes passes and sometimes fails the bold gate (CONTROL/CONTROL-CASE and the clean malt baseline flipped between runs). Matches the bold instability documented in BENCHMARK_NOTES.md.
- **Brand vs fanciful name** — the model picks a different prominent name across runs (e.g. malt "MALT & HOP BREWERY" vs "Honey Huckleberry Pie"; "JON'S" lands PASS one run, FAIL the next), so the brand verdict is unstable around the fuzzy cutoff.
  - (brand stable this run)

## 6. structural-out-of-scope — not verifiable from a flat image

- **Designation separate and apart** (spirits,malt,wine, 27 CFR 5.141 / 7.52(b) / 4.34) — Layout/positioning; not verifiable from per-field text extraction. README note.
- **Same field of vision (brand+ABV+class)** (spirits, 27 CFR 5.63) — FOV co-location of brand/ABV/class on one side; not verifiable from independent field reads.
- **Health warning - separate and apart / one statement** (spirits,malt,wine, 27 CFR Part 16) — Separation/position of the warning is not verifiable from the transcribed text alone.

## 7. Fixture gaps (checklist items with NO test fixture yet)

- Misspelled designation (e.g. varietal "Chardonay") — "spelled correctly" is an explicit checklist line; no fixture, and fuzzy match would likely tolerate it.
- Flavored-malt with the ABV statement OMITTED — would wrongly PASS (flavor trigger not modeled); no fixture.
- `rum_back_reworded_CONTAMINATED.png` (GW-REWORD) is `clean=no` in the manifest (duplicate word + re-rendered layout) — not a clean single-defect fixture.
- Manifest data fix: the CONTROL-CASE row had an unquoted comma in its defect field (CSV mis-parse); quoting was added so the row scores correctly.
- Conditional disclosures (country of origin / sulfites / Yellow #5 / cochineal / aspartame) — no observable trigger; transcribed but not asserted, by design.