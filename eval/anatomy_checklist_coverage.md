# Anatomy / Checklist coverage report — synthetic COLA audit

_Audit of the three artificial application JSONs (`test_labels/applications/{rum,malt,wine}.json`) against the
synthetic baseline labels (`test_labels/baseline_labels/baseline_{1,2,3}`), the TTB **mandatory-label-information
checklists** (authoritative), the **Anatomy of a Label** examples (placement), and the **Beverage Alcohol Manuals**
(deeper context). It maps each checklist item to: where it appears on the baseline label, whether the **LLM should
extract it as evidence**, whether the **deterministic verifier should judge it**, and what stays **reviewer-only** or
**out-of-scope**. No production code was changed; only the JSON `_checklist` notes were refreshed to match current
verifier behavior._

> **Update (post-audit schema change).** Acting on this audit's Stage-4 finding, `extraction.py` was subsequently
> extended with three **dedicated, evidence-only** extraction fields — `fanciful_name`, `statement_of_composition`,
> and `sulfite_declaration`. They are now their **own** schema fields (no longer buried in `class_type` / `brand_name` /
> `additional_statements`), captured extract-if-visible while the extractor stays blind to the application. **They are
> NOT consumed by `verify()` and change no verdict** — they remain **reviewer-only / not verifier-judged**. So every
> "not a dedicated field (gap)" / "schema gap" note below is **now an extraction field, but still not a judged field**;
> the per-row "Verifier" / "Judge?" columns for these three are unchanged (still `no`). A 10× baseline coverage audit
> after the change measured `fanciful_name`, `statement_of_composition`, and `sulfite_declaration` at 10/10 capture
> where expected, `class_type` holding 10/10 (the designation is kept in `class_type` even when also captured in the
> new fields — they are not mutually exclusive), and `name_and_address` leaks at 0/30.

## Authority used (in priority order)
1. **Checklist PDFs** = the bible for what is mandatory/conditional by beverage type.
2. **Anatomy examples** = where/how each item appears visually (front vs "other"/back, prominence).
3. **Beverage Alcohol Manuals** = definitions/exceptions for ambiguous items.
4. **Project brief** = prototype scope; does **not** override the checklists.

Top-level JSON values are taken from the **baseline label images**, not guessed from the manuals.

## Stage 2 result — application values vs labels: **all match**
Verified two ways (my direct read of the 6 images + an independent agent read). Every top-level `verify()`-facing
value in all three JSONs matches its baseline label. **No value corrections were needed.** Two interpretation nuances
where the JSON is the *better* reading: rum's `Spiced Rum` is treated as the class/type (with `Rum with natural flavors
added` as the statement of composition) rather than a fanciful name; and malt's `name_and_address` is the assembled
front-label block (`Brewed & Bottled By Malt & Hop Brewery, Hyattsville, MD`), which sits on the **front**.

## Legend
- **Req**: M = mandatory · C = conditional (trigger shown) · O = optional · P = placement · T = typography
- **LLM extract?**: yes (dedicated schema field) · partial (folded into another field/`additional_statements`) · reviewer-only · no
- **Verifier**: yes · partial · no · reviewer-only · structural/out-of-scope
- **Judge?** (deterministic Python): yes · partial · no · future

---

## Distilled spirits — `rum.json` / baseline_1 (Captain John's Spiced Rum)

| Checklist item | Cite | Req | Anatomy (side) | On baseline_1 | Schema field | LLM extract? | Verifier | Judge? |
|---|---|---|---|---|---|---|---|---|
| Brand name | 5.64 | M | Brand (front) | `CAPTAIN JOHN'S` (front) | `brand_name` | yes | yes (fuzzy vs brand/fanciful union) | yes |
| Class/type designation | 5.165/5.141 | M | Class/type display (front) | `SPICED RUM` (front) | `class_type` | yes | partial (fuzzy) | partial — spelling/recognized-class not checked |
| Statement of composition | 5.141 | **C** (specialty/flavored designation path) | SoC banner (front) | `RUM WITH NATURAL FLAVORS ADDED` (front) | `statement_of_composition` | yes (dedicated, evidence-only) | no | no — **dedicated field, not verifier-judged** |
| Distinctive/fanciful name | 5.141 | **C** (required only for DSS specialty needing a SoC; optional otherwise) | (the designation doubles as the distinctive name here) | — | `fanciful_name` | yes (dedicated, evidence-only) | no | no — alone it does **not** satisfy class/type |
| Alcohol content (ABV) | 5.65 | M | Alcohol content (front) | `20% ALCOHOL BY VOLUME (40 PROOF)` | `alcohol_content` | yes | yes (numeric **+ notation**) | yes |
| Proof | 5.65 | M (when shown) | (within ABV stmt) | `(40 PROOF)` | `alcohol_content.proof` | yes | **yes — proof = 2×ABV** | yes |
| Net contents | 5.70/5.203 | M | Net contents (other) | `750 ML` (back) | `net_contents` | yes | partial (**unit-aware volume**) | partial — **standard-of-fill not enforced** |
| Name & address | 5.66–68 | M | Name/address (other) | `DISTILLED & BOTTLED BY: ABC DISTILLERY / FREDERICK, MD` | `name_and_address` | yes | partial (normalized fuzzy + coverage guard) | partial — **producer-substitution gap; no-intervening-text not checked** |
| Health warning | part 16 | M | GW (other) | back, mixed-case | `government_warning` | yes | yes wording/CAPS/S-G; **bold reviewer-leaning** | yes (text) / reviewer (bold) |
| Country of origin | 5.69 / 19 CFR 134 | C (imports) | — | absent (domestic) | `country_of_origin` | yes | yes (when imported) | yes |
| Sulfite / coloring / FD&C #5 / cochineal | 5.63(c)(5–7) | C | — | absent | `additional_statements` | reviewer-only | no | no — trigger not observable |
| Wood treatment / commodity-neutral-spirits / state-of-distillation / age | 5.71–5.74, 5.66(f) | C (whisky/blends) | — | n/a (rum) | `additional_statements` | reviewer-only | no | no — needs product/formula data |
| Same field of vision (brand+ABV+class) | 5.63 | P | — | — | — | no | **structural/out-of-scope** | no |
| Designation separate & apart | 5.165/5.141 | P | — | — | — | no | **structural/out-of-scope** | no |
| GW separate & apart / single statement | part 16 | P | — | — | — | no | **structural/out-of-scope** | no |
| GW typography — CAPS + **BOLD** + S/G | part 16 | T | — | header bold-ish | `government_warning.header_bold/body_bold` | yes (observe) | CAPS+S/G yes · **BOLD reviewer (confidence-gated, unstable)** | yes (caps) / reviewer (bold) |
| Formula number | 5.141 | application-side | — | — | — | no | **out-of-scope** (not modeled) | no |
| Website / marketing / UPC | — | O | website/promo/UPC (other) | `www.ttbcaptjohn.com`, slogan, barcode | `additional_statements` | reviewer-only | no | no — excluded from `name_and_address` by prompt clause |

---

## Malt beverage — `malt.json` / baseline_2 (Malt & Hop Brewery "Honey Huckleberry Pie")

| Checklist item | Cite | Req | Anatomy (side) | On baseline_2 | Schema field | LLM extract? | Verifier | Judge? |
|---|---|---|---|---|---|---|---|---|
| Brand name | 7.64 | M | Brand (front) | `MALT & HOP BREWERY` (front) | `brand_name` | yes | partial (fuzzy union; brand/fanciful read unstable) | partial |
| Distinctive/fanciful name | part 7 subpart I | **C** (required only if not known to the trade under a class/type + uses formula/SoC path; optional otherwise) | Fanciful name (front) | `Honey Huckleberry Pie` (front) | `fanciful_name` | yes (dedicated, evidence-only) | no | no — **dedicated field, not verifier-judged**; alone it does **not** satisfy class/type |
| Designation (class/type or SoC) | 7.63 / subpart I | M | Class/SoC (front) | `Ale with Honey and Huckleberry Flavor` | `class_type` (+ dedicated `statement_of_composition`) | yes (class judged; SoC dedicated, evidence-only) | partial (fuzzy, on `class_type`) | partial |
| Net contents | 7.70 | M (US customary) | Net contents (front) | `1 PINT 0.9 FL. OZ.` (front) | `net_contents` | yes | partial (**unit-aware volume**, compound parsed) | partial — largest-whole-unit/format not validated |
| Alcohol content (ABV) | 7.65 | **C — triggered** | Alcohol content (front) | `5% ALC./VOL.` (front) | `alcohol_content` | yes | yes (numeric **+ notation**) | yes — **but the flavor TRIGGER is not modeled** (a *missing* flavored-malt ABV would wrongly PASS) |
| Name & address (bottler) | 7.66 | M (domestic; conditional on wholly-US-fermented) | Name/address (front) | `BREWED & BOTTLED By MALT & HOP BREWERY … HYATTSVILLE, MD` (front) | `name_and_address` | yes | partial (normalized fuzzy + coverage) | partial — no-intervening-text not checked |
| Health warning | part 16 | M | GW (other) | back | `government_warning` | yes | yes wording/CAPS/S-G; bold reviewer | yes / reviewer |
| Country of origin / importer name | 7.69 / 7.67–68 | C (imports) | — | absent (domestic) | `country_of_origin` | yes | yes (when imported) | yes |
| FD&C #5 / cochineal / sulfite / **aspartame** | 7.63(b)(1–4) | C | — | absent | `additional_statements` | reviewer-only | no | no — trigger not observable (aspartame needs CAPS+separate, also structural) |
| Designation separate & apart | 7.52(b) | P | — | — | — | no | **structural/out-of-scope** | no |
| GW single / separate & apart | part 16 | P | — | — | — | no | **structural/out-of-scope** | no |
| GW typography — CAPS + **BOLD** + S/G | part 16 | T | — | — | `government_warning` | yes (observe) | CAPS+S/G yes · **BOLD reviewer** | yes / reviewer |
| Formula number | subpart I | application-side | — | — | — | no | **out-of-scope** | no |
| Series tagline / marketing / website / UPC | — | O | promo/website/UPC (other) | `FARM TO TABLE SERIES #1`, paragraph, `www.maltandhopbrewery.com`, barcode | `additional_statements` | reviewer-only | no | no |

---

## Wine — `wine.json` / baseline_3 (Lighthouse "Stormchaser White" Chardonnay 2018)

| Checklist item | Cite | Req | Anatomy (side) | On baseline_3 | Schema field | LLM extract? | Verifier | Judge? |
|---|---|---|---|---|---|---|---|---|
| Brand name | 4.33 | M | Brand (front) | `LIGHTHOUSE` (front) | `brand_name` | yes | partial (fuzzy union; brand/fanciful/varietal unstable) | partial |
| Fanciful name | 4.34 | **O/C** (NOT mandatory; may appear on any label; does **not** satisfy class/type) | Fanciful name (front) | `STORMCHASER WHITE` (front) | `fanciful_name` | yes (dedicated, evidence-only) | no | no — **dedicated field, not verifier-judged** |
| Class/type = grape varietal | 4.21/4.34/4.91 | M | Class/varietal (front) | `Chardonnay` (front) | `class_type` | yes | partial (fuzzy; not matched to 4.91 approved list) | partial |
| Appellation of origin | 4.25/4.34 | **C — triggered** (varietal+vintage) | Appellation (front) | `HUDSON RIVER REGION` (front) | `appellation` | yes | **yes — presence required when triggered** (`_check_appellation`, from label) | yes |
| Vintage date | 4.27 | C | Vintage (front) | `2018` (front) | `vintage` | yes | trigger-only (forces appellation) | partial — not independently matched |
| Alcohol content | 4.36 | **C** (≤14% table wine may state "table wine") | Alcohol content (other) | `ALC. 13.5% BY VOL.` (back) | `alcohol_content` | yes | yes (numeric **+ notation**; table-wine omission handled) | yes |
| Net contents | 4.37/4.72 | M | Net contents (other) | `750 ML` (back) | `net_contents` | yes | partial (**unit-aware volume**) | partial — standard-of-fill 4.72 not enforced |
| Name & address | 4.35 | M | Name/address (other) | `PRODUCED and BOTTLED By LIGHTHOUSE VINTNERS / Kingston, NY` | `name_and_address` | yes | partial (normalized fuzzy + coverage) | partial — no-intervening-text not checked |
| Sulfite declaration | 4.32(e) | **C — present** (≥10ppm) | Sulfite decl. (other) | `CONTAINS SULFITES` (back) | `sulfite_declaration` | yes (dedicated, evidence-only) | no — presence not asserted | no — trigger not observable |
| Health warning | part 16 | M | GW (other) | back, ALL-CAPS (compliant) | `government_warning` | yes | yes wording/CAPS/S-G; bold reviewer | yes / reviewer |
| Country of origin | 19 CFR 134 | C (imports) | — | absent (domestic) | `country_of_origin` | yes | yes (when imported) | yes |
| % foreign wine / FD&C #5 / cochineal | 4.32(a)(4),(c),(d) | C | — | absent | `additional_statements` | reviewer-only | no | no — trigger not observable |
| Designation separate & apart | 4.21/4.34 | P | — | — | — | no | **structural/out-of-scope** | no |
| Name/addr bottler phrase, no intervening text | 4.35 | P | — | — | — | no | **structural/out-of-scope** | no |
| GW separate & apart | part 16 | P | — | — | — | no | **structural/out-of-scope** | no |
| GW typography — CAPS + **BOLD** + S/G | part 16 | T | — | header all-caps | `government_warning` | yes (observe) | CAPS+S/G yes · **BOLD reviewer** | yes / reviewer |
| Grape-varietal approved-for-domestic (4.91) | 4.91 | C | — | (Chardonnay) | — | no | **out-of-scope** (no approved-list match) | no |
| Marketing / website / UPC | — | O | promo/website/UPC (other) | paragraph, `www.lighthousestormchaser.com`, barcode | `additional_statements` | reviewer-only | no | no |

---

## Stage 4 — extraction schema vs mandatory items

- **Is the LLM asked to extract every mandatory item visible on these labels?** Almost. The dedicated schema covers
  brand, class/type, alcohol content (+proof), net contents, name/address, country, appellation, vintage, and the
  government warning — i.e. all mandatory **scalar** items present on the baselines.
- **Conditional designation-path items — now dedicated, evidence-only schema fields** (closed since this audit; see
  the update banner at the top). Each is captured extract-if-visible and surfaced to the reviewer, but **none is
  consumed by `verify()`** — they add evidence, not a verdict:
  1. **`fanciful_name`** — **conditional, NOT universally mandatory.** For **wine** it is optional, may appear on any
     label, and does **not** satisfy the required class/type designation. For **spirits/malt** it is required only for
     specialty/formula products that use the *distinctive-name + statement-of-composition* designation path; optional
     otherwise. Now its **own** extraction field (was conflated with `brand_name`); the verifier's `{brand, fanciful}`
     union still scores `brand_name` against the application's fanciful value — the new field does not feed that union.
     *(see each JSON `_checklist.fanciful_name`)*
  2. **`statement_of_composition`** — **conditional**: required *when* the distinctive/fanciful-name path is used (DSS
     specialty / flavored / formula products; present here on rum and malt). Now its **own** extraction field (was
     folded into `class_type` / `additional_statements`); still **no dedicated verifier check**.
  3. **`sulfite_declaration`** — **conditional, present** on wine (≥10ppm SO₂). Now its **own** extraction field (was
     transcribed into `additional_statements`); presence is **not asserted** and the ≥10ppm trigger is not observable,
     so it stays **reviewer-only**.
  - **`class_type` / the designation is the universally-required element** for all three beverage types — a fanciful
    name alone never satisfies it. The prompt keeps the designation in `class_type` even when the same text is also
    captured in `fanciful_name` / `statement_of_composition` (the three fields are **not mutually exclusive**), so the
    brand/class verifier check is unaffected. These were **schema gaps** at audit time; they are now **dedicated
    extraction fields but still not verifier-judged fields**.
- **Is anything optional over-collected into mandatory fields?** Not anymore. The earlier over-assembly of URLs /
  net-contents / marketing into `name_and_address` was closed by the prompt exclusion clause (measured 0/30 leaks);
  marketing/URLs/UPC now land in `additional_statements` (reviewer-only, no verdict).
- **Where conditional disclosures go:** sulfite, coloring, FD&C #5, cochineal, aspartame, %-foreign-wine, age,
  commodity, wood — all transcribed into `additional_statements` as **reviewer-only** evidence. Their triggers
  (≥10ppm SO₂, ingredient use, blend composition, formula) are **not observable from the image**, so the verifier
  asserts nothing — a deliberate scope decision, not an omission.

## Stage 5 — stale `_checklist` notes reconciled (what changed in the JSONs)
Updated `verifier_checks` notes to match **current production behavior** (verified this session):

| item | was | now |
|---|---|---|
| `proof` (rum) | "not validated" | **yes — proof = 2×ABV checked** (`_check_proof_consistency`, `PROOF_ABV_TOLERANCE`) |
| `alcohol_content` (all) | "notation not checked" | **+ notation now checked** (bare `ABV` rejected, `_check_abv_notation`) |
| `net_contents` (all) | "string/fuzzy; std-of-fill not checked" | **unit-aware VOLUME compare** (same-volume/different-unit → review); std-of-fill still not enforced |
| `name_and_address` (all) | "fuzzy; no-intervening-text" | **punctuation/relationship-prefix-normalized fuzzy + coverage guard**; **producer-name substitution = known gap (xfail)** |
| `health_warning` (all) | "exact wording + caps + S/G + bold" | clarified: wording/CAPS/S-G **deterministic**; **bold confidence-gated (`WARNING_BOLD_POLICY`, default `medium_pass_gate` since 2026-06-11) and run-to-run unstable** |
| malt `alcohol_content` | — | clarified: ABV is **conditional** (flavor trigger / state); **flavor trigger not modeled** |
| wine `alcohol_content` / `appellation` / `sulfite` | "yes" | marked **conditional** per the checklist (table-wine ABV; varietal/vintage-triggered appellation; ≥10ppm sulfite) |
| (new key) `_layout_typography_structural` | — | added to all three: same-field-of-vision, separate-and-apart, no-intervening-text, GW bold typography → **structural/out-of-scope or reviewer-leaning** |

## Stage 6 — are the JSONs complete artificial COLA records?
Yes. Each carries the **complete set of mandatory label items** for its beverage type as top-level values (matching
the label), plus `_checklist` (citation + applicability + verifier support), `_not_applicable` (conditional items
absent on this product), and now `_layout_typography_structural`. The records are sufficient to exercise both LLM
extraction (evidence) and verifier matching (judgment).

## Known gaps & out-of-scope (consolidated)
- **Dedicated extraction fields, not verifier-judged (closed schema gaps):** `fanciful_name`, `statement_of_composition`, and `sulfite_declaration` are now their own evidence-only extraction fields (no longer buried in `class_type`/`brand_name`/`additional_statements`), but **`verify()` does not consume them** — they stay reviewer-only and change no verdict. The remaining gap is a *verifier* one (no dedicated check), not an *extraction* one.
- **Verifier gaps:** producer-name substitution (xfail/known), flavored-malt ABV **trigger** not modeled (missing ABV
  would wrongly PASS), grape-varietal-vs-4.91-approved-list, recognized-class/spelling.
- **Reviewer-only:** all conditional disclosures (sulfite/coloring/FD&C/cochineal/aspartame/%-foreign), the
  government-warning **bold** observation (confidence-gated, unstable), marketing/website/UPC.
- **Structural/out-of-scope (not verifiable from flat per-field text):** same field of vision, designation
  separate-and-apart, name/address no-intervening-text, warning separate-and-apart/single, standard-of-fill table,
  formula number, container/blow-molded contents.

## Uncertainty / source limitations
- The baseline labels are **synthetic** and were transcribed into the JSONs; the values were re-verified against the
  images (two independent reads agreed), so confidence is high. Barcode digits are illustrative.
- The Anatomy examples are **generic TTB teaching labels**, not the synthetic products — used only to confirm *where*
  each item type appears (front vs other, prominence), which matched the baselines.
- Checklist PDFs were read by a vision/text agent; citations were cross-checked against the JSON citations and the
  `eval/coverage_matrix.csv`. A couple of conditional citations (e.g. malt name/address "wholly fermented in the
  U.S.") are summarized from the checklist wording and should be confirmed against the manual if used for a real COLA.
- This audit changed **no production code** and **no top-level application values**; only `_*` metadata notes and this
  report were edited/created.
