# Bold-requirement research memo

*Date: 2026-06-08. Source: a deep-research pass (multi-source web search → adversarial
verification of each claim, 24/25 claims confirmed). This memo records the regulatory and
technical findings behind how the app treats the government-warning **bold** rule, and
complements the empirical results in `BENCHMARK_NOTES.md`.*

## TL;DR (bottom line)

Two independent methods — our own benchmarks (`BENCHMARK_NOTES.md`, `scripts/benchmarks/bold_variant_benchmark.py`)
and this regulatory + literature research — converge on the **same** conclusion:

> **Do not auto-FAIL a compliant label on a noisy bold boolean.** There is no quantitative
> bold standard to verify against; the regulator itself judges bold *by eye* from the same
> lossy raster images we ingest; and photo-based font-weight detection is unreliable at
> warning-text scale. Gate on **legibility** ("can the stroke weight even be judged in this
> image?"), route uncertain reads to **review / "submit a clearer image,"** and reserve
> **auto-FAIL for high-confidence affirmative defects** (a clearly thin/regular header, or —
> newly identified below — an *all-bold* warning).

This is the variant-D direction (`formatting_quality → REVIEW`) we derived empirically. The
research validates the **direction**; our pending variant-D real-label run measures the
**magnitude** (the REVIEW rate).

## 1. The regulation does not define "bold" quantitatively

27 CFR §16.22(a)(2), verbatim:

> "The first two words of the statement required by § 16.21, i.e., 'GOVERNMENT WARNING,'
> shall appear in capital letters and in bold type. **The remainder of the warning statement
> may not appear in bold type.**"

- No named font weight, stroke width, or stroke-to-height ratio. "Bold" is **purely
  qualitative**, judged by eye. (Confirmed across eCFR, Cornell LII, and all three TTB
  category pages — beer/wine/spirits.)
- TTB quantifies **type size** but not **weight**: minimum height by container volume —
  **1 mm** (≤237 ml / 8 fl oz), **2 mm** (>237 ml–3 L), **3 mm** (>3 L) — with
  character-density caps (40/25/12 chars per inch). Bold is the one formatting attribute
  with zero quantification.
- **New, actionable:** the rule forbids the **remainder** from being bold, so an *all-bold*
  warning is itself a §16.22 defect — one the app does **not** currently check, and one that
  rides on an *affirmative* read (which VLMs handle more reliably than the negative).

**Implication:** building a pixel-precise bold detector solves a *harder* problem than the
law imposes. The requirement is conspicuousness/format judged qualitatively, not a measurement.

## 2. TTB reviews from lossy raster, not vector artwork

- COLAs Online accepts label images **only as JPEG or PNG** (TIFF rejected; PDF/vector only
  for *supporting attachments*, not the label image). An upcoming change narrows this further
  *within* raster (toward JPEG/PNG).
- TTB **explicitly anticipates phone photos / scans** and tells applicants to crop the
  background out. The recommended image regime is modest and lossy-tolerant: **120–170 dpi**,
  **≤1.5 MB** per image, JPEG quality **"Medium" (~70/100)** — framed so specialists "can read
  every word," *not* so stroke thickness can be measured.

**Correction to an earlier hypothesis.** We previously floated "verify bold from the vector
artwork in the COLA submission" as a likely top fix. The research refutes that path at the
regulator: **TTB does not have or use vector for the label image — it judges bold by eye from
the same kind of degraded raster we ingest.** Source-artwork verification is only available if
*our own product workflow* can require applicants to upload print-ready design files — an open
question about deployment, not something COLA provides.

## 3. Photo-based bold detection is unreliable (corroborates our ~18%)

- **Bold is learnable on clean scans** — a CNN reaches ~0.98 patch / 1.00 page accuracy on
  *font emphasis* (none/bold/italic/bold-italic). But that ceiling holds **only** on clean
  typewritten, flatbed-scanned documents (150–600 dpi) — no photos. This matches our
  synthetic-image reliability.
- **It collapses on photographs** — the "real-to-synthetic domain gap." DeepFont reaches
  ~0 top-5 error on synthetic renderings but only **~80% top-5 (worse top-1) on real
  photographed text**, even with domain adaptation. The gap persists into the VLM era
  (~15–30% on font-recognition tasks). Mirrors our synthetic-reliable / real-unreliable split.
- **Classical stroke-width CV won't save it.** The Stroke Width Transform is *font-invariant*
  by design (built to *detect* text, not measure weight) and its documented failure cases are
  exactly ours: strong highlights/glare, blur, too-small type, curved baselines. This
  corroborates our inconclusive `scripts/benchmarks/measure_bold.py` result.
- **VLM self-confidence is miscalibrated ("confident-wrong").** Measured ECE is high and
  models assert wrong answers at near-100% confidence (e.g. GPT-4V: 62.6% stated confidence
  at 51.2% accuracy; Gemini worse). This directly explains our intermittent, high-confidence
  ~18% false-fails — and is **why the legibility signal should drive routing more than the
  bold-confidence number.**

## 4. Recommended decision policy

1. **Source artwork is the gold path** — *if* the workflow can require applicants to upload
   print-ready files, verify weight there (explicit in font metadata / measurable at full
   resolution) and skip the photo problem for those cases. Not available via COLA.
2. **Photo-only:** make **legibility** (`formatting_quality`) the primary gate. Marginal /
   unusable → **REVIEW** ("could not verify bold — submit a clearer image / human review"),
   not FAIL. (= variant D.)
3. **Reserve auto-FAIL for high-confidence affirmative defects** on a clear image: a clearly
   thin/regular header, or an **all-bold** warning (§16.22 remainder defect).
4. **False-fail vs false-pass asymmetry:** auto-failing a *compliant* label on a noisy boolean,
   for an attribute the regulator itself doesn't measure, is hard to justify. The asymmetry
   favors caution against false-fails here — route, don't reject.

## Caveats / open questions

- "TTB judges by eye" is a strong **inference** (no public TTB procedure documents day-to-day
  bold adjudication); it follows from the absence of any measurement standard + the lossy
  raster regime, not a quoted TTB process.
- The VLM-calibration studies tested GPT-4V / Gemini, not GPT-5-class; the *direction*
  (verbalized confidence is untrustworthy) is robust, the *magnitude* on our model is not.
- The font-ID studies are multi-way identification, not binary bold — they bound the
  difficulty by analogy. **Our own ~18% remains the most directly relevant datapoint.**
- Unresolved: can our applicants supply source artwork at all? And is the legibility signal
  itself well-calibrated enough to be the gate (its own false-route rate is unmeasured)?

## Update — the crop/resolution lever was tested (Experiment B3) and is unsafe

The open question "would a higher-resolution crop of just the warning fix bold?" was tested
directly: **B3** (Cloud Vision OCR-locate → tight crop → gpt-5.4-mini "is the header thicker than
the body?"). On compliant labels the crop reduced some false-FAILs, but **controlled negative
testing showed a 60% FALSE-PASS rate on font-controlled not-bold / all-bold headers** (including a
confident false-pass on a *clean* regular-weight header), and the model never abstained. So **more
pixels improved perception but not honesty** — the model reports its prior ("warning headers are
bold") rather than the strokes. Bold-from-photo remains unsafe in the false-pass direction even
with the resolution lever; B3 is benchmark-only and **not** wired into the verdict. See
`BENCHMARK_NOTES.md` → "Side experiments" for the full result. The production posture stands:
auto-check wording + caps; do not auto-pass bold on model judgment; fail closed / reviewer
confirmation when bold can't be verified.

> **Update 2026-06-11:** the bold PASS gate was relaxed per course-staff guidance — production
> now runs `WARNING_BOLD_POLICY = "medium_pass_gate"`, whose PASS accepts a **medium-or-high**
> confidence model judgment on both bold rules (so a confident-enough model read *does*
> auto-pass). The fail-closed core is unchanged: auto-FAIL stays reserved for high-confidence
> violations, and unknown/low-confidence reads still route to review. See `BENCHMARK_NOTES.md`
> (`medium_pass_gate` section) for the measured trade.

## Sources

Regulatory (primary):
- 27 CFR §16.22 — https://www.ecfr.gov/current/title-27/chapter-I/subchapter-A/part-16/subpart-C/section-16.22
- 27 CFR §16.21 — https://www.law.cornell.edu/cfr/text/27/16.21
- TTB COLAs Online FAQs (image formats / resolution) — https://www.ttb.gov/faqs/colas-and-formulas-online-faqs
- TTB malt-beverage health-warning guidance — https://www.ttb.gov/regulated-commodities/beverage-alcohol/beer/labeling/malt-beverage-health-warning

Technical (primary / peer-reviewed / arXiv):
- Font-emphasis CNN (clean-scan ceiling), arXiv:2108.13382
- DeepFont (real-to-synthetic gap), arXiv:1507.03196 — and arXiv:1504.00028
- VLMs lost in font recognition, arXiv:2503.23768
- Stroke Width Transform, Epshtein et al., CVPR 2010 (Microsoft Research)
- VLM overconfidence, arXiv:2504.14848
- LLM/VLM calibration (ECE figures), Groot & Valdenegro-Toro, NAACL TrustNLP 2024, arXiv:2405.02917
