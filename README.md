# Alcohol label verification (prototype)

Checks that an alcohol beverage label (beer, wine, or distilled spirits) matches its
application data and meets the federal labeling rules. Upload the label image(s) — front and back together — enter the
expected values, and the app reads the label and returns **pass**, **needs review**, or
**fail** with a reason per field. A **batch mode** screens many labels at once.

## Setup

```bash
pip install -r requirements.txt
```

Add your OpenAI API key (one of these):

- Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste your key, or
- Export it: `export OPENAI_API_KEY=sk-...` (PowerShell: `$env:OPENAI_API_KEY = "sk-..."`)

The key comes from the OpenAI **API** platform (platform.openai.com) with billing
enabled — that is separate from a ChatGPT subscription.

## Run

```bash
streamlit run app.py
```

## Test

**Unit tests** — pure, no network (~150 tests):

```bash
pip install pytest
pytest
```

They cover the regulation-critical paths — the government-warning rules (graded wording,
deterministic caps, confidence-gated bold), the class-dependent ABV rule, proof handling — plus the
extraction schema coercion and the per-model request params.

**End-to-end smoke test** — calls the real model on local images:

```bash
# point smoke_test at folders of front+back image pairs — it reads each folder's top level
# only (not recursive), so pass the leaf folders, not test_labels/ itself:
python scripts/smoke_test.py --group test_labels/real_labels test_labels/baseline_labels
```

It runs the full pipeline (extraction → rules screening) per **bottle** (front + back read
together), printing the extracted JSON, the verdict, and the per-bottle read time, and
writes a timestamped report to `output/`. Use it to validate extraction accuracy and the
~6–7s latency on real labels without opening the UI.

**Evaluation harness** — end-to-end scored pipeline check:

```bash
python eval/run_eval.py
```

This runs the full pipeline against the `error_labels/` single-defect fixtures (each errored
face paired with a clean other face) and the clean baselines, grading each field per-check and
writing a checklist-driven report to
`eval/results/{results.csv, completeness.csv, gaps.md, summary.md}`. It does not modify
production code.

## Approach

Two stages — **the model reads, deterministic Python judges**:

1. **Extraction** (`extraction.py`) — a vision model (OpenAI API) reads the label and
   returns a fixed JSON schema. Each field is `{present, value, confidence}`, which
   distinguishes "absent from the label" from "present but unreadable." The model only
   **transcribes and reports what it sees** (including whether the warning header is in
   caps/bold); it never judges compliance, and it never sees the expected values, so it
   can't echo back the answers. You can upload several images for one product (front +
   back) — they're sent together and read as one label, since mandatory items are split
   across labels (the government warning, net contents, and name/address are usually on
   the back).
2. **Verification** (`verification.py`) — plain Python compares each field, with a
   strategy per field type:
   - brand name, class/type → **fuzzy** match (so "Stone's Throw" == "STONE'S THROW")
   - alcohol content → **numeric** comparison, with the class rule below
   - net contents → unit-aware **volume** compare: exact match passes; the same volume in a
     different unit/format (e.g. "16.9 FL OZ" vs "1 PINT 0.9 FL OZ") → *needs review*; a
     materially different volume → fail; an unparseable value falls back to fuzzy
   - name & address, country of origin → forgiving subset/partial fuzzy match
   - government warning → wording matched on the body (exact → pass-path · near-miss read
     → *needs review* · large deviation → fail), the **caps** rule verified
     deterministically from the transcription when the header appears in it (else from the
     model's caps observation, fail-closed), and **bold** confidence-gated — the model
     judges it and the verifier fails closed when it can't be confirmed (see Regulatory
     grounding below)

   Each field gets pass / needs review / fail; the overall result is the worst of them,
   with the government warning as a hard gate. Low-confidence reads are downgraded from
   pass to *needs review* so a human double-checks them.

**Model selection:** gpt-5.4-mini was selected after a 5× stability benchmark because it
caught the controlled not-bold warning every run while avoiding gpt-4.1's false failures on
realistic compliant labels.

### Regulatory grounding (TTB Beverage Alcohol Manuals + checklists)

Field definitions and rules are grounded in the three TTB BAMs (cited in `config.py`) —
Distilled Spirits (TTB P 5110.7), Wine (TTB-G-2018-7), Malt Beverages (TTB P 5130.3) —
and cross-checked against TTB's official "Checklist of Mandatory Label Information" for
each class. Three class-specific rules are implemented:

- **Government warning** — identical across all three classes (27 CFR part 16): the exact
  wording, "GOVERNMENT WARNING" in **capital letters and bold**, and the "S"/"G" in
  "Surgeon General" capitalized. Wording is matched on the body (all-caps printing is
  fine), caps are verified deterministically, and **bold** is confidence-gated — the model
  judges header stroke weight and the verifier **fails closed** when bold can't be
  confidently verified (see Limitations and `BENCHMARK_NOTES.md`). Title case fails.
- **Alcohol content** — **required** for spirits, **conditional** for wine (≤14% "table
  wine"/"light wine" may omit it), and **optional** for malt beverages. The verifier
  treats a missing ABV accordingly instead of failing every blank.
- **Appellation of origin (wine)** — **conditionally mandatory**: required when the label
  names a grape varietal, a vintage date, or an estate-bottled claim
  (27 CFR 4.25 / 4.34). The model extracts `appellation` and `vintage`; for a wine with a
  varietal/vintage the verifier **fails** when no appellation is found; a present-but-unreadable
  appellation goes to review instead (the full varietal list is 27 CFR 4.91, so an unrecognized
  varietal simply doesn't trigger the requirement). This is the one
  conditional requirement whose trigger is visible on the label.

Other BAM-mandatory-but-conditional disclosures (sulfites ≥10 ppm, FD&C Yellow #5,
saccharin/aspartame, cochineal, age/commodity/state-of-distillation statements) are
**transcribed verbatim into `additional_statements` and shown to the reviewer**, not
given dedicated pass/fail logic — their triggers (e.g. ppm) aren't observable from the
image or the application data, so a human makes the call.

## Batch mode

Upload many labels at once. Each is screened concurrently (thread pool, bounded by the
API rate limit) against the fixed rules — the government warning and mandatory-field
presence — with a progress bar, a results table sorted worst-first, and per-label
detail. One unreadable image is reported as an error row rather than sinking the batch.
Per-application *matching* (vs the typed values) is single-label only; in production the
expected values would come from COLA (see Assumptions).

## Tools

Streamlit (UI), OpenAI API (vision extraction), RapidFuzz (fuzzy matching).

## Assumptions

- The reviewer supplies the expected values (in real use these would come from COLA;
  this prototype does not integrate with COLA).
- Handles beer/wine/spirits; the sample in the brief is distilled spirits.
- The government warning text is the standard federal statement (27 CFR part 16).
- Appellation of origin (wine) is **conditionally mandatory** and is now checked when the
  label shows a varietal or vintage (27 CFR 4.25 / 4.34). Vintage itself is optional — it
  only triggers the appellation requirement. Matching appellation against an application
  value (vs. presence-when-required) is a future enhancement.

## Limitations / next steps

- **Net contents** parses value + unit and compares by **volume** (so "0.75 L" and "750 mL"
  are recognized as the same volume → *needs review* on the unit/format difference, not a silent
  pass). Still out of scope: the regulatory **standard-of-fill** table (the list of permitted
  container sizes) — only volume agreement with the application is checked.
- **Thresholds** in `config.py` are starting points. Tune them against a small labeled
  test set (include deliberately bad labels: title-case warning, missing warning, ABV
  mismatch, brand mismatch).
- **Model & latency** — `gpt-5.4-mini` is the default (~4s/bottle, shown in the UI; 30s
  timeout in `config.py`), chosen by a **5× stability pass**: under `confidence_gate` it
  caught the not-bold adversarial (`03_notbold`) 5/5, passed the compliant one 5/5, failed
  title-case/reworded 5/5, **and** passed the realistic baselines 14/15 — where the prior
  default `gpt-4.1` caught the violation but **false-failed every realistic baseline** (read
  their bold headers as not-bold). `gpt-5.5` is the accuracy ceiling (same behavior, slower).
  The model is swappable via the `EXTRACTION_MODEL` env var, and the request params adapt
  automatically (Structured Outputs with a JSON-mode fallback; the gpt-5/o-series use
  `max_completion_tokens` + low/minimal reasoning instead of `temperature`). The full cross-model
  comparison and the stability pass are in `BENCHMARK_NOTES.md`.
- **Warning bold (font weight)** — 27 CFR 16.22 has **two** visual rules: "GOVERNMENT
  WARNING" must be **bold** *and* the remainder of the warning may **not** be bold. The
  default `header_body_gate` policy checks both: it **passes** only when the header is bold
  **and** the body is not-bold, **each read at high confidence**; it **fails** on a high-confidence
  violation of either, and routes anything uncertain — including **medium**-confidence reads — to
  **needs review** (a human verifies) rather than hard-failing or guessing. Font-weight detection
  on small photographed labels is unreliable — a broad model benchmark (`BENCHMARK_NOTES.md`) shows
  vision models disagree on bold and report high confidence even when wrong — so bold is never
  auto-passed on a shaky read, and **medium confidence is never sufficient for a PASS**. (The
  **caps** rule, by contrast, is verified deterministically from the transcription.) Two other
  modes are retained for comparison via `WARNING_BOLD_POLICY` (never the default): the older
  header-only `confidence_gate` (fail-closed, no body check), and the **experimental**
  `medium_pass_gate` (like the default but accepts medium confidence on PASS — benchmarked and
  rejected as default because it adds false passes on not-bold violations).
- **Warning legibility / type size** (minimum mm by container size) is not verified from the
  image — out of scope for a prototype.
- **Transcription variance** — the vision model is not perfectly deterministic and can
  misread small print (e.g. one word of the warning). The warning check absorbs this: an
  exact read passes the wording gate, a near-miss goes to *needs review* for a human to
  verify, and only a large deviation fails — so a model slip never silently passes nor
  hard-fails a compliant label.
- **Same field of vision** (spirits) — TTB's modernized Part 5 requires the brand name,
  alcohol content, and class/type to share one field of vision (27 CFR 5.63). That needs
  label-layout / container geometry a single cropped image can't reliably establish, so
  it is not auto-verified — noted as a reviewer check.
- **Batch matching** screens labels against the rules only; per-application matching at
  batch scale would ingest expected values from COLA.
- **Firewall** — this prototype calls an external vision API. In TTB's production
  network, outbound calls to a cloud API would be blocked, so a production version would
  need an on-prem or allowlisted model (e.g. one deployed inside the agency's own cloud
  tenant).

Do not commit `secrets.toml`, test label images (`test_labels/`), or run artifacts
(`output/`) — all are gitignored.
