# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An app that verifies a U.S. alcohol beverage label (beer, wine, or distilled spirits)
against the values an applicant submitted and the federal labeling rules (TTB / 27 CFR).
It returns **pass** / **needs review** / **fail** per field plus an overall verdict. Two
modes: **single label** (match against typed application values) and **batch** (screen
many labels against the fixed rules).

**Production is a Next.js + Vercel web app** (`src/` frontend + a thin FastAPI serverless
function in `api/`) over a shared, UI-agnostic Python **engine** (`extraction.py`,
`verification.py`, `config.py`). The original **Streamlit prototype (`app.py`) is legacy**
— it still runs locally and is kept for reference, but it is excluded from deploys and is
not the production surface. When a change affects "the app," it almost always means the
web app (`src/` + `api/`), not `app.py`. The engine is the source of truth for all
regulatory logic and is imported untouched by the web API, the Streamlit prototype, the
test suite, and the eval/benchmark harnesses — so the invariants below apply no matter
which front end calls them.

## Commands

```bash
# --- production web app (Next.js frontend + FastAPI serverless API) ---
pip install -r requirements-dev.txt           # engine + API deps + dev/test tools (uvicorn, pytest, httpx)
npm install                                    # frontend deps
npm run dev:api                                # terminal 1: API on :8000 (uvicorn api.index:app --reload --port 8000)
npm run dev                                     # terminal 2: frontend on :3000 (proxies /api/py/* to :8000) — open http://localhost:3000
npm run build                                   # production Next.js build
# Interactive API docs while dev:api runs: http://localhost:8000/api/py/docs

# --- tests (pure, no network — the model call is mocked in the API tests) ---
pytest                                         # all Python tests: engine (tests/test_verification.py, ~157) + API glue (tests/test_api.py, ~19)
pytest tests/test_verification.py::test_warning_absent_fails   # a single engine test
npm run test:api                               # just the API layer (= pytest tests/test_api.py -q)
npm run test:web                               # batch grouping / application-file parsing (node:test, src/lib/__tests__/*.mts)

# --- legacy Streamlit prototype (local only, not deployed) ---
pip install -r requirements-streamlit.txt      # adds streamlit + pillow on top of requirements.txt
streamlit run app.py

# end-to-end checks that DO call the real model (need an API key, cost money):
python scripts/smoke_test.py --group test_labels/real_labels test_labels/baseline_labels
#   full pipeline; groups <stem>_Front/_Other into one product. ~6–7s/bottle for an accurate front+back detail=high read.
#   Pass the LEAF image folders: smoke_test's _gather reads a folder's TOP LEVEL only (not recursive), so `test_labels` alone finds nothing.
#   Also: --each (one product per image), or no flag (all images = one product). Writes a timestamped report to output/result_<ts>.{txt,json}.
python scripts/generate_adversarial.py             # (re)generate adversarial/*.png — compliant / titlecase / notbold / reworded:
                                                    # ground-truth images for validating the government-warning rule (needs Pillow + Windows Arial).

python eval/run_eval.py                            # checklist-driven eval harness (real model). Runs the production pipeline
#   (extract_fields -> verify) over the error_labels fixtures (errored face + clean other face, scored PER-CHECK on the field
#   each defect exercises) plus a completeness pass on the clean baselines. Reads eval/coverage_matrix.csv for known gaps;
#   writes eval/results/{results.csv, completeness.csv, gaps.md, summary.md}. Does NOT modify production code.

# model/policy benchmarks (also call the real model) that justify the choices in config.py;
# their findings are written up in BENCHMARK_NOTES.md:
python scripts/benchmarks/stability_benchmark.py              # 5x stability pass — why gpt-5.4-mini is the default model
python scripts/benchmarks/model_benchmark.py                  # cross-model accuracy/latency comparison
```

The app needs an OpenAI **API platform** key (platform.openai.com, billing enabled —
not a ChatGPT subscription):

- **Web app / engine / tests:** set the `OPENAI_API_KEY` env var (the FastAPI layer reads
  only `os.environ`). PowerShell: `$env:OPENAI_API_KEY = "sk-..."` (README shows the bash
  `export` form). On Vercel, set it in the project's environment variables. Other
  env-overridable knobs: `EXTRACTION_MODEL`, `WARNING_BOLD_POLICY` (see `config.py`).
- **Streamlit prototype only:** can additionally read the key from
  `.streamlit/secrets.toml` (copy `.streamlit/secrets.toml.example`).

Do not commit `secrets.toml` or test label images (both are gitignored).

## Architecture

**Two stages: the model reads, deterministic Python judges.** This separation is the
core design principle — keep it intact.

1. `extraction.py` — `extract_fields(images, media_type)` sends the image(s) to a
   vision model and returns a **fixed JSON schema** (see below). `images` is one bytes
   object or a list (front + back of one product, read together in a single call). `_coerce()` normalizes
   whatever the model returns into that schema with safe defaults, so the verifier never
   defends against missing/odd keys. The model **transcribes and reports observations
   only** (e.g. whether the warning header is caps/bold) — it never judges compliance.
2. `verification.py` — `verify(extracted, application)` (single label, matches against
   the application) and `verify_label_only(extracted)` (batch, rules-only screening).
   Both return `{"overall", "fields": [FieldResult], "beverage_type",
   "additional_statements", "image_quality_notes"}`.

Every front end is **UI only** — it wires the two engine stages and never judges. The
production web app does this across the browser/serverless boundary (see "The web app"
below); the legacy `app.py` does it in one Streamlit process.

`app.py` (legacy Streamlit prototype) collects inputs →
`_extract` (→ `extract_fields`) → `verify`/`verify_label_only` → `_render_product`. Both modes accept **optional
application data** (single: the form, typed or prefilled from an uploaded CSV/JSON file; batch:
an uploaded CSV/JSON matched to products by filename stem). A product **with** application
values runs `verify()` (label-vs-application); **without**, it runs `verify_label_only()`
(rules-only screening). Batch fans out over a `ThreadPoolExecutor` (results keyed by index, not
label — products can share a stem) and by default groups uploaded files into products by
filename stem (`_group_uploads` / `_stem`, same `_Front`/`_Other` convention as
`smoke_test.py`) so a front+back pair is read together instead of the front false-failing the
warning that lives on the back. Results persist in `st.session_state`.

### The web app (production front end)

Next.js App Router frontend (`src/`, TypeScript + Tailwind **v3** — see below) + a **thin**
FastAPI serverless function (`api/index.py`) over the same engine, shipped as **one Vercel
project**. The web layer adds **no third stage**: `api/index.py` contains zero business
logic — it validates the upload (count/size, **magic-byte sniffing**; the client's
`Content-Type` is untrusted), validates the optional application JSON (`api/_models.py`
Pydantic, `extra="forbid"` so a drifted payload fails loudly), orchestrates
`extract_fields` → `verify`/`verify_label_only`, and maps `extraction.failure_kind()` to
HTTP status + a user-facing message. **The engine invariants below are enforced here too**:
the extractor never receives application data, and a blank/absent form runs
`verify_label_only` and is never auto-filled (the API has no "copy extracted into the form"
path — that single `app.py` convenience is the one deliberate non-port).

- **Engine import strategy.** The engine stays at the repo root; `api/index.py` prepends
  the root to `sys.path`. Vercel's Python builder bundles every non-excluded file into the
  function, so the engine ships automatically — **`.vercelignore` is the mechanism that
  scopes the deploy** (it excludes tests, fixtures, benchmarks, docs, `app.py`, and the
  Streamlit/dev requirements). Do **not** copy the engine into `api/` (two diverging copies
  of regulatory logic) or package it.
- **Routing / no CORS.** The frontend always calls relative `/api/py/*`. `next.config.ts`
  proxies that to uvicorn `:8000` in dev and rewrites it to the `api/index.py` function in
  prod (Vercel Next.js + FastAPI hybrid pattern). Same-origin in both cases → **no CORS
  surface by design**, not `allow_origins=["*"]`. If the API is ever split onto its own
  host, add an explicit allow-list.
- **Type safety end to end.** `api/_models.py` (Pydantic response models) is mirrored
  **1:1** by `src/lib/types.ts` (TS interfaces). The response is the engine's `FieldResult`
  contract serialized verbatim (`asdict`), plus the raw coerced `extracted` read as
  reviewer evidence. **Change the two model files together.**
- **Batch is client-side orchestration, not a batch endpoint.** Where Streamlit fanned out
  over a server-side `ThreadPoolExecutor`, the browser groups files into products and issues
  **one `/api/py/verify` request per product** (`src/lib/batch.ts`, `BATCH_CONCURRENCY = 8`
  to match `config.BATCH_MAX_WORKERS`) — each product is its own serverless invocation, so
  the platform scales it and one bad product becomes an error row, never a sunk batch.
- **The grouping and application-file logic are 1:1 TypeScript ports of `app.py`** —
  `src/lib/stem.ts` (↔ `_stem`/`_group_uploads`) and `src/lib/applications.ts` (↔
  `_parse_applications`/`_app_row_for`/`_pick_application_row`), including Python-truthiness
  string coercion and prototype-safe key handling. **Keep them in lockstep with `app.py`**;
  they are pinned by `src/lib/__tests__/*.mts` (`npm run test:web`).
- **Vercel's 4.5 MB body limit is handled client-side first.** `src/lib/image.ts`
  downscales images over ~1 MB to ≤2048 px and re-encodes as JPEG — lossless for the model
  (its high-detail pipeline caps input at 2048 px anyway). The API still enforces hard
  limits (`MAX_IMAGES=4`, 4 MB/file, 4.3 MB total) with clear JSON errors so no raw
  413/500 reaches the user. Products are capped at **4 images** (front/back/neck/strip) — a
  consequence of the request budget that `app.py`'s single-server upload didn't have.
- **Tailwind v3, not v4** — v4's native engine ships no 32-bit Windows binaries and the dev
  machine runs 32-bit Node (output is identical for this UI). Don't "upgrade" it.
- **Requirements are split by purpose:** `requirements.txt` is the lean **deploy manifest**
  (fastapi, pydantic, python-multipart, openai, rapidfuzz — **no streamlit**);
  `requirements-dev.txt` adds uvicorn/pytest/httpx; `requirements-streamlit.txt` adds the
  legacy prototype's streamlit + pillow. The error-attribution principle holds in both UIs:
  never blame the photo read for an upload-size/format/transport failure.

### Invariants worth preserving

- **The extractor is blind to the expected values.** It sees only the image, never the
  application data. Never thread application values into `extraction.py`.
- **The application data is an independent witness — never auto-fill it from the extraction.**
  The whole design is the model's label read checked against the applicant's *separately
  supplied* values. A blank form screens rules-only; it is **never** silently populated from
  the model's own read (that would make every field trivially pass — the model checked against
  itself, manufacturing confident false passes). `app.py` has one explicit convenience path
  (copy extracted values into the form to edit), and any result produced from unedited copied
  values is flagged "not an independent comparison."
- **Result statuses are shared string constants** `PASS / REVIEW / FAIL`
  (`"pass" / "needs_review" / "fail"`) in `verification.py`, imported by `app.py`. The
  middle tier (**needs review**) is deliberate.
- **Rollup** = worst field status (fail > review > pass). The government warning's FAIL is
  the worst severity, so the worst-status rollup already enforces it as a hard gate.
- **The government warning is judged FAIL-CLOSED** (`_check_warning`). Real labels print
  the warning all-caps and the model sometimes omits the `GOVERNMENT WARNING:` header
  from `text`, so: wording is matched on the **body** (case-insensitive, header stripped);
  header **caps** is judged deterministically from the text when the header is present,
  else from `header_all_caps` (explicit `False` always fails); the `S`/`G` in
  `Surgeon General` must be capitalized (all-caps satisfies this); and **bold** is judged per
  `config.WARNING_BOLD_POLICY` (default `"header_body_gate"`). 27 CFR 16.22 has **two** visual
  rules — `GOVERNMENT WARNING` must be **bold** AND the **remainder/body may NOT be bold** — so
  the gate checks both, from the model's `header_bold`/`header_bold_confidence` and
  `body_bold`/`body_bold_confidence`. It **PASSES only when** wording matches, the header is
  all-caps, `header_bold` is `True` at **high** confidence, **AND** `body_bold` is `False` at
  **high** confidence. It **FAILS** on a high-confidence violation of either (`header_bold False`
  → header not bold, or `body_bold True` → body bold). Anything uncertain (`None` / medium / low
  on either field) → **needs review**. `header_bold=True` **by itself can no longer pass** — the
  body/remainder must be confirmed not-bold. This closes the structural gap the benchmark series
  found (the old header-only `confidence_gate` auto-passed ~93% of all-bold-body violations
  because it never inspected the body weight; see `BENCHMARK_NOTES.md`). The other modes (kept
  for benchmarking, **never the production default**) are `"medium_pass_gate"`
  (**experimental, env-only**: identical to `header_body_gate` except its PASS gate also accepts
  **medium** confidence — FAIL is unchanged. A benchmark showed it cuts false reviews on clean
  labels but adds false PASSES on medium-confidence not-bold misreads, so it is **not** the
  default; production still requires **high** confidence on both bold rules to PASS — see
  `BENCHMARK_NOTES.md`), `"confidence_gate"` (prior default — header-only fail-closed gate: bold
  `True` + medium/high → pass, `False` + medium/high → FAIL, `None`/low → FAIL "submit a clearer
  image"; does **not** check the body), `"note"` (bold = telemetry only, otherwise-valid warning
  passes with a note), `"review"` (always escalate to a human), and `"trust_model"` (judge from
  `header_bold` alone, ignoring confidence). Title case fails; a near-miss wording read (the
  model misreading small print) goes to review, and nothing non-exact ever auto-passes. The
  project's most important correctness rule. **Medium confidence is never sufficient for a
  production PASS** (only the experimental `medium_pass_gate` accepts it).
- **Alcohol content is class-dependent** (`_check_abv` / `_abv_missing_by_class`):
  required for spirits, conditional for wine (≤14% "table"/"light" wine may omit it),
  optional for beer. Don't "simplify" this into failing every missing ABV.
  `_check_abv` also runs two **label-only** regulatory checks before the class/match logic
  (independent of the application): the notation must be a TTB form — the bare abbreviation
  `ABV` FAILS (`NONCOMPLIANT_ABV_NOTATIONS`, 27 CFR 4.36/5.65/7.65) — and the **proof must equal
  2× the ABV** (`PROOF_ABV_TOLERANCE`, 27 CFR 5.65), so an internally inconsistent label (e.g.
  50 proof on 20% ABV) FAILS even when the ABV number matches the application.
- **Net contents is compared by VOLUME, not just the printed string** (`_check_net_contents` /
  `_parse_volume`, `config.NET_CONTENTS_VOLUME_TOLERANCE`): an exact/whitespace-equivalent match
  PASSES; the **same parsed volume in a different unit/format** (e.g. `16.9 FL OZ` vs
  `1 PINT 0.9 FL OZ`, or `0.75 L` vs `750 mL`) routes to **needs review** — not an auto-pass, a
  human verifies the unit and standard of fill; a **materially different** parsed volume FAILS;
  an **unparseable** value falls back to the existing fuzzy string compare. `_parse_volume` sums
  compound US quantities ("1 PINT 0.9 FL OZ") but RECONCILES — does not sum — a dual declaration
  like `16.9 FL OZ (500 mL)`. The regulatory **standard-of-fill** table (the list of permitted
  sizes) is still **out of scope**; only volume agreement is checked.
- **Wine appellation is conditionally mandatory** (`_check_appellation`): a wine that names a
  grape varietal, shows a vintage, uses a semi-generic type designation (27 CFR 4.24(b):
  Burgundy/Chablis/Champagne/Chianti/Sherry/Port/…), or makes an estate-bottled claim must
  carry an appellation of origin (27 CFR 4.25/4.34).
  The check is appended to `results` **only for `beverage_type == "wine"`** (so non-wine field
  lists are unchanged — the batch six-field test still holds). Trigger detection uses
  `_COMMON_VARIETALS` (substring; full list is 27 CFR 4.91) and `_SEMI_GENERIC_DESIGNATIONS`
  (word-boundary, so "port" doesn't fire on "Portland"); a confident absence fails, an
  unreadable read reviews. An appellation embedded in the designation (e.g. "California
  Burgundy") satisfies it. It's the one conditional disclosure whose trigger is visible on the
  label (unlike sulfites/FD&C #5/etc., which stay in `additional_statements`).
- **Low-confidence reads escalate** a pass to review (`_escalate`, gated by
  `ESCALATE_LOW_CONFIDENCE`).
- **Brand/class match the UNION of application fields with a containment-aware scorer.**
  `verify()` scores `brand_name` against the best of the application's `{brand_name,
  fanciful_name}` and `class_type` against `{class_type, statement_of_composition}` using
  `fuzz.token_set_ratio` (not the default `token_sort_ratio`), via `_check_text` +
  `_candidates`. This is deliberate: the vision model routinely tags the fanciful name as the
  brand, reads the statement of composition as the class, or returns a more-verbose superset of
  the legal value — all the *same* label text under a different application key. The extractor
  stays blind (it never sees these values); the union is a verifier-side allowance. A genuine
  mismatch still scores far below `FUZZY_REVIEW_FLOOR`, so it does not mask wrong reads (verified
  by `test_brand_union_does_not_mask_genuine_mismatch`). Don't revert brand/class to a
  single-field `token_sort_ratio` compare. `_check_text`'s `expected` accepts a string or a
  list of acceptable values. Brand/class also use an **edit-distance near-miss guard**
  (`near_miss_review=True`, `TEXT_NEAR_MISS_EDIT_DISTANCE`): an otherwise-passing read that
  differs from the matched value by 1–2 characters (a likely typo, e.g. `JON'S` vs `JOHN'S`,
  which scores ~96) is routed to review, not auto-passed. A superset read has a large edit
  distance and is unaffected; an exact (normalized) match is distance 0.

### Extraction schema (the cross-module contract)

`extract_fields` returns this shape; `verify*` reads it; `app.py` renders it. **Changing
a field touches all three plus the prompt AND the strict `_EXTRACTION_SCHEMA` (Structured
Outputs) in `extraction.py`, and a test.**

```jsonc
{
  "beverage_type": "beer|wine|spirits|unknown",
  "brand_name|fanciful_name|class_type|statement_of_composition|net_contents|name_and_address|country_of_origin|appellation|vintage|sulfite_declaration":
      { "present": bool, "value": str|null, "confidence": "high|medium|low" },
  "alcohol_content": { "present", "value", "abv_percent": num|null, "proof": num|null, "confidence" },
  "government_warning": { "present", "text": str|null, "header_all_caps": bool|null, "header_bold": bool|null, "header_bold_confidence": "high|medium|low", "header_bold_basis": str|null, "body_bold": bool|null, "body_bold_confidence": "high|medium|low", "confidence" },
  "additional_statements": [ { "value": str, "kind": str|null, "confidence" } ],
  "image_quality_notes": str|null
}
```

`fanciful_name`, `statement_of_composition`, and `sulfite_declaration` are **evidence-only**
extraction fields (extract-if-visible). They give the conditional designation/disclosure
evidence its own home instead of burying it in `class_type` / `brand_name` /
`additional_statements`, but **`verify*` does not consume them** — they carry no pass/fail
logic and change no verdict; they are surfaced to the reviewer. The extractor stays blind, so
capturing the fanciful name here does **not** feed the brand/class union (which still reads
`class_type`); the prompt requires `class_type` to remain populated with the designation even
when the same text is also captured in `fanciful_name`/`statement_of_composition` (the three
fields are not mutually exclusive).

`additional_statements` is the catch-all for the remaining BAM-conditional disclosures that
lack a dedicated field (FD&C Yellow #5, saccharin/aspartame, cochineal,
age/commodity/state-of-distillation; the sulfite declaration now has its own
`sulfite_declaration` field): the model transcribes them verbatim and they're shown to the
reviewer, but they get **no dedicated pass/fail logic** (their triggers aren't observable).
This is a deliberate scope decision, not an omission.

`verify*` returns `{"overall", "fields": [FieldResult, ...], "beverage_type",
"additional_statements", "image_quality_notes"}` — `fields` is a list of `FieldResult`
dataclasses (`.field`, `.status`, `.reason`, `.extracted`, `.expected`, `.cause`), not dicts.
`.cause` is the optional machine-readable verdict-reason category for the government warning
(`absence`/`wording`/`caps`/`bold`/`low_confidence`); it is for programmatic branching, not the
user-facing `.reason` string, which is display text and may be reworded freely.

### Configuration

`config.py` centralizes everything regulatory or tunable: the canonical
`GOVERNMENT_WARNING` text (exact match — must be exactly right; verified verbatim against
all three BAMs), fuzzy/ABV/name-address thresholds, `WARNING_BOLD_POLICY` (default
`"header_body_gate"` — Pass/Review/Fail on BOTH the bold-header and non-bold-body rules; see
the warning invariant above and `BENCHMARK_NOTES.md`), `EXTRACTION_MODEL` (default `gpt-5.4-mini`, chosen by a 5× stability
pass), `REQUEST_TIMEOUT_SECONDS`, and `BATCH_MAX_WORKERS`. The BAM publication IDs are
documented there. Most knobs also read an env var override (e.g. `EXTRACTION_MODEL`) for A/B
testing without editing the file.

### Per-model request params (extraction.py)

`extract_fields` adapts its request to the model family (`_model_params` + a retry that
downgrades on error): it asks for **Structured Outputs** (strict `json_schema`) and falls
back to plain `json_object` for models that don't support it; the gpt-5 / o-series use
`max_completion_tokens` + `reasoning_effort` (their floor is `"low"`, not `"minimal"`) instead
of `temperature`, while the gpt-4 family keeps `temperature=0`. So swapping `EXTRACTION_MODEL`
generally "just works" without touching the call site.

### Startup detail

`app.py` copies the key from `st.secrets` into `os.environ` **before** importing
`extraction`, and the OpenAI client is created lazily (`_get_client`, with the timeout and
`max_retries=0`) so the env var is in place by first use. Keep client creation lazy —
instantiating it at import time would break this ordering. `max_retries=0` is deliberate:
it makes `REQUEST_TIMEOUT_SECONDS` a true ceiling (the SDK default of 2 retries would re-issue
a timed-out request); the per-model param-rejection retries live in `_create_with_fallbacks`,
not the client. `_create_with_fallbacks` also gives rate-limit 429s a short bounded backoff
(`RATE_LIMIT_MAX_RETRIES`, realistic when a batch bursts `BATCH_MAX_WORKERS` concurrent
calls) — a 429 returns immediately, so this does not undermine the timeout ceiling.
`extraction.failure_kind(exc)` coarsely classifies a failure (auth / quota / rate_limit /
timeout / connection / bad_response / unknown) so `app.py` can show accurate guidance instead
of blaming every error on the photo; `insufficient_quota` 429s (out of credits — permanent)
are never retried and classify as `quota`, not `rate_limit`.

### Scripts & repository layout

Production code is the **shared engine** (`extraction.py`, `verification.py`, `config.py`),
the **web app** (`src/` Next.js frontend + `api/` FastAPI serverless function), and
`tests/` (`test_verification.py` = engine, `test_api.py` = API glue;
`src/lib/__tests__/*.mts` = the TS ports). `app.py` is the **legacy** Streamlit prototype
(local only, `.vercelignore`'d). Everything else is tooling:

- `scripts/` — the two dev tools: `smoke_test.py` (run the real pipeline on local images;
  the ad-hoc end-to-end harness) and `generate_adversarial.py` (regenerate the `adversarial/`
  ground-truth set). These two stay here.
- `scripts/benchmarks/` — all model/policy experiments (each documented in
  `BENCHMARK_NOTES.md`) plus their shared `_paths.py`. **This project does not use Google
  services**: the Cloud Vision / Gemini helper modules and every benchmark that depended on
  them were deleted in the Google teardown (their findings remain recorded in
  `BENCHMARK_NOTES.md`). Do not reintroduce Google integrations or credentials.
- `eval/` — the **scored** end-to-end harness (the deliverable counterpart to `smoke_test.py`'s
  free-form runs): `run_eval.py` grades the production pipeline per-check against the
  `test_labels/error_labels/` fixtures and writes `eval/results/{results.csv, completeness.csv,
  gaps.md, summary.md}`. `coverage_matrix.csv` is the hand-built map of every checklist item to
  whether the verifier supports it (the source of the "known gap" rows); `coverage_matrix_starter.csv`
  is the unscored template. Unlike the benchmarks, this resolves ROOT with plain `os.path.dirname`,
  not `_paths`, because it sits one level down, not in `scripts/benchmarks/`.
  `eval/results_flag_on/` and `eval/results_flag_off/` are deliberately **tracked frozen
  snapshots** — the A/B evidence behind BENCHMARK_NOTES' "warning-crop flag stays off" verdict
  (`run_eval.py` itself only ever writes `eval/results/`); don't archive or delete them.
- `docs/` — `bold_research.md`, the regulatory research memo behind `WARNING_BOLD_POLICY`
  (the legal/CFR counterpart to `BENCHMARK_NOTES.md`'s empirical findings).
- `test_labels/` (gitignored images) is organized by purpose, and the harnesses read **leaf**
  folders: `baseline_labels/` (clean synthetic front+back pairs), `real_labels/` (photographed
  bottles), `error_labels/` (single-defect fixtures + `test_fixtures_manifest.csv`, consumed by
  `run_eval.py`), and `applications/` (`rum.json` / `malt.json` / `wine.json` — the typed
  "application" values the eval pairs each product against).

**Benchmark scripts resolve paths through `scripts/benchmarks/_paths.py`, never hardcoded
`os.path.dirname` depth.** Start a new benchmark script with
`import _paths; _paths.ensure_paths(); ROOT = _paths.ROOT`. `_paths` finds the repo root by
walking up to the `config.py` + `extraction.py` sentinel, and `ensure_paths()` puts three
dirs on `sys.path`: ROOT (for `extraction`/`config`), `scripts/` (for
`smoke_test`, which several benchmarks import but which lives one level up), and
`scripts/benchmarks/` (sibling modules). Hardcoding dirname depth silently breaks when files
move, and `from smoke_test import ...` fails unless `scripts/` is on the path — both were
real regressions this layout fixes.

The repo is on GitHub (`origin` → `abelma2/TTB-Compliance-App`, default branch `main`).
`.gitignore` ignores `secrets.toml`, caches, `output/`, `archive/`, scratch, and the
`test_labels/` **images** — but keeps the `test_labels/applications/*.json` fixtures (the typed
application values the eval/app compare against) and the small ground-truth fixture sets
(`adversarial/`, `bold_safety/`). Superseded run artifacts and scratch files live under
`archive/`. (Dev shell is Windows PowerShell; for multi-line commit messages use
`git commit -F <file>` — inline `-m` mangles quoting in PowerShell 5.1.)
