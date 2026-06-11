# TTB Label Verifier

Verifies a U.S. alcohol beverage label (beer, wine, or distilled spirits) against the
applicant's submitted values and the federal labeling rules (TTB / 27 CFR). Upload the
label image(s) — front and back together — optionally enter the application values, and
each field comes back **pass**, **needs review**, or **fail** with a reason.

**Production web app:** Next.js (App Router) frontend + a thin FastAPI serverless
function, deployed as one Vercel project. The original Streamlit prototype (`app.py`)
still works locally and is kept for reference.

## Architecture

```
Browser (Next.js, TypeScript + Tailwind)
   │  POST /api/py/verify  (multipart: 1–2 images + optional application JSON)
   ▼
FastAPI function (api/index.py — validation, orchestration, error mapping ONLY)
   │
   ├─► extraction.py    the vision model READS the label (never sees expected values)
   └─► verification.py  deterministic Python JUDGES (rules + application match)
            ▲
        config.py        regulatory constants & thresholds
```

Two stages — **the model reads, deterministic Python judges** — and the web layer adds
no third stage: `api/index.py` contains zero business logic. The engine modules at the
repo root are untouched and shared by the web API, the Streamlit prototype, the unit
tests, and the eval/benchmark harnesses.

Routing: the frontend always calls relative `/api/py/*`. In dev, `next.config.ts`
proxies that to uvicorn on `:8000`; in production a rewrite sends it to the
`api/index.py` function (Vercel's Next.js + FastAPI hybrid pattern). The browser is
same-origin with the API in both cases, so there is **no CORS surface at all** — by
design, not by `allow_origins=["*"]`.

## Repository layout

```
api/                  FastAPI serverless function (index.py) + Pydantic models (_models.py)
src/                  Next.js app: app/ (pages), components/, lib/ (types, api client, image prep)
config.py             regulatory constants & tunable thresholds (env-overridable)
extraction.py         vision extraction -> fixed JSON schema
verification.py       deterministic verification -> pass/needs_review/fail per field
tests/                test_verification.py (engine) + test_api.py (the new API glue)
app.py                legacy Streamlit prototype (local-only, excluded from deploys)
scripts/, eval/       smoke test, fixture generator, benchmarks, scored eval harness
vercel.json           Python function config (maxDuration, includeFiles)
.vercelignore         keeps all dev/test/tooling content out of deployments
```

## Key decisions (and why)

- **Engine stays at the repo root, imported via `sys.path`.** `api/index.py` prepends the
  repo root to `sys.path`. Vercel's Python builder bundles all non-ignored project files
  into the function by default, so the engine modules ship automatically — `.vercelignore`
  is the mechanism that scopes the bundle. Considered: copying the engine into `api/`
  (rejected — two diverging copies of regulatory logic) and packaging it
  (rejected — pyproject/src-layout machinery is overkill for a three-module engine that
  other in-repo harnesses import directly).
- **The API layer is deliberately thin.** Upload validation (count, size, magic-byte
  sniffing — the client's Content-Type is never trusted), application-JSON validation
  (Pydantic, `extra="forbid"` so a drifted frontend fails loudly), engine orchestration,
  and `extraction.failure_kind()` → HTTP status mapping. Nothing else.
- **Engine invariants are enforced at the boundary.** The extractor never receives
  application data; a blank form runs rules-only screening (`verify_label_only`) and is
  never auto-filled from the model's own read — that would let the model grade itself.
- **Type safety end to end.** Pydantic request/response models (`api/_models.py`)
  mirrored 1:1 by TypeScript types (`src/lib/types.ts`); the response is the engine's
  `FieldResult` contract serialized verbatim.
- **Vercel's 4.5 MB body limit is handled client-side first.** Images over ~1.8 MB are
  downscaled in the browser to ≤2048 px and re-encoded as JPEG — lossless for the model,
  whose high-detail vision pipeline caps input at 2048 px anyway. The API still enforces
  hard limits (4 MB/file, 4.3 MB total) with clear JSON errors, so no raw 413/500 ever
  reaches the user.
- **Tailwind v3, not v4.** v4's native engine (lightningcss/oxide) ships no 32-bit
  Windows binaries and the dev machine runs 32-bit Node; v3 is pure JS with identical
  output for this UI. Swap to v4 if your toolchain is 64-bit.
- **`requirements.txt` is the deploy manifest** (fastapi, python-multipart, openai,
  rapidfuzz — no Streamlit). Dev/test tooling lives in `requirements-dev.txt`; the
  Streamlit prototype's extras in `requirements-streamlit.txt`.

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | yes | OpenAI **API platform** key (platform.openai.com, billing enabled — not a ChatGPT subscription) |
| `EXTRACTION_MODEL` | no | override the vision model (default `gpt-5.4-mini`, chosen by a 5× stability benchmark) |
| `WARNING_BOLD_POLICY` | no | bold-gate policy for the government warning (default `header_body_gate`; see `config.py`) |

Locally you can also put the key in `.streamlit/secrets.toml` (the Streamlit prototype
reads it there); the FastAPI layer uses only `os.environ`. Never commit secrets — both
files are gitignored, and `.streamlit/` is additionally excluded from deploys.

## Local development

```bash
pip install -r requirements-dev.txt
npm install

# terminal 1 — API on :8000
npm run dev:api        # = uvicorn api.index:app --reload --port 8000
# (set OPENAI_API_KEY in this terminal's environment)

# terminal 2 — frontend on :3000 (proxies /api/py/* to :8000)
npm run dev
```

Open http://localhost:3000. Interactive API docs: http://localhost:8000/api/py/docs.

**Tests** (pure, no network — the model call is mocked in the API tests):

```bash
pytest                       # engine tests + API-layer tests
npm run test:api             # just the API layer
```

**Legacy Streamlit prototype:**

```bash
pip install -r requirements-streamlit.txt
streamlit run app.py
```

End-to-end harnesses that call the real model (cost money): `python scripts/smoke_test.py
--group test_labels/real_labels test_labels/baseline_labels` and `python eval/run_eval.py`.

## Deploying to Vercel

1. Push the repo to GitHub and **Import** it in Vercel (framework auto-detects Next.js),
   or run `npx vercel` from the repo root.
2. Set the `OPENAI_API_KEY` environment variable in the Vercel project settings.
3. Deploy. `vercel.json` configures the Python function (`maxDuration: 60` — a 2-image
   read takes ~7–10 s and the engine's own request ceiling is 30 s), and `.vercelignore`
   keeps tests, fixtures, benchmarks, docs, and the Streamlit prototype out of the
   deployment entirely (the Python builder bundles everything it doesn't exclude).

Notes: request bodies are capped at 4.5 MB by the platform (handled client-side, see
above). The Python runtime is 3.12 (the engine needs ≥3.10 for its type syntax).

## The engine (unchanged): how verification works

1. **Extraction** (`extraction.py`) — the vision model reads the image(s) and returns a
   fixed JSON schema (Structured Outputs, with a JSON-mode fallback). Each field is
   `{present, value, confidence}`, distinguishing "absent from the label" from "present
   but unreadable". The model only transcribes and reports what it sees; it never judges
   compliance and never sees the expected values. Front + back images of one product are
   read together as one label.
2. **Verification** (`verification.py`) — deterministic comparison per field type:
   fuzzy match for brand/class (against the union of the application's brand/fanciful
   and class/composition values, with a near-miss typo guard), numeric ABV comparison
   with the class-dependent presence rule (required for spirits, conditional for wine,
   optional for beer) plus label-only notation and proof = 2×ABV checks, unit-aware
   volume comparison for net contents, forgiving subset matching for name/address and
   country, and the fail-closed government-warning gate. Overall = worst field status;
   low-confidence reads escalate pass → needs review.

### Regulatory grounding

Rules are grounded in the three TTB Beverage Alcohol Manuals (cited in `config.py`) and
TTB's "Checklist of Mandatory Label Information" per class:

- **Government warning** (27 CFR part 16) — exact wording; "GOVERNMENT WARNING" in caps
  **and bold**, body **not** bold; "S"/"G" in Surgeon General capitalized. Wording and
  caps are judged deterministically from the transcription; bold is confidence-gated
  (`header_body_gate`): pass only when both bold rules are confirmed at high confidence,
  fail on a high-confidence violation, needs-review otherwise. Font-weight detection from
  photos is unreliable (see `BENCHMARK_NOTES.md`), so nothing uncertain ever auto-passes.
- **Alcohol content** — class-dependent presence; bare "ABV" notation fails (not a TTB
  form); a proof inconsistent with the stated ABV fails.
- **Wine appellation** — conditionally mandatory when the label shows a varietal,
  vintage, semi-generic designation, or estate claim (27 CFR 4.25/4.34).

Conditional disclosures whose triggers aren't visible (sulfites ppm, FD&C Yellow #5,
aspartame, …) are transcribed verbatim into `additional_statements` for the reviewer —
deliberately no automated pass/fail.

### Assumptions & limitations

- Expected values are supplied by the reviewer (production would integrate COLA).
- Standard-of-fill container sizes, type-size/legibility rules, and "same field of
  vision" layout rules are out of scope (layout/geometry isn't reliably verifiable from
  one photo).
- The vision model isn't perfectly deterministic; the warning check absorbs this (near
  misses go to review, never silent passes), and bold reads vary run to run — which is
  why the gate fails closed.
- This calls an external vision API; a production deployment inside TTB's network would
  need an on-prem or allowlisted model.

A screening aid — not a final legal determination.
