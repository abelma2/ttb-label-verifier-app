# TTB Label Verifier

Verifies a U.S. alcohol beverage label (beer, wine, or distilled spirits) against the
applicant's submitted values and the federal labeling rules (TTB / 27 CFR). Upload the
label image(s) — front and back together — optionally enter the application values, and
each field comes back **pass**, **needs review**, or **fail** with a reason. A **batch
mode** screens many labels at once: files pair into products by filename stem
(`oldtom_front.jpg` + `oldtom_back.jpg`), an optional CSV/JSON application file matches
rows to products, and results land in a worst-first table with per-product detail.

**Production web app:** Next.js (App Router) frontend + a thin FastAPI serverless
function, deployed as one Vercel project.

> `main` carries only what you need to download and run the app (plus a ready-made
> example label in `examples/`). The full development history — benchmarks, the scored
> eval harness, ground-truth fixtures, research notes, and the retired Streamlit
> prototype this app grew out of — is preserved on the
> [`dev-archive`](../../tree/dev-archive) branch.

## Architecture

```
Browser (Next.js, TypeScript + Tailwind)
   │  POST /api/py/verify  (multipart: 1–4 images + optional application JSON)
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
repo root are untouched and shared by the web API and the unit tests.

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
examples/             a ready-to-use demo label (front + back images + application file)
vercel.json           Python function config (maxDuration)
.vercelignore         keeps all dev/test/tooling content out of deployments
```

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | yes | OpenAI **API platform** key (platform.openai.com, billing enabled — not a ChatGPT subscription) |
| `EXTRACTION_MODEL` | no | override the vision model (default `gpt-5.4-mini`, chosen by a 5× stability benchmark) |
| `WARNING_BOLD_POLICY` | no | bold-gate policy for the government warning (default `header_medium_gate` — header bold gates, body bold is a tracked note; set `medium_pass_gate` or `header_body_gate` for the stricter two-rule gates; see `config.py`) |

The FastAPI layer reads the key from the environment only (`os.environ`). Locally, copy
`.env.example` to `.env` (gitignored) and the dev servers load it for you; on Vercel, set
the key in the project's environment variables. Never commit secrets.

## Local development

Prerequisites: **Python ≥ 3.10** (the engine uses modern type syntax) and
**Node ≥ 22.6** (`npm run test:web` uses Node's built-in TypeScript stripping).

### Fastest path

```bash
cp .env.example .env     # then paste your OpenAI key into OPENAI_API_KEY
npm run setup            # = pip install -r requirements-dev.txt && npm install
npm run dev:all          # API on :8000 + frontend on :3000, in one terminal
```

Open http://localhost:3000. The one prerequisite the repo can't supply for you is an
OpenAI **API platform** key (see [Environment variables](#environment-variables)); paste
it into `.env` and you're running. Interactive API docs: http://localhost:8000/api/py/docs.

### Manual (two terminals)

If you'd rather run the servers separately — or pass the key via the environment instead
of `.env`:

```bash
pip install -r requirements-dev.txt
npm install

# terminal 1 — API on :8000
export OPENAI_API_KEY="sk-..."              # PowerShell: $env:OPENAI_API_KEY = "sk-..."
npm run dev:api        # = uvicorn api.index:app --reload --port 8000 --env-file .env

# terminal 2 — frontend on :3000 (proxies /api/py/* to :8000)
npm run dev
```

`dev:api` loads `.env` via `--env-file`; an `OPENAI_API_KEY` already set in the
environment takes precedence, so the `export` form above still works.

**Tests** (pure, no network — the model call is mocked in the API tests):

```bash
pytest                       # engine tests + API-layer tests
npm run test:api             # just the API layer
npm run test:web             # batch grouping / application-file parsing (node:test)
```

## Deploying to Vercel

1. Push the repo to GitHub and **Import** it in Vercel (framework auto-detects Next.js),
   or run `npx vercel` from the repo root.
2. Set the `OPENAI_API_KEY` environment variable in the Vercel project settings.
3. Deploy. `vercel.json` configures the Python function (`maxDuration: 60` — a 2-image
   read takes ~7–10 s and the engine's own request ceiling is 30 s), and `.vercelignore`
   keeps tests, the `examples/` demo assets, and any local dev content out of the
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
  **and bold**; "S"/"G" in Surgeon General capitalized. Wording and caps are judged
  deterministically from the transcription; header bold is confidence-gated
  (`header_medium_gate`): pass when the bold header is confirmed at medium-or-high
  confidence, fail only on a high-confidence not-bold header, needs-review otherwise.
  Font-weight detection from photos is unreliable (see `BENCHMARK_NOTES.md` on the
  `dev-archive` branch), so a null/low-confidence bold read never auto-passes. The
  body-not-bold observation is recorded and surfaced to the reviewer as a note but does
  not affect the verdict. (The stricter two-rule gates that also judge the body remain
  available: `WARNING_BOLD_POLICY=medium_pass_gate` or `header_body_gate`.)
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
