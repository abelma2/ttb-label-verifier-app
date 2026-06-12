# TTB Label Verifier

Verifies a U.S. alcohol beverage label (beer, wine, or distilled spirits) against the
applicant's submitted values and the federal labeling rules (TTB / 27 CFR). Upload the
label image(s) — front and back together — optionally enter the application values, and
each field comes back **pass**, **needs review**, or **fail** with a reason, with the
uploaded label shown beside the verdicts (click to zoom) so every read can be confirmed
against the image. A **batch mode** (the "Multiple labels" tab) screens many labels at
once: files pair into products by filename stem (`oldtom_front.jpg` + `oldtom_back.jpg`
— a trailing `_front`/`_back`/`_label`/`_other` marker is stripped, any capitalization;
a single stitched image `oldtom.jpg` works the same), an optional Excel (`.xlsx`)
application workbook matches rows to products — a ready-to-fill template with an
instructions sheet is downloadable in the app — and results land in a worst-first table
with per-product detail.

**Production web app:** Next.js (App Router) frontend + a thin FastAPI serverless
function, deployed as one Vercel project.

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
tests/                test_verification.py (engine) + test_api.py (API glue; mocks the model)
examples/             ready-to-use demo labels (images only — type application values
                      manually, or start from the in-app Excel template)
vercel.json           Python function config (maxDuration)
.vercelignore         keeps all dev/test/tooling content out of deployments
```

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | yes | OpenAI **API platform** key (platform.openai.com, billing enabled — not a ChatGPT subscription) |
| `EXTRACTION_MODEL` | no | override the vision model (default `gpt-5.4-mini`, chosen by a 5× stability benchmark) |
| `WARNING_SUPPLEMENT_MODEL` | no | model for the dedicated parallel second read of the government warning (default `gpt-4.1`; set empty to disable — see "Regulatory grounding" below) |
| `WARNING_BOLD_POLICY` | no | how the warning's bold observation is judged (default `supplement_gate`: bold → pass, not bold or unreadable → needs review, never fail; other modes in `config.py`) |
| `RATE_LIMIT_MAX_RETRIES` | no | bounded backoff retries for OpenAI rate-limit (429) responses when a batch bursts concurrent calls (default `2`; out-of-credit `insufficient_quota` errors are never retried) |

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
Then try the bundled demo labels — [examples/README.md](examples/README.md) is a
one-minute walkthrough.

> `npm install` resolves the `xlsx` dependency from the SheetJS CDN tarball (the npm
> registry's version is frozen at 0.18.5 with known vulnerabilities), so installs need
> `cdn.sheetjs.com` reachable.

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
npm run build                # production Next.js build — also the TypeScript type gate
```

`test:web` lists its test files explicitly in `package.json` — a new
`src/lib/__tests__/*.test.mts` file must be added there or it will silently never run.
There is no separate lint/typecheck script; `npm run build` (or `npx tsc --noEmit`) is
the type gate.

## Deploying to Vercel

1. Push the repo to GitHub and **Import** it in Vercel (framework auto-detects Next.js),
   or run `npx vercel` from the repo root.
2. Set the `OPENAI_API_KEY` environment variable in the Vercel project settings.
3. Deploy. `vercel.json` configures the Python function (`maxDuration: 60` — a 2-image
   read takes ~7–10 s and the engine's own request ceiling is 30 s), and `.vercelignore`
   keeps tests, the `examples/` demo assets, and any local dev content out of the
   deployment entirely (the Python builder bundles everything it doesn't exclude).

Notes: request bodies are capped at 4.5 MB by the platform — the frontend downscales
large images before upload (≤2048 px JPEG, lossless for the model's pipeline;
`src/lib/image.ts`) and the API still enforces hard limits (4 images, 4 MB/file, 4.3 MB
total) with clear JSON errors. The Python runtime is 3.12 (the engine needs ≥3.10 for
its type syntax).

## The engine: how verification works

1. **Extraction** (`extraction.py`) — the vision model reads the image(s) — front +
   back together as one label — and returns a fixed JSON schema. Each field is
   `{present, value, confidence}`, distinguishing "absent from the label" from "present
   but unreadable". The model only transcribes what it sees; it never judges compliance
   and never sees the expected values.
2. **Verification** (`verification.py`) — deterministic comparison per field type:
   fuzzy brand/class match against the union of the application's brand/fanciful and
   class/composition values (with a near-miss typo guard); ABV with the class-dependent
   presence rule (required for spirits, conditional for wine, optional for beer) plus
   label-only notation and proof = 2×ABV checks; unit-aware volume comparison for net
   contents; forgiving subset matching for name/address (the U.S. responsible party —
   on imports the importer; other producer/bottler statements surface as evidence) and
   country; and the fail-closed government-warning gate. Overall = worst field status;
   low-confidence reads escalate pass → needs review.

### Regulatory grounding

Rules are grounded in the three TTB Beverage Alcohol Manuals (cited in `config.py`) and
TTB's "Checklist of Mandatory Label Information" per class:

- **Government warning** (27 CFR part 16) — exact wording; "GOVERNMENT WARNING" in caps
  **and bold**; "S"/"G" in Surgeon General capitalized. The warning is read by a
  **dedicated second model** (`WARNING_SUPPLEMENT_MODEL`) in parallel with the main
  extraction — on ground truth it scored 100% on the full warning verdict (60/60) versus
  70% for the main read, and it transcribes what is printed rather than reciting the
  federal text. Wording and caps are judged deterministically from that transcription;
  bold at worst routes to **needs review** (confirm against the label image beside the
  verdicts) — it can never fail a label. When only one of the two readers finds a
  warning at all, absence is a review, never a one-reader fail.
- **Alcohol content** — class-dependent presence; bare "ABV" notation fails (not a TTB
  form); a proof inconsistent with the stated ABV fails.
- **Wine appellation** — conditionally mandatory when the label shows a varietal,
  vintage, semi-generic designation, or estate claim (27 CFR 4.25/4.34); a U.S. origin
  (state, county, AVA) or a foreign region (e.g. "CHAMPAGNE" near the brand) both
  satisfy it.

Conditional disclosures whose triggers aren't visible get no automated pass/fail: the
sulfite declaration is extracted into its own evidence-only field; the rest (FD&C
Yellow #5, aspartame, age statements, …) are transcribed verbatim into
`additional_statements` for the reviewer.

### Assumptions & limitations

- Expected values are supplied by the reviewer (production would integrate COLA).
- Standard-of-fill container sizes, type-size/legibility rules, and "same field of
  vision" layout rules are out of scope (layout/geometry isn't reliably verifiable from
  one photo).
- The vision model isn't perfectly deterministic; the checks absorb this — near misses
  go to review, never silent passes.
- This calls an external vision API; a production deployment inside TTB's network would
  need an on-prem or allowlisted model.

A screening aid — not a final legal determination.
