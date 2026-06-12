"""Thin FastAPI layer over the existing verification engine.

This module contains NO business logic — it validates the upload, orchestrates
the two engine stages (extraction.extract_fields -> verification.verify /
verify_label_only), and maps failures to meaningful HTTP responses. The engine
modules at the repo root are the source of truth and are imported untouched.

Import strategy (deliberate): the engine stays at the repo root (it is shared
with the test suite), and
this file puts the root on sys.path before importing it. On Vercel the Python
builder bundles ALL project files not excluded by .vercelignore into the
function by default (it honors `excludeFiles` only — there is no include knob),
so the root modules ship automatically; .vercelignore is what scopes the bundle.
Locally and in tests the same two-line bootstrap resolves them. We chose this
over copying the engine into api/ (a drift hazard: two copies of regulatory
logic) and over packaging it (pyproject + src layout — more moving parts than a
three-module engine warrants).

Engine invariants preserved here:
  - The extractor is blind: application data is parsed and held API-side and is
    never passed to extract_fields.
  - A blank/absent application form runs verify_label_only (rules-only
    screening); values are never auto-filled from the extraction.

CORS: none on purpose. The API is served same-origin with the Next.js frontend
(one Vercel deployment; in dev, next.config.ts proxies /api/py/* to uvicorn),
so no cross-origin surface exists. If the API is ever split onto its own host,
add CORSMiddleware with an explicit allow-list — not "*".
"""
import json
import logging
import os
import sys
from dataclasses import asdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from config import EXTRACTION_MODEL
from extraction import extract_fields, failure_kind
from verification import verify, verify_label_only

try:  # package import (uvicorn api.index:app, tests)
    from api._models import (ApplicationData, ErrorBody, ErrorResponse,
                             HealthResponse, VerifyResponse)
except ImportError:  # Vercel runs index.py as a top-level module, not a package
    from _models import (ApplicationData, ErrorBody, ErrorResponse,
                         HealthResponse, VerifyResponse)

logger = logging.getLogger("api")

# Vercel serverless functions reject request bodies over 4.5 MB before our code
# runs, so we enforce a slightly lower total here to own the error message; the
# frontend downscales images client-side (to the model's effective input size)
# so real requests stay well under both limits.
MAX_IMAGES = 4                       # one product: front + back/other (+ neck/strip labels)
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = int(4.3 * 1024 * 1024)

# Magic-byte signatures for the accepted formats. The client-sent Content-Type
# is untrusted; the sniffed type is what gets passed to the engine.
_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)


def _sniff_image_type(data: bytes) -> str | None:
    for sig, media_type in _SIGNATURES:
        if data.startswith(sig):
            return media_type
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


class ApiError(Exception):
    """Carries a status code + machine-readable kind to the error envelope."""

    def __init__(self, status_code: int, kind: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.kind = kind
        self.message = message


# extraction.failure_kind() -> (HTTP status, user-facing message). Statuses are
# chosen so the frontend can distinguish "your upload" (4xx) from "the service"
# (5xx): auth/quota are OUR misconfiguration, never the user's fault.
_FAILURE_RESPONSES = {
    "auth": (503, "The server's OpenAI API key is missing or invalid. "
                  "This is a server configuration problem, not an issue with your label."),
    "quota": (503, "The server's OpenAI account is out of credits. "
                   "This is a server configuration problem, not an issue with your label."),
    "rate_limit": (429, "The vision model is rate-limiting requests. "
                        "Wait a few seconds and try again."),
    "timeout": (504, "Reading the label took too long and was cancelled. "
                     "Try again, or upload a smaller/clearer image."),
    "connection": (502, "Could not reach the vision model service. "
                        "Check the server's network connection and try again."),
    "bad_response": (502, "The vision model returned an unusable response. "
                          "Try again; if it persists, try a clearer image."),
    "unknown": (500, "Label verification failed unexpectedly. Try again; "
                     "if it persists, check the server logs."),
}

app = FastAPI(
    title="TTB Label Verifier API",
    description="Verifies a U.S. alcohol beverage label against the applicant's "
                "submitted values and the federal labeling rules (27 CFR). "
                "The vision model transcribes; deterministic Python judges.",
    version="1.0.0",
    docs_url="/api/py/docs",
    openapi_url="/api/py/openapi.json",
)


@app.exception_handler(ApiError)
async def _api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=ErrorBody(kind=exc.kind, message=exc.message)).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(error=ErrorBody(
            kind="invalid_request",
            message="The request was malformed: " + "; ".join(
                f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', '')}"
                for e in exc.errors()[:3]),
        )).model_dump(),
    )


@app.exception_handler(Exception)
async def _unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
    """Catch-all so even an unexpected crash (e.g. in verify() or response
    serialization) returns the documented {error:{kind,message}} envelope
    instead of a bare text/plain 500."""
    logger.exception("unhandled error: %s", exc)
    status_code, message = _FAILURE_RESPONSES["unknown"]
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=ErrorBody(kind="unknown", message=message)).model_dump(),
    )


def _strip_phantom_422(openapi_fn):
    """FastAPI auto-documents a 422 on every route, but our validation handler
    remaps all validation failures to 400 — so a 422 can never occur. Strip it
    from the generated schema so /api/py/docs matches real behavior."""
    def wrapped():
        schema = openapi_fn()
        for path in schema.get("paths", {}).values():
            for operation in path.values():
                if isinstance(operation, dict):
                    operation.get("responses", {}).pop("422", None)
        return schema
    return wrapped


app.openapi = _strip_phantom_422(app.openapi)


def _parse_application(raw: str | None) -> dict:
    """Parse the optional application-data form field into the dict verify()
    expects. Returns {} for absent/blank input (-> rules-only screening)."""
    if raw is None or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ApiError(400, "invalid_application",
                       f"The application data is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ApiError(400, "invalid_application",
                       "The application data must be a JSON object of field values.")
    try:
        return ApplicationData.model_validate(data).cleaned()
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", [])) or "application"
        raise ApiError(400, "invalid_application",
                       f"Invalid application data ({loc}: {first.get('msg', 'invalid')}).") from exc


async def _read_validated_images(images: list[UploadFile]) -> list[tuple[bytes, str]]:
    """Read uploads into (bytes, sniffed_media_type) pairs, enforcing count,
    per-file size, total size, and real image signatures."""
    if not images:
        raise ApiError(400, "no_images", "Upload at least one label image.")
    if len(images) > MAX_IMAGES:
        raise ApiError(400, "too_many_images",
                       f"Upload at most {MAX_IMAGES} images (front and back of one product).")
    pairs: list[tuple[bytes, str]] = []
    total = 0
    for upload in images:
        # Bounded read: read(limit+1) proves oversize without buffering a huge
        # body into memory (matters on the dev path, where no platform cap exists).
        data = await upload.read(MAX_FILE_BYTES + 1)
        name = upload.filename or "image"
        if not data:
            raise ApiError(400, "empty_file", f"'{name}' is empty.")
        if len(data) > MAX_FILE_BYTES:
            raise ApiError(413, "file_too_large",
                           f"'{name}' exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB "
                           f"per-image limit.")
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise ApiError(413, "payload_too_large",
                           "The images together exceed the 4.3 MB upload limit. "
                           "Use smaller images (about 2000 px on the long side is plenty).")
        media_type = _sniff_image_type(data)
        if media_type is None:
            raise ApiError(400, "unsupported_type",
                           f"'{name}' is not a PNG, JPEG, or WebP image.")
        pairs.append((data, media_type))
    return pairs


@app.get("/api/py/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model=EXTRACTION_MODEL)


@app.post(
    "/api/py/verify",
    response_model=VerifyResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
               429: {"model": ErrorResponse}, 500: {"model": ErrorResponse},
               502: {"model": ErrorResponse}, 503: {"model": ErrorResponse},
               504: {"model": ErrorResponse}},
)
async def verify_label(
    images: list[UploadFile] = File(..., description="1–4 label images (front, back/other, "
                                                     "neck/strip) of ONE product"),
    application: str | None = Form(None, description="Optional JSON object of the "
                                                     "applicant-submitted field values"),
) -> VerifyResponse:
    pairs = await _read_validated_images(images)
    app_values = _parse_application(application)

    try:
        # The extractor sees only the images — never app_values (engine invariant).
        # run_in_threadpool: the engine's OpenAI call is synchronous (and its 429
        # backoff sleeps), so running it inline would block the event loop for the
        # whole read — freezing concurrent requests locally and under Vercel's
        # in-function concurrency. The engine has no shared mutable state (its
        # one global is the lazily-created thread-safe OpenAI client), so
        # concurrent threaded use is safe.
        extracted = await run_in_threadpool(extract_fields, pairs)
    except Exception as exc:  # noqa: BLE001 — failure_kind classifies, we map to HTTP
        kind = failure_kind(exc)
        status_code, message = _FAILURE_RESPONSES.get(kind, _FAILURE_RESPONSES["unknown"])
        logger.error("extraction failed (%s): %s", kind, exc)
        raise ApiError(status_code, kind, message) from exc

    result = verify(extracted, app_values) if app_values else verify_label_only(extracted)
    return VerifyResponse(
        mode="application_match" if app_values else "rules_only",
        overall=result["overall"],
        beverage_type=result["beverage_type"],
        fields=[asdict(f) for f in result["fields"]],
        additional_statements=result["additional_statements"],
        image_quality_notes=result["image_quality_notes"],
        # the raw coerced read: evidence for the reviewer (warning observations,
        # evidence-only fields, full JSON readout) — never re-judged client-side
        extracted=extracted,
    )
