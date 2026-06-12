"""Vision extraction: send a label image to the OpenAI API and get back structured fields.

The model only ever sees the image -- never the expected application values -- so it
can't simply echo back the answers we're checking against. It TRANSCRIBES and reports
visual observations; the deterministic verifier (verification.py) does the judging.

Each field object is ``{"present": bool, "value": str|null, "confidence": "high|medium|low"}``,
distinguishing "absent from the label" (present=false) from "present but unreadable"
(present=true, value=null). ``_coerce`` guarantees the schema shape so the verifier
never has to defend against missing/odd keys.
"""
import base64
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

from openai import (OpenAI, APIConnectionError, APITimeoutError, AuthenticationError,
                    RateLimitError)

from config import (EXTRACTION_MODEL, RATE_LIMIT_MAX_RETRIES, REQUEST_TIMEOUT_SECONDS,
                    WARNING_SUPPLEMENT_MODEL)


class ExtractionError(Exception):
    """The model response could not be turned into a usable extraction (truncated, empty,
    or invalid JSON). Raised instead of silently coercing to an 'all-absent' read, which
    would surface as a confident (wrong) verdict rather than a read error."""


# Scalar fields sharing the plain {present, value, confidence} shape. fanciful_name /
# statement_of_composition / sulfite_declaration are evidence-only (extract-if-visible,
# not consumed by verify()); vintage triggers the conditional wine appellation check.
_SCALAR_FIELDS = (
    "brand_name", "fanciful_name", "class_type", "statement_of_composition", "net_contents",
    "name_and_address", "country_of_origin", "appellation", "vintage", "sulfite_declaration",
)

_PROMPT = """You are a transcription assistant for U.S. TTB alcohol-beverage label review. \
You are shown one OR MORE photos of a SINGLE alcohol beverage product (beer/malt, wine, or \
distilled spirits) -- for example the front and back/"other" labels of the same product. \
Read what is printed across ALL the images and return ONLY a JSON object in the exact schema \
below. Combine the images: report each field once, wherever it appears (brand and class/type \
are usually on the front; the government warning, net contents, and name/address are often on \
the back). You transcribe and report what you SEE -- you do NOT decide whether the label is \
compliant.

RULES
- Output JSON only. No prose, no markdown.
- Transcribe VERBATIM. Preserve original capitalization, punctuation, %, and symbols exactly \
as printed. Do not normalize, correct spelling, expand abbreviations, or tidy up text.
- present vs value:
    * If a field is genuinely not on the label, set "present": false and "value": null.
    * If it IS on the label but you cannot read it (glare, blur, angle), set "present": true, \
"value": null, and "confidence": "low".
- confidence ("high"|"medium"|"low") reflects how sure you are of YOUR reading of that field, \
based on image clarity -- not whether the value looks correct.
- Never guess or infer values that are not visibly printed.

FIELDS (definitions per the TTB Beverage Alcohol Manuals)
- beverage_type: "beer" (malt), "wine", or "spirits" from the class/type and overall label; \
"unknown" if undeterminable.
- brand_name: the brand under which the product is marketed -- usually the most prominent name.
- fanciful_name: a descriptive or distinctive product name used IN ADDITION to the brand name (e.g. a \
flavor, edition, or series name such as "SPICED RUM", "Honey Huckleberry Pie", or "STORMCHASER \
WHITE"). Extract it verbatim if visible; set present=false if there is no separate fanciful/distinctive \
name. A fanciful name does NOT by itself satisfy the class/type designation. These fields are NOT \
mutually exclusive: capturing a phrase here does NOT mean removing it from class_type.
- class_type: the class/type designation (specific identity), e.g. "Kentucky Straight Bourbon \
Whiskey", "Spiced Rum", "Cabernet Sauvignon", "Chardonnay", "India Pale Ale", "Ale". ALWAYS populate \
class_type with the visible product designation / class / type evidence, EVEN IF the same or related \
text is also captured in fanciful_name or statement_of_composition. Do not leave class_type empty or \
move the designation out of it just because those dedicated fields exist -- the verifier reads \
class_type.
- statement_of_composition: the composition/designation text describing what the product is made of, \
e.g. "RUM WITH NATURAL FLAVORS ADDED" or "Ale with Honey and Huckleberry Flavor". Extract it verbatim \
if visible; set present=false if none. Capture it here even if related text also appears in class_type \
-- these are not mutually exclusive; do not remove the designation from class_type.
- alcohol_content: value = the alcohol statement verbatim, e.g. "45% Alc./Vol. (90 Proof)"; \
abv_percent = the alcohol-by-volume number only as a number, e.g. 45.0 (null if none printed); \
proof = the proof number if printed, e.g. 90 (null otherwise).
- net_contents: as printed, e.g. "750 mL", "12 FL OZ".
- name_and_address: the name-and-address statement of the party RESPONSIBLE for the product in \
the U.S. -- for an IMPORTED product the importer statement (e.g. "IMPORTED BY: ABC IMPORTS INC. \
MIAMI, FL"); for a domestic product the bottler/producer statement (e.g. "DISTILLED AND BOTTLED \
BY OLD TOM DISTILLERY, BARDSTOWN, KY"). Look across the ENTIRE label for the relationship phrase \
(e.g. "BREWED & BOTTLED BY", "BOTTLED BY", "PRODUCED AND BOTTLED BY", "DISTILLED BY", "IMPORTED \
BY"), the company name, and the city/state. These pieces may be separated, stacked, curved, or \
printed in different parts of the same label -- combine them into ONE verbatim value in reading \
order. If the label shows MORE THAN ONE company statement, pick ONE for this field by this \
priority: (1) a statement whose phrase contains "IMPORTED" (e.g. "IMPORTED BY: ...") ALWAYS \
wins -- use it even when a producer/estate statement is larger, more prominent, or appears \
first; (2) otherwise the statement with a bottling/production phrase (e.g. "BOTTLED BY", \
"ESTATE BOTTLED BY", "PRODUCED AND BOTTLED BY"); (3) otherwise the bare company/address line. \
Transcribe EVERY other company statement as its own additional_statements item (kind \
"producer_statement") -- never merge two companies into this one field, and never drop a \
company statement entirely. If you can read the \
relationship phrase, the company name, OR the city/state, set present=true and transcribe what \
you can (use confidence "low" if only partial). Do not include URLs, website text, net contents, \
alcohol content, government warning text, slogans, marketing copy, trademark notices, barcode \
text, or unrelated statements in name_and_address.
- country_of_origin: the country-of-origin statement for imports, e.g. "PRODUCT OF SCOTLAND"; \
present=false for domestic product with no such statement.
- appellation: (WINE) the appellation of origin -- where the grapes were grown -- a country, \
state, county, American Viticultural Area (AVA), or a FOREIGN region/controlled designation, \
e.g. "Napa Valley", "Hudson River Region", "Sonoma County", "American", "Champagne", \
"Chianti", "Rioja", "Aglianico del Taburno". On imported wine the appellation is often the \
foreign region printed NEAR THE BRAND NAME (e.g. "CHAMPAGNE" under the brand) or inside the \
designation -- transcribe the printed region word(s) here too in that case. present=false if \
none is printed; for beer/spirits present=false.
- vintage: (WINE) the vintage year if a year is printed, e.g. "2018". present=false if no \
vintage year appears; for beer/spirits present=false.
- government_warning: the federal health warning.
    * text: transcribe the ENTIRE warning verbatim, INCLUDING the "GOVERNMENT WARNING:" header \
words at the start, with EXACT capitalization preserved (do not drop the header). Do NOT \
reconstruct the warning from memory or from the standard federal wording -- transcribe ONLY \
the visible printed words; if a word is unreadable, transcribe what you can and set confidence \
to "low" rather than inventing the expected text.
    * header_all_caps: looking at the printed header words "GOVERNMENT WARNING", are they in \
ALL CAPITAL LETTERS? true / false / null if not determinable.
    * header_bold: compare the visible stroke weight of the printed "GOVERNMENT WARNING" header \
with the warning body text IMMEDIATELY AFTER it. true if the header letter strokes are visibly \
thicker/heavier than that body text; false if they are the same weight, lighter, or not bold; \
null if you cannot compare the strokes (blur, glare, cropping, tiny text, or no body text to \
compare against). Do NOT infer bold from capitalization, font size, darkness, contrast, or \
expectation -- do not assume bold just because warning headers are usually bold. If uncertain, \
use null and low confidence.
    * header_bold_confidence: "high" / "medium" / "low" -- how sure you are of THIS bold judgment \
specifically (separate from your transcription confidence), based on how clearly you can see and \
compare the two stroke weights.
    * header_bold_basis: one short phrase describing what you ACTUALLY saw, e.g. "header strokes \
visibly thicker than body text" or "header same stroke weight as body"; null if not determinable.
    * body_bold: is the WARNING BODY TEXT itself (the sentences AFTER the header, beginning \
"(1) According to the Surgeon General...") bold? Judge the body's OWN letter strokes, separately \
from the header. true if the body strokes are visibly thick/heavy; false if the body is \
normal/regular weight; null if you cannot tell (blur, glare, tiny text). Do NOT infer from \
capitalization, size, darkness, contrast, or expectation. If uncertain, use null and low confidence.
    * body_bold_confidence: "high" / "medium" / "low" -- how sure you are of THIS body_bold judgment, \
based on how clearly you can see the body's letter strokes.
    Report what you SEE; do not judge compliance.
- sulfite_declaration: the sulfite disclosure text if printed, e.g. "CONTAINS SULFITES". Extract it \
verbatim if visible. Set present=false if none. Put it in THIS field -- do not bury it only inside \
additional_statements.
- additional_statements: array of any OTHER mandatory/disclosure text present that does NOT already \
have its own field above, each transcribed verbatim, e.g. "CONTAINS FD&C YELLOW #5", an age or \
commodity statement, or a secondary producer/bottler/estate statement that is NOT the responsible \
party's name_and_address (kind "producer_statement"). Optional "kind" hint per item. Empty array \
if none.
- image_quality_notes: a short note about any condition that hurt your reading (glare, shadow, \
rotation, low resolution, cropping); null if the image is clean.

Return EXACTLY this structure (use null where appropriate):
{
  "beverage_type": "beer|wine|spirits|unknown",
  "brand_name":        {"present": false, "value": null, "confidence": "low"},
  "fanciful_name":     {"present": false, "value": null, "confidence": "low"},
  "class_type":        {"present": false, "value": null, "confidence": "low"},
  "statement_of_composition": {"present": false, "value": null, "confidence": "low"},
  "alcohol_content":   {"present": false, "value": null, "abv_percent": null, "proof": null, "confidence": "low"},
  "net_contents":      {"present": false, "value": null, "confidence": "low"},
  "name_and_address":  {"present": false, "value": null, "confidence": "low"},
  "country_of_origin": {"present": false, "value": null, "confidence": "low"},
  "appellation":       {"present": false, "value": null, "confidence": "low"},
  "vintage":           {"present": false, "value": null, "confidence": "low"},
  "sulfite_declaration": {"present": false, "value": null, "confidence": "low"},
  "government_warning":{"present": false, "text": null, "header_all_caps": null, "header_bold": null, "header_bold_confidence": "low", "header_bold_basis": null, "body_bold": null, "body_bold_confidence": "low", "confidence": "low"},
  "additional_statements": [],
  "image_quality_notes": null
}"""

_client = None


def _get_client() -> OpenAI:
    """Lazily create the client so the API key can be set at app start-up."""
    global _client
    if _client is None:
        # max_retries=0 keeps REQUEST_TIMEOUT_SECONDS a real ceiling (SDK retries would
        # re-issue a timed-out request); param-rejection retries live in _create_with_fallbacks
        _client = OpenAI(timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)  # reads OPENAI_API_KEY from env
    return _client


def _normalize_images(images, media_type):
    """Accept a single bytes object, or a list of bytes / (bytes, media_type) tuples,
    and return a list of (bytes, media_type)."""
    if isinstance(images, (bytes, bytearray)):
        return [(bytes(images), media_type)]
    out = []
    for item in images:
        if isinstance(item, (bytes, bytearray)):
            out.append((bytes(item), media_type))
        else:
            data, mt = item
            out.append((bytes(data), mt or media_type))
    return out


def _image_blocks(images, media_type="image/png") -> list:
    """Base64-encode the images into content blocks, once -- shared by the main and
    supplement prompts. detail="high" so small back-label text is rendered at full
    fidelity rather than the 512px low-res tile."""
    blocks = []
    for img_bytes, mt in _normalize_images(images, media_type):
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        blocks.append({"type": "image_url",
                       "image_url": {"url": f"data:{mt};base64,{b64}", "detail": "high"}})
    return blocks


def _build_content(images, media_type="image/png", prompt=_PROMPT):
    """Build the user-message content: the prompt followed by one block per image."""
    return [{"type": "text", "text": prompt}] + _image_blocks(images, media_type)


# --- Structured Outputs schema ----------------------------------------------
# Strict JSON Schema mirroring the extraction contract: the API guarantees the response
# SHAPE, not the correctness of any observation inside it. _coerce still normalizes values
# and _parse_response still guards against truncated/empty/invalid responses.
_CONF_ENUM = {"type": "string", "enum": ["high", "medium", "low"]}
_SCALAR_FIELD_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"present": {"type": "boolean"}, "value": {"type": ["string", "null"]},
                   "confidence": _CONF_ENUM},
    "required": ["present", "value", "confidence"],
}
_EXTRACTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "beverage_type": {"type": "string", "enum": ["beer", "wine", "spirits", "unknown"]},
        "brand_name": _SCALAR_FIELD_SCHEMA, "fanciful_name": _SCALAR_FIELD_SCHEMA,
        "class_type": _SCALAR_FIELD_SCHEMA, "statement_of_composition": _SCALAR_FIELD_SCHEMA,
        "net_contents": _SCALAR_FIELD_SCHEMA, "name_and_address": _SCALAR_FIELD_SCHEMA,
        "country_of_origin": _SCALAR_FIELD_SCHEMA, "sulfite_declaration": _SCALAR_FIELD_SCHEMA,
        "appellation": _SCALAR_FIELD_SCHEMA, "vintage": _SCALAR_FIELD_SCHEMA,
        "alcohol_content": {
            "type": "object", "additionalProperties": False,
            "properties": {"present": {"type": "boolean"}, "value": {"type": ["string", "null"]},
                           "abv_percent": {"type": ["number", "null"]},
                           "proof": {"type": ["number", "null"]}, "confidence": _CONF_ENUM},
            "required": ["present", "value", "abv_percent", "proof", "confidence"],
        },
        "government_warning": {
            "type": "object", "additionalProperties": False,
            "properties": {"present": {"type": "boolean"}, "text": {"type": ["string", "null"]},
                           "header_all_caps": {"type": ["boolean", "null"]},
                           "header_bold": {"type": ["boolean", "null"]},
                           "header_bold_confidence": _CONF_ENUM,
                           "header_bold_basis": {"type": ["string", "null"]},
                           "body_bold": {"type": ["boolean", "null"]},
                           "body_bold_confidence": _CONF_ENUM,
                           "confidence": _CONF_ENUM},
            "required": ["present", "text", "header_all_caps", "header_bold",
                         "header_bold_confidence", "header_bold_basis",
                         "body_bold", "body_bold_confidence", "confidence"],
        },
        "additional_statements": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False,
                      "properties": {"value": {"type": "string"},
                                     "kind": {"type": ["string", "null"]}, "confidence": _CONF_ENUM},
                      "required": ["value", "kind", "confidence"]},
        },
        "image_quality_notes": {"type": ["string", "null"]},
    },
    "required": ["beverage_type", "brand_name", "fanciful_name", "class_type",
                 "statement_of_composition", "net_contents", "name_and_address", "country_of_origin",
                 "sulfite_declaration", "appellation", "vintage", "alcohol_content",
                 "government_warning", "additional_statements", "image_quality_notes"],
}
_STRUCTURED_RF = {"type": "json_schema",
                  "json_schema": {"name": "label_extraction", "strict": True,
                                  "schema": _EXTRACTION_SCHEMA}}
_JSON_OBJECT_RF = {"type": "json_object"}   # fallback for models without Structured Outputs

# --- Warning-only supplement (WARNING_SUPPLEMENT_MODEL) -----------------------
# A second, cross-family model reads ONLY the government warning, in parallel with the main
# extraction. Same wording as _PROMPT's government_warning section plus one deliberate,
# load-bearing divergence -- the all-bold clarification (on all-bold labels the comparative
# bold question is literally answerable "no" even though the header IS bold) -- do NOT
# "fix" the prompts back into sync. Rationale: config.WARNING_SUPPLEMENT_MODEL.
_WARNING_PROMPT = """You are a transcription assistant for U.S. TTB alcohol-beverage label \
review. You are shown one OR MORE photos of a SINGLE alcohol beverage product. Find the \
federal health warning statement (the government warning), wherever it appears across the \
images. Return ONLY a JSON object in exactly this shape:
{"present": true|false, "text": "..."|null, "header_all_caps": true|false|null, \
"header_bold": true|false|null, "header_bold_confidence": "high"|"medium"|"low", \
"header_bold_basis": "..."|null, "body_bold": true|false|null, "body_bold_confidence": \
"high"|"medium"|"low", "confidence": "high"|"medium"|"low"}
- text: transcribe the ENTIRE warning verbatim, INCLUDING the "GOVERNMENT WARNING:" header \
words at the start, with EXACT capitalization preserved (do not drop the header). End the \
transcription at the END of the warning statement itself -- do NOT include adjacent label \
text that is not part of the warning (e.g. "CONTAINS SULFITES", net contents, or other \
statements printed near it). Do NOT reconstruct the warning from memory or from the \
standard federal wording -- transcribe ONLY the visible printed words; if a word is \
unreadable, transcribe what you can and set confidence to "low" rather than inventing the \
expected text.
- header_all_caps: looking at the printed header words "GOVERNMENT WARNING", are they in ALL \
CAPITAL LETTERS? true / false / null if not determinable.
- header_bold: compare the visible stroke weight of the printed "GOVERNMENT WARNING" header \
with the warning body text IMMEDIATELY AFTER it. true if the header letter strokes are visibly \
thicker/heavier than that body text; false if they are the same weight, lighter, or not bold; \
null if you cannot compare the strokes (blur, glare, cropping, tiny text, or no body text to \
compare against). If BOTH the header and the body letter strokes appear heavy/bold, report \
header_bold true -- the header IS in bold type even though it is not bolder than the body. \
Do NOT infer bold from capitalization, font size, darkness, contrast, or expectation -- do \
not assume bold just because warning headers are usually bold. If uncertain, use null and \
low confidence.
- header_bold_confidence: "high" / "medium" / "low" -- how sure you are of THIS bold judgment \
specifically, based on how clearly you can see and compare the stroke weights.
- header_bold_basis: one short phrase describing what you ACTUALLY saw; null if not determinable.
- body_bold: is the WARNING BODY TEXT itself (the sentences AFTER the header) bold? Judge the \
body's OWN letter strokes, separately from the header. true / false / null. Do NOT infer from \
capitalization, size, darkness, contrast, or expectation. If uncertain, use null.
- body_bold_confidence: "high" / "medium" / "low".
- confidence: how sure you are of YOUR transcription overall, based on image clarity.
Report what you SEE; do not judge compliance. If no government warning is visible anywhere \
in the images, set present=false and null values."""

# The supplement returns the extraction contract's government_warning shape, so it reuses
# that subschema and _coerce_warning.
_WARNING_RF = {"type": "json_schema",
               "json_schema": {"name": "warning_check", "strict": True,
                               "schema": _EXTRACTION_SCHEMA["properties"]["government_warning"]}}

# How long extract_fields waits for the supplement AFTER the main call returns. The
# supplement is normally faster than the main call, so this only bites when it hangs --
# and then we fall back to the main read rather than stretch the request.
_SUPPLEMENT_WAIT_SECONDS = 10


def _model_params(model: str) -> dict:
    """Per-model request params. Uses Structured Outputs (strict json_schema) so the response
    SHAPE is guaranteed by the API, not just by _coerce. The gpt-5/o-series reasoning models
    reject `max_tokens` and a non-default `temperature`, so they use `max_completion_tokens`
    and low/minimal reasoning; the gpt-4 family keeps `temperature=0`. Models that don't
    support Structured Outputs fall back to json_object in the caller."""
    params = {
        "model": model,
        "response_format": _STRUCTURED_RF,
        "max_completion_tokens": 3000,  # headroom for high-detail multi-image reads
    }
    if model.startswith(("gpt-5.4", "gpt-5.5")):
        params["reasoning_effort"] = "low"   # gpt-5.4/5.5 reject 'minimal'; 'low' is their floor
    elif model.startswith("gpt-5"):
        params["reasoning_effort"] = "minimal"
    elif model.startswith(("o1", "o3", "o4")):
        params["reasoning_effort"] = "low"   # the o-series rejects 'minimal'; 'low' is its floor
    else:
        params["temperature"] = 0
    return params


def _create_with_fallbacks(client, content, params):
    """chat.completions.create with retries on known per-model param rejections (no strict
    Structured Outputs -> json_object; reasoning_effort 'minimal' rejected -> 'low'). Each
    retry adjusts one param, so the loop converges in a couple of attempts. 429s get a
    short bounded backoff (they return immediately, so this does not undermine the
    REQUEST_TIMEOUT_SECONDS ceiling). Any other error is re-raised."""
    last = None
    rate_limit_tries = 0
    for _ in range(3 + RATE_LIMIT_MAX_RETRIES):
        try:
            return client.chat.completions.create(
                messages=[{"role": "user", "content": content}], **params)
        except RateLimitError as exc:
            last = exc
            # insufficient_quota is also an HTTP 429 but PERMANENT -- never retry it
            if _is_quota_error(exc) or rate_limit_tries >= RATE_LIMIT_MAX_RETRIES:
                raise
            rate_limit_tries += 1
            delay = 2.0 * rate_limit_tries   # 2s, 4s: enough for a burst 429 to clear
            try:    # honor the server's Retry-After when it asks for longer (capped for UX)
                delay = min(max(delay, float(exc.response.headers.get("retry-after", 0))), 15.0)
            except Exception:
                pass
            time.sleep(delay)
        except Exception as exc:
            last = exc
            msg = str(exc)
            if ("json_schema" in msg or "response_format" in msg) and params.get("response_format") != _JSON_OBJECT_RF:
                params["response_format"] = _JSON_OBJECT_RF   # Structured Outputs unsupported
                logging.getLogger(__name__).warning(
                    "Structured Outputs unsupported by %s; fell back to json_object "
                    "(response shape is now enforced by _coerce, not the API).",
                    params.get("model"))
            elif "reasoning_effort" in msg and params.get("reasoning_effort") == "minimal":
                params["reasoning_effort"] = "low"            # model rejects 'minimal'
            else:
                raise
    raise last


def _is_quota_error(exc) -> bool:
    """True for the permanent out-of-credits 429 (code 'insufficient_quota'): the SDK raises
    it as RateLimitError exactly like a transient burst 429, but no retry can clear it."""
    return (getattr(exc, "code", None) == "insufficient_quota"
            or "insufficient_quota" in str(exc))


def failure_kind(exc) -> str:
    """Coarse classification of an extraction failure so the UI can give accurate guidance
    instead of blaming every error on the photo: 'auth', 'quota' (out of credits),
    'rate_limit', 'timeout', 'connection', 'bad_response', or 'unknown'. Timeout is checked
    before connection because APITimeoutError subclasses APIConnectionError; the api_key
    string check catches the client-constructor error when no key is configured at all."""
    if isinstance(exc, AuthenticationError):
        return "auth"
    if isinstance(exc, RateLimitError):
        return "quota" if _is_quota_error(exc) else "rate_limit"
    if isinstance(exc, APITimeoutError):
        return "timeout"
    if isinstance(exc, APIConnectionError):
        return "connection"
    if isinstance(exc, ExtractionError):
        return "bad_response"
    if "api_key" in str(exc).lower():
        return "auth"
    return "unknown"


def _extract_warning_supplement(client, image_blocks) -> dict:
    """Run the warning-only supplement read and return the coerced government_warning-shaped
    dict. ``image_blocks`` is the already-encoded list shared with the main call. Raises on
    any failure -- the caller treats that as 'supplement unavailable'."""
    content = [{"type": "text", "text": _WARNING_PROMPT}] + image_blocks
    params = _model_params(WARNING_SUPPLEMENT_MODEL)
    # patched AFTER _model_params: _create_with_fallbacks mutates this dict in place on
    # its retries, so both patches survive the json_object / reasoning-effort fallbacks
    params["response_format"] = _WARNING_RF
    params["max_completion_tokens"] = 1000   # the warning block only, not the full schema
    payload = _parse_json_payload(_create_with_fallbacks(client, content, params))
    return _coerce_warning(payload)


def _apply_warning_supplement(extracted: dict, supp: dict | None) -> dict:
    """Merge the supplement's warning read into the extraction's government_warning: the
    supplement's fields become THE warning the verifier judges, and the main model's read
    moves to the main_* keys as evidence. On a failed supplement (supp=None) the main read
    stays in place; warning_observer records "supplement" vs "main-fallback". Mutates and
    returns ``extracted``; the merge is not idempotent, so an already-merged read
    (warning_observer set) is returned untouched."""
    gw = extracted.get("government_warning")
    if not isinstance(gw, dict):
        return extracted
    if gw.get("warning_observer") is not None:   # already merged -- keep the evidence intact
        return extracted
    if supp is None:
        gw["warning_observer"] = "main-fallback"
        return extracted
    gw["main_present"] = gw.get("present")
    gw["main_text"] = gw.get("text")
    gw["main_header_all_caps"] = gw.get("header_all_caps")
    gw["main_header_bold"] = gw.get("header_bold")
    gw["main_header_bold_confidence"] = gw.get("header_bold_confidence")
    gw["main_header_bold_basis"] = gw.get("header_bold_basis")
    gw["main_body_bold"] = gw.get("body_bold")
    gw["main_body_bold_confidence"] = gw.get("body_bold_confidence")
    gw.update(supp)
    gw["warning_observer"] = "supplement"
    return extracted


def extract_fields(images, media_type: str = "image/png") -> dict:
    """Return the structured label fields extracted from the image(s), coerced to schema.

    ``images`` may be a single bytes object or a list of bytes / (bytes, media_type)
    tuples -- e.g. the front and back labels of the SAME product, read together.

    When WARNING_SUPPLEMENT_MODEL is set, a warning-only read by that model runs IN
    PARALLEL with the main extraction and is merged in by _apply_warning_supplement.
    A supplement failure NEVER fails the extraction -- the main read is used with a
    fallback marker."""
    client = _get_client()   # created before threading so both calls share one client
    image_blocks = _image_blocks(images, media_type)   # encoded once, shared by both calls
    content = [{"type": "text", "text": _PROMPT}] + image_blocks
    params = _model_params(EXTRACTION_MODEL)

    if not WARNING_SUPPLEMENT_MODEL:
        return _parse_response(_create_with_fallbacks(client, content, params))

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        supp_future = executor.submit(_extract_warning_supplement, client, image_blocks)
        extracted = _parse_response(_create_with_fallbacks(client, content, params))
        try:
            supp = supp_future.result(timeout=_SUPPLEMENT_WAIT_SECONDS)
        except Exception as exc:  # noqa: BLE001 -- ANY supplement failure is non-fatal
            logging.getLogger(__name__).warning(
                "warning supplement (%s) unavailable (%s); judging the warning from the "
                "main read", WARNING_SUPPLEMENT_MODEL, failure_kind(exc))
            supp = None
        return _apply_warning_supplement(extracted, supp)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _parse_json_payload(response) -> dict:
    """Decode the model response, raising ExtractionError on a truncated/empty/invalid one --
    a bad response must NOT be silently coerced into an 'all-absent' object (that would
    surface as a confident wrong verdict instead of a read error)."""
    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        raise ExtractionError("the model response was cut off at the token limit")
    content = choice.message.content
    if not content:
        raise ExtractionError("the model returned an empty response")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"the model did not return valid JSON: {exc}") from exc


def _parse_response(response) -> dict:
    """Turn an OpenAI response into the coerced schema, or raise ExtractionError."""
    return _coerce(_parse_json_payload(response))


# --- Defensive coercion ------------------------------------------------------
# Normalize whatever the model returns into the exact schema with safe defaults.
_CONF = {"high", "medium", "low"}


def _conf(value) -> str:
    return value if isinstance(value, str) and value in _CONF else "low"


def _field(raw, extra=None) -> dict:
    """Normalize one {present, value, confidence, ...} field object."""
    d = raw if isinstance(raw, dict) else {}
    value = d.get("value")
    if value is not None and not isinstance(value, str):
        value = str(value)
    present = d.get("present")
    if not isinstance(present, bool):
        present = value is not None
    out = {"present": present, "value": value, "confidence": _conf(d.get("confidence"))}
    for key, default in (extra or {}).items():
        out[key] = d.get(key, default)
    return out


def _num(x):
    """Best-effort numeric coercion; returns float or None."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = re.search(r"\d+(?:\.\d+)?", x)
        if m:
            return float(m.group())
    return None


def _bool_or_none(x):
    return x if isinstance(x, bool) else None


_BEVERAGE_SYNONYMS = {
    "beer": "beer", "malt": "beer", "malt beverage": "beer", "ale": "beer", "lager": "beer",
    "wine": "wine",
    "spirits": "spirits", "spirit": "spirits", "distilled spirits": "spirits",
    "distilled spirit": "spirits", "liquor": "spirits",
    "unknown": "unknown",
}


def _normalize_beverage_type(value) -> str:
    """Casefold + map common synonyms so 'Spirits'/'malt'/'Distilled Spirits' don't fall
    through to 'unknown' and weaken the class-specific ABV rule."""
    if not isinstance(value, str):
        return "unknown"
    return _BEVERAGE_SYNONYMS.get(value.strip().casefold(), "unknown")


def _coerce_warning(gw) -> dict:
    """Normalize a government_warning object (the main extraction and the supplement
    return the same shape); feeds verification._check_warning."""
    gw = gw if isinstance(gw, dict) else {}
    text = gw.get("text")
    if text is not None and not isinstance(text, str):
        text = str(text)
    present = gw.get("present")
    if not isinstance(present, bool):
        present = bool(text)
    bold_basis = gw.get("header_bold_basis")
    return {
        "present": present,
        "text": text,
        "header_all_caps": _bool_or_none(gw.get("header_all_caps")),
        "header_bold": _bool_or_none(gw.get("header_bold")),
        "header_bold_confidence": _conf(gw.get("header_bold_confidence")),
        "header_bold_basis": str(bold_basis) if bold_basis else None,
        "body_bold": _bool_or_none(gw.get("body_bold")),
        "body_bold_confidence": _conf(gw.get("body_bold_confidence")),
        "confidence": _conf(gw.get("confidence")),
    }


def _coerce(raw: dict) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    bev = _normalize_beverage_type(raw.get("beverage_type"))

    out = {"beverage_type": bev}
    for name in _SCALAR_FIELDS:
        out[name] = _field(raw.get(name))

    ac = _field(raw.get("alcohol_content"), extra={"abv_percent": None, "proof": None})
    ac["abv_percent"] = _num(ac.get("abv_percent"))
    ac["proof"] = _num(ac.get("proof"))
    out["alcohol_content"] = ac

    out["government_warning"] = _coerce_warning(raw.get("government_warning"))

    clean = []
    stmts = raw.get("additional_statements")
    if isinstance(stmts, list):
        for s in stmts:
            if isinstance(s, dict) and s.get("value"):
                clean.append({"value": str(s["value"]), "kind": s.get("kind"),
                              "confidence": _conf(s.get("confidence"))})
            elif isinstance(s, str) and s.strip():
                clean.append({"value": s.strip(), "kind": None, "confidence": "low"})
    out["additional_statements"] = clean

    iqn = raw.get("image_quality_notes")
    out["image_quality_notes"] = str(iqn) if iqn else None
    return out
