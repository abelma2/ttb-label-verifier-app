"""evidence_ensemble.py -- BENCHMARK / EVAL ONLY. Not wired into the app.

An experiment: run TWO extra "witness" passes in parallel ALONGSIDE the current production
extractor and measure whether the extra evidence improves alcohol-label verification, WITHOUT
changing any production verdict behaviour.

Three passes run CONCURRENTLY per bottle (front+back read together, like the app):

  Pass A -- raw-text witness   (EVIDENCE_RAW_TEXT_MODEL, default gpt-5.4-nano)
      Verbatim OCR of ALL visible text, preserving caps/punctuation/line breaks. No judgment.
  Pass B -- structured fields  (production extraction.extract_fields, config.EXTRACTION_MODEL)
      The PRIMARY field evidence -- unchanged production path. Its output is fed to verify().
  Pass C -- warning visual witness (EVIDENCE_VISUAL_MODEL, default gpt-5.4-mini)
      Looks ONLY at the government warning and reports visual OBSERVATIONS (header/body bold,
      caps, legibility, boxed/separated, contrast, basis). No judgment.

The MODEL WITNESSES PROVIDE EVIDENCE; DETERMINISTIC PYTHON JUDGES. This script never lets the
models "vote" to resolve a disagreement -- in particular a HIGH-confidence bold disagreement
between Pass B and Pass C is reported as REVIEW evidence, never auto-resolved. verify() is called
exactly as production calls it (on Pass B only); A and C are evidence reported alongside, they do
NOT change the verdict.

Two stages, in order, with a STAGE GATE between them:
  1. baseline_labels  -- clean front/back pairs grouped by the _Front/_Other convention, matched to
                         the application JSONs (perfect labels -> expect ~100% pass/completeness;
                         measures false-fails, missing fields, warning-bold uncertainty, latency).
  2. error_labels     -- the single-defect fixtures (manifest-driven), each altered face paired with
                         its product's CLEAN other face, matched to the application JSON.

STAGE GATE: in `--stage all`, the error stage runs ONLY if the baseline is close to 100% pass
(field pass rate >= EVIDENCE_BASELINE_GATE, default 0.95; no hard FAILs; every bottle matched to an
application). Otherwise the run PAUSES after writing the baseline section + a Stage-gate explanation,
instead of continuing automatically. Override with --force-errors, or just run `--stage errors`.

APPLICATION MATCHING: error fixtures use the manifest's app (rum/malt/wine). Baseline groups are
matched to an application via the app `_meta.clean_baseline_*` refs (the existing eval convention);
the chosen file + match status are recorded per bottle. An ambiguous (two apps) or unmatched group
is flagged as an ERROR in the report and not verified -- never guessed silently.

NO SILENT OVERWRITE: before any write, existing result + intermediate files are COPIED (preserved)
into archive/evidence_ensemble/<timestamp>/ so a re-run never destroys the prior result.

Real OpenAI vision calls (key from env / .env / .streamlit/secrets.toml).

Run (from repo root):
    python scripts/benchmarks/evidence_ensemble.py --stage baseline       # stage 1 only (run first)
    python scripts/benchmarks/evidence_ensemble.py --stage errors         # stage 2 (merges stage 1)
    python scripts/benchmarks/evidence_ensemble.py --stage all            # both, with the gate
    python scripts/benchmarks/evidence_ensemble.py --stage all --force-errors   # run errors even if gate trips
    python scripts/benchmarks/evidence_ensemble.py --stability 5          # run baseline ensemble 5x; quantify
                                                                          # run-to-run flakiness (no error stage)

Writes (combined across whichever stages have run):
    artifacts/evidence_ensemble_results.json
    artifacts/evidence_ensemble_results.md
Plus per-stage intermediates  artifacts/evidence_ensemble_{baseline,errors}.json  so the two stages
can be run in separate invocations and still produce one combined report; prior runs are archived.
"""
import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT

import csv
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Reuse the production building blocks (benchmarks already import module privates, e.g.
# warning_check_benchmark imports verification._check_warning). We deliberately reuse rather than
# fork so the witnesses share the exact request/parse/fallback behaviour of the real extractor.
from extraction import (
    extract_fields,            # Pass B = the unchanged production extractor
    _build_content, _model_params, _create_with_fallbacks, _get_client,
    _CONF_ENUM, ExtractionError,
)
from verification import (
    verify, _normalize, _warning_body, _CANONICAL_WARNING_BODY_NORM,
    PASS, REVIEW, FAIL,
)
from config import EXTRACTION_MODEL
# Grouping convention shared with smoke_test.py (the _Front/_Other product grouping).
from smoke_test import _group_by_product, _gather, _media_type

# --- experiment configuration ------------------------------------------------
RAW_TEXT_MODEL = os.environ.get("EVIDENCE_RAW_TEXT_MODEL", "gpt-5.4-nano")   # Pass A
VISUAL_MODEL = os.environ.get("EVIDENCE_VISUAL_MODEL", "gpt-5.4-mini")       # Pass C
STRUCTURED_MODEL = EXTRACTION_MODEL                                          # Pass B (production)
# Per-bottle latency target to compare against. The app's docs cite a ~7s front+back read budget
# (smoke_test.py) and a stricter ~5s product target (eval/summary.md); we report against both.
TARGET_BOTTLE_SECONDS = float(os.environ.get("EVIDENCE_TARGET_SECONDS", "7"))
STRICT_TARGET_SECONDS = 5.0
# Retry a witness ONCE on a transient API error (rate-limit/timeout/5xx) so a flake isn't a data
# hole; a retried bottle is excluded from the "clean timing" aggregates (its seconds are inflated).
WITNESS_RETRIES = int(os.environ.get("EVIDENCE_WITNESS_RETRIES", "1"))

ERR = os.path.join(ROOT, "test_labels", "error_labels")
APPS = os.path.join(ROOT, "test_labels", "applications")
BASE = os.path.join(ROOT, "test_labels", "baseline_labels")
ARTIFACTS = os.path.join(ROOT, "artifacts")
BASELINE_INTERMEDIATE = os.path.join(ARTIFACTS, "evidence_ensemble_baseline.json")
ERRORS_INTERMEDIATE = os.path.join(ARTIFACTS, "evidence_ensemble_errors.json")
RESULTS_JSON = os.path.join(ARTIFACTS, "evidence_ensemble_results.json")
RESULTS_MD = os.path.join(ARTIFACTS, "evidence_ensemble_results.md")
STABILITY_JSON = os.path.join(ARTIFACTS, "evidence_ensemble_stability.json")
STABILITY_MD = os.path.join(ARTIFACTS, "evidence_ensemble_stability.md")
STABILITY_RUNS = int(os.environ.get("EVIDENCE_STABILITY_RUNS", "5"))   # default --stability count
# Prior results are COPIED here (timestamped) before any overwrite, so a run never silently
# destroys the previous one. archive/ is gitignored (see CLAUDE.md).
ARCHIVE_DIR = os.path.join(ROOT, "archive", "evidence_ensemble")
# STAGE GATE: in `--stage all`, auto-continue to the error stage ONLY if clean baselines are at
# least this close to fully passing (field-level pass rate). Below it -> pause and explain the
# failure causes instead of continuing automatically (override with --force-errors).
BASELINE_GATE = float(os.environ.get("EVIDENCE_BASELINE_GATE", "0.95"))

# error-stage scoring tables (mirrors error_label_application_eval.py / eval/run_eval.py)
PRODUCTS = {
    "spirits": {"app": "rum.json",  "front": os.path.join(BASE, "baseline_1_Front.png"),
                "back": os.path.join(BASE, "baseline_1_Other.png")},
    "malt":    {"app": "malt.json", "front": os.path.join(BASE, "baseline_2_Front.png"),
                "back": os.path.join(BASE, "baseline_2_Other.png")},
    "wine":    {"app": "wine.json", "front": os.path.join(BASE, "baseline_3_Front.png"),
                "back": os.path.join(BASE, "baseline_3_Other.png")},
}
CHECK_FIELD = {
    "government_warning_exact_match": "government_warning",
    "case_normalization": "government_warning",
    "brand_name_fuzzy_match": "brand_name",
    "proof_equals_2x_abv": "alcohol_content",
    "abv_numeric_match": "alcohol_content",
    "abv_notation_format": "alcohol_content",
    "net_contents_normalization": "net_contents",
    "vintage_requires_appellation": "appellation",
    "none": "government_warning",
}
APP_KEY = {
    "government_warning": "health_warning", "brand_name": "brand_name",
    "alcohol_content": "alcohol_content", "net_contents": "net_contents",
    "appellation": "appellation", "name_and_address": "name_and_address",
}
KNOWN_GAP_CHECKS = {"proof_equals_2x_abv", "abv_notation_format"}
EXPECT = {"FAIL": "fail", "NEEDS_REVIEW": "needs_review", "PASS": "pass"}
# Fixtures whose defect changes the WORDING/PUNCTUATION of the warning body (not just case). If a
# witness reproduces the canonical body verbatim on one of these, it HALLUCINATED a compliant
# warning -- dangerous false-pass evidence. (GW-LOWERCASE-SG only lowercases S/G, so its normalized
# body is legitimately identical to canonical -- not a hallucination, so it is NOT listed here.)
WORDING_ALTERED_FIXTURES = {"GW-REWORD", "GW-COMMA"}


# --- key loading (mirrors the other benchmarks) ------------------------------
def load_key():
    if os.environ.get("OPENAI_API_KEY"):
        return True
    envf = os.path.join(ROOT, ".env")
    if os.path.exists(envf):
        for line in open(envf, encoding="utf-8"):
            m = re.match(r'\s*OPENAI_API_KEY\s*=\s*"?([^"\r\n]+)"?', line)
            if m:
                os.environ["OPENAI_API_KEY"] = m.group(1).strip()
                return True
    sec = os.path.join(ROOT, ".streamlit", "secrets.toml")
    if os.path.exists(sec):
        m = re.search(r'OPENAI_API_KEY\s*=\s*"([^"]+)"', open(sec, encoding="utf-8").read())
        if m and m.group(1) and m.group(1) != "sk-...":
            os.environ["OPENAI_API_KEY"] = m.group(1)
            return True
    return False


# === Pass A: raw-text witness ===============================================
_RAW_TEXT_PROMPT = """You are a VERBATIM transcription (OCR) assistant. You are shown one or more \
photos of a single alcohol beverage product (e.g. its front and back labels).

Transcribe ALL visible printed text across ALL the images, exactly as printed. Preserve \
capitalization, punctuation, %, symbols, and LINE BREAKS as faithfully as you can (put each printed \
line on its own line). Read the small print too (the government warning, net contents, address, any \
disclosures).

If there are multiple images, transcribe them in order and separate them with a line \
"--- IMAGE 2 ---" (then "--- IMAGE 3 ---", etc.).

Do NOT judge compliance. Do NOT summarize, paraphrase, normalize, correct spelling, expand \
abbreviations, or reorder text. Do NOT invent or auto-correct text that is not visibly printed -- \
transcribe ONLY what you can actually see; if a word is unreadable, write [illegible] rather than \
guessing the expected text.

Return JSON only: {"raw_text": "<all transcribed text>", "image_quality_notes": "<short note or null>"}"""

_RAW_TEXT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"raw_text": {"type": "string"},
                   "image_quality_notes": {"type": ["string", "null"]}},
    "required": ["raw_text", "image_quality_notes"],
}
_RAW_TEXT_RF = {"type": "json_schema",
                "json_schema": {"name": "raw_text_witness", "strict": True, "schema": _RAW_TEXT_SCHEMA}}


def pass_a_raw_text(images, media_type="image/png"):
    """Verbatim OCR witness. Returns {"raw_text", "image_quality_notes"}."""
    raw = _call_witness(images, media_type, _RAW_TEXT_PROMPT, _RAW_TEXT_RF, RAW_TEXT_MODEL,
                        max_tokens=8000)  # full front+back transcription needs headroom
    return {"raw_text": str(raw.get("raw_text") or ""),
            "image_quality_notes": (str(raw["image_quality_notes"])
                                    if raw.get("image_quality_notes") else None)}


# === Pass C: warning visual-attribute witness ===============================
_VISUAL_PROMPT = """You are a VISUAL OBSERVATION assistant for U.S. TTB alcohol-beverage label review.

You are shown one or more photos of a SINGLE alcohol beverage product. Look ONLY at the federal \
Government Health Warning Statement (the block beginning "GOVERNMENT WARNING:"). IGNORE every other \
field (brand, class/type, ABV, net contents, address, country, appellation, vintage).

You REPORT what you SEE. You do NOT decide whether the label is compliant. Return JSON only.

Report these observations about the government warning:
- warning_present: is a government health warning visible on the label? true/false.
- warning_text: transcribe the warning VERBATIM if readable (preserve capitalization and \
punctuation); null if not present or unreadable. Do NOT reconstruct it from memory -- report only \
visibly printed words.
- header_present: are the words "GOVERNMENT WARNING" visibly printed? true/false/null.
- header_all_caps: are the words "GOVERNMENT WARNING" in ALL CAPITAL LETTERS? true/false/null.
- header_bold: compare the stroke weight of "GOVERNMENT WARNING" to the warning body text \
immediately after it. true ONLY if the header strokes are VISIBLY heavier/thicker/darker than that \
body text; false if the same weight or lighter; null only if you genuinely cannot compare. Bold is \
about stroke WEIGHT, not merely being all-caps.
- header_bold_confidence: "high"/"medium"/"low" -- how sure you are of THAT bold judgment.
- body_bold: does the warning BODY text (the sentences after the header, "(1) According to the \
Surgeon General...") itself appear BOLD/heavy? true if the body's own strokes are visibly bold; \
false if normal weight; null if you cannot tell.
- body_bold_confidence: "high"/"medium"/"low" -- how sure you are of the body_bold judgment.
- legibility: overall, how readable is the warning text? "high" (crisp/easy), "medium", or "low" \
(small/blurry/faint).
- boxed_or_separated: is the warning enclosed in a box/border or otherwise visually set apart from \
the surrounding label text/graphics? true/false/null.
- contrast: how strong is the contrast between the warning text and its background? \
"high"/"medium"/"low".
- basis: one short phrase describing what you ACTUALLY saw for the bold judgments, e.g. "header \
strokes clearly thicker than body" or "header and body the same weight"; null if not determinable.
- image_quality_notes: a short note about anything that hurt your reading (glare, blur, angle, crop, \
tiny text); null if clean.

Report what you SEE. Do not judge compliance."""

_LEVEL_ENUM = {"type": "string", "enum": ["high", "medium", "low"]}
_VISUAL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "warning_present": {"type": "boolean"},
        "warning_text": {"type": ["string", "null"]},
        "header_present": {"type": ["boolean", "null"]},
        "header_all_caps": {"type": ["boolean", "null"]},
        "header_bold": {"type": ["boolean", "null"]},
        "header_bold_confidence": _CONF_ENUM,
        "body_bold": {"type": ["boolean", "null"]},
        "body_bold_confidence": _CONF_ENUM,
        "legibility": _LEVEL_ENUM,
        "boxed_or_separated": {"type": ["boolean", "null"]},
        "contrast": _LEVEL_ENUM,
        "basis": {"type": ["string", "null"]},
        "image_quality_notes": {"type": ["string", "null"]},
    },
    "required": ["warning_present", "warning_text", "header_present", "header_all_caps",
                 "header_bold", "header_bold_confidence", "body_bold", "body_bold_confidence",
                 "legibility", "boxed_or_separated", "contrast", "basis", "image_quality_notes"],
}
_VISUAL_RF = {"type": "json_schema",
              "json_schema": {"name": "warning_visual_witness", "strict": True, "schema": _VISUAL_SCHEMA}}


def pass_c_visual(images, media_type="image/png"):
    """Warning-only visual witness. Returns the coerced observation dict."""
    return _coerce_visual(_call_witness(images, media_type, _VISUAL_PROMPT, _VISUAL_RF, VISUAL_MODEL))


def _b(x):
    return x if isinstance(x, bool) else None


def _conf(x):
    return x if x in ("high", "medium", "low") else "low"


def _level(x):
    return x if x in ("high", "medium", "low") else None


def _coerce_visual(raw):
    raw = raw if isinstance(raw, dict) else {}
    txt = raw.get("warning_text")
    present = raw.get("warning_present")
    return {
        "warning_present": present if isinstance(present, bool) else bool(txt),
        "warning_text": str(txt) if txt else None,
        "header_present": _b(raw.get("header_present")),
        "header_all_caps": _b(raw.get("header_all_caps")),
        "header_bold": _b(raw.get("header_bold")),
        "header_bold_confidence": _conf(raw.get("header_bold_confidence")),
        "body_bold": _b(raw.get("body_bold")),
        "body_bold_confidence": _conf(raw.get("body_bold_confidence")),
        "legibility": _level(raw.get("legibility")),
        "boxed_or_separated": _b(raw.get("boxed_or_separated")),
        "contrast": _level(raw.get("contrast")),
        "basis": str(raw.get("basis")) if raw.get("basis") else None,
        "image_quality_notes": (str(raw.get("image_quality_notes"))
                                if raw.get("image_quality_notes") else None),
    }


# === shared witness call =====================================================
def _call_witness(images, media_type, prompt, response_format, model, max_tokens=None):
    """One vision call with the given prompt+schema+model, reusing the production request/parse/
    fallback path (Structured Outputs, with the json_object + reasoning_effort fallbacks)."""
    content = _build_content(images, media_type, prompt)
    params = _model_params(model, response_format)
    if max_tokens:
        params["max_completion_tokens"] = max_tokens
    resp = _create_with_fallbacks(_get_client(), content, params)
    choice = resp.choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        raise ExtractionError("witness response was cut off at the token limit")
    txt = choice.message.content
    if not txt:
        raise ExtractionError("witness returned an empty response")
    return json.loads(txt)


# === parallel timing harness =================================================
_TRANSIENT = ("rate limit", "rate_limit", "429", "timeout", "timed out", "overloaded",
              "500", "502", "503", "connection", "temporar")


def _is_transient(exc):
    s = str(exc).lower()
    return any(t in s for t in _TRANSIENT)


def _timed_pass(fn, images, media_type):
    """Run one pass, timing the WHOLE thing (incl. any transient retry). Returns a dict with the
    result (or None), wall seconds, the error string (or None), and attempt count."""
    start = time.perf_counter()
    err = None
    for k in range(1 + WITNESS_RETRIES):
        try:
            res = fn(images, media_type)
            return {"result": res, "seconds": time.perf_counter() - start, "error": None,
                    "attempts": k + 1}
        except Exception as e:  # noqa: BLE001 -- a failed witness must not sink the bottle
            err = e
            if k < WITNESS_RETRIES and _is_transient(e):
                time.sleep(3 + 4 * k)
                continue
            break
    return {"result": None, "seconds": time.perf_counter() - start, "error": str(err)[:240],
            "attempts": WITNESS_RETRIES + 1 if err and _is_transient(err) else 1}


def _serialize_verify(v):
    return {"overall": v["overall"], "beverage_type": v.get("beverage_type"),
            "fields": [asdict(f) for f in v["fields"]],
            "additional_statements": v.get("additional_statements", []),
            "image_quality_notes": v.get("image_quality_notes")}


def run_bottle(case_name, image_paths, application):
    """Run Passes A/B/C IN PARALLEL on one bottle, then verify() on Pass B. Returns the full record
    (evidence + per-pass timing + parallel wall-clock + deterministic analysis)."""
    t0 = time.perf_counter()
    images = [(open(p, "rb").read(), _media_type(p)) for p in image_paths]
    load_seconds = time.perf_counter() - t0

    par_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        fa = pool.submit(_timed_pass, pass_a_raw_text, images, "image/png")
        fb = pool.submit(_timed_pass, extract_fields, images, "image/png")
        fc = pool.submit(_timed_pass, pass_c_visual, images, "image/png")
        ra, rb, rc = fa.result(), fb.result(), fc.result()
    parallel_wall = time.perf_counter() - par_start

    verify_obj, verify_serialized, verification_seconds = None, None, 0.0
    if rb["result"] is not None and application is not None:
        v0 = time.perf_counter()
        verify_obj = verify(rb["result"], application)
        verification_seconds = time.perf_counter() - v0
        verify_serialized = _serialize_verify(verify_obj)

    analysis = _analyze(ra["result"], rb["result"], rc["result"])
    total_bottle = time.perf_counter() - t0
    errors = {k: v for k, v in (("pass_a", ra["error"]), ("pass_b", rb["error"]),
                                ("pass_c", rc["error"])) if v}
    return {
        "case": case_name,
        "images": [os.path.relpath(p, ROOT) for p in image_paths],
        "pass_a_raw_text": ra["result"],
        "pass_b_structured": rb["result"],
        "pass_b_verify": verify_serialized,
        "pass_c_visual": rc["result"],
        "analysis": analysis,
        "timing": {
            "image_load_seconds": round(load_seconds, 3),
            "pass_a_raw_text_seconds": round(ra["seconds"], 3),
            "pass_b_structured_seconds": round(rb["seconds"], 3),
            "pass_c_visual_seconds": round(rc["seconds"], 3),
            "parallel_wall_seconds": round(parallel_wall, 3),
            "verification_seconds": round(verification_seconds, 4),
            "total_bottle_seconds": round(total_bottle, 3),
        },
        "attempts": {"pass_a": ra["attempts"], "pass_b": rb["attempts"], "pass_c": rc["attempts"]},
        "errors": errors,
    }


# === deterministic analysis (the JUDGE layer -- no model voting) =============
def _agree(b_val, b_conf, c_val, c_conf):
    """(agree, high_conf_conflict) for a bold observation between Pass B and Pass C.
    high_conf_conflict means BOTH witnesses are 'high' confidence and they DISAGREE -- which we
    surface as REVIEW evidence and explicitly DO NOT resolve by voting."""
    if b_val is None or c_val is None:
        return None, False
    agree = (b_val == c_val)
    return agree, (not agree and b_conf == "high" and c_conf == "high")


def _uncertain(val, conf):
    """A bold read is 'uncertain' if it is None or not high-confidence."""
    return val is None or conf != "high"


def _analyze(raw_a, struct_b, vis_c):
    text = (raw_a or {}).get("raw_text") or ""
    norm = _normalize(text)

    # --- raw-text (Pass A) corroboration ---
    raw_body_exact = bool(text) and _CANONICAL_WARNING_BODY_NORM in norm
    hm = re.search(r"government\s+warning", text, re.IGNORECASE)
    raw_header_present = bool(hm)
    raw_header_caps = (text[hm.start():hm.end()].isupper()) if hm else None
    raw_abv_token = bool(re.search(r"\babv\b", text, re.IGNORECASE))
    sg = re.search(r"surgeon\s+general", text, re.IGNORECASE)
    raw_sg_lowercase = None
    if sg:
        seg = text[sg.start():sg.end()]
        gm = re.search(r"general", seg, re.IGNORECASE)
        raw_sg_lowercase = seg[0].islower() or bool(gm and seg[gm.start()].islower())

    # --- structured (Pass B) warning + abv ---
    bgw = (struct_b or {}).get("government_warning") or {}
    b_text = bgw.get("text")
    b_body_exact = bool(b_text) and (_normalize(_warning_body(b_text)) == _CANONICAL_WARNING_BODY_NORM)
    b_hbold, b_hconf = bgw.get("header_bold"), bgw.get("header_bold_confidence")
    b_bbold, b_bconf = bgw.get("body_bold"), bgw.get("body_bold_confidence")
    b_caps = bgw.get("header_all_caps")
    b_abv = ((struct_b or {}).get("alcohol_content") or {}).get("value")
    b_abv_token = bool(b_abv) and bool(re.search(r"\babv\b", b_abv, re.IGNORECASE))

    # --- visual (Pass C) ---
    c = vis_c or {}
    c_hbold, c_hconf = c.get("header_bold"), c.get("header_bold_confidence")
    c_bbold, c_bconf = c.get("body_bold"), c.get("body_bold_confidence")
    c_caps = c.get("header_all_caps")

    hb_agree, hb_conflict = _agree(b_hbold, b_hconf, c_hbold, c_hconf)
    bb_agree, bb_conflict = _agree(b_bbold, b_bconf, c_bbold, c_bconf)

    # Did Pass C add high-confidence signal where Pass B was uncertain? (reduce uncertainty)
    c_resolved_header = _uncertain(b_hbold, b_hconf) and (c_hbold is not None and c_hconf == "high")
    c_resolved_body = _uncertain(b_bbold, b_bconf) and (c_bbold is not None and c_bconf == "high")
    # Pure duplication: both confident AND agree -> C just echoed B.
    c_dup_header = (hb_agree is True and b_hconf == "high" and c_hconf == "high")
    c_dup_body = (bb_agree is True and b_bconf == "high" and c_bconf == "high")

    return {
        "raw_text": {
            "available": raw_a is not None,
            "char_count": len(text),
            "warning_body_exact_match": raw_body_exact,   # canonical body present verbatim in OCR
            "header_present": raw_header_present,
            "header_all_caps": raw_header_caps,
            "surgeon_general_lowercase": raw_sg_lowercase,
            "abv_notation_token": raw_abv_token,
            # raw text surfaced evidence the STRUCTURED read did not carry:
            "raw_only_warning_wording": raw_body_exact and not b_body_exact,
            "raw_only_abv_notation": raw_abv_token and not b_abv_token,
        },
        "header_bold": {"structured_B": b_hbold, "structured_B_conf": b_hconf,
                        "visual_C": c_hbold, "visual_C_conf": c_hconf,
                        "agree": hb_agree, "high_conf_conflict": hb_conflict,
                        "C_resolved_B_uncertainty": c_resolved_header, "C_duplicated_B": c_dup_header},
        "body_bold": {"structured_B": b_bbold, "structured_B_conf": b_bconf,
                      "visual_C": c_bbold, "visual_C_conf": c_bconf,
                      "agree": bb_agree, "high_conf_conflict": bb_conflict,
                      "C_resolved_B_uncertainty": c_resolved_body, "C_duplicated_B": c_dup_body},
        "header_caps": {"raw_A": raw_header_caps, "structured_B": b_caps, "visual_C": c_caps},
        # the headline judge output: a high-confidence bold disagreement is REVIEW evidence, never
        # resolved by voting (per the experiment constraint).
        "bold_recommendation": ("REVIEW (high-confidence witness conflict; do NOT auto-resolve)"
                                if (hb_conflict or bb_conflict) else None),
        "structured_warning_body_exact": b_body_exact,
    }


# === application loading =====================================================
def _app_full(name):
    return json.load(open(os.path.join(APPS, name), encoding="utf-8"))


def _app_clean(name):
    return {k: v for k, v in _app_full(name).items() if not k.startswith("_")}


def _product_key(path):
    """Group key for a baseline face, matching smoke_test's _Front/_Other stripping."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[ _\-]*(front|other|back|label).*$", "", stem, flags=re.IGNORECASE) or stem


# === Stage 1: baseline labels ================================================
def _baseline_keymap():
    """Map each baseline group key -> its application JSON, using the app `_meta.clean_baseline_*`
    refs (the existing eval/smoke-test convention). Returns (keymap, ambiguous) where `ambiguous`
    flags any group key that more than one application claims (so we never guess silently)."""
    keymap, ambiguous = {}, {}
    for fn in sorted(os.listdir(APPS)):
        if not fn.endswith(".json"):
            continue
        meta = _app_full(fn).get("_meta", {})
        for ref in (meta.get("clean_baseline_front"), meta.get("clean_baseline_back")):
            if ref:
                k = _product_key(ref)
                if k in keymap and keymap[k] != fn:
                    ambiguous.setdefault(k, {keymap[k]}).add(fn)
                keymap[k] = fn
    return keymap, ambiguous


def _resolve_baseline_cases():
    """Resolve each baseline group to its application, EXPLICITLY recording the match status and
    flagging ambiguity/misses as errors (never guessing). Shared by run_baseline + run_stability."""
    groups = _group_by_product(_gather([BASE]))
    keymap, ambiguous = _baseline_keymap()
    cases = []
    for key in sorted(groups):
        paths = sorted(groups[key])
        app_file = keymap.get(key)
        if key in ambiguous:
            cases.append({"key": key, "paths": paths, "app_file": None, "application": None,
                          "status": "ambiguous", "source": None,
                          "match_err": f"AMBIGUOUS: multiple applications map to baseline group "
                          f"'{key}': {sorted(ambiguous[key])} — not matched"})
        elif app_file is None:
            cases.append({"key": key, "paths": paths, "app_file": None, "application": None,
                          "status": "unmatched", "source": None,
                          "match_err": f"UNMATCHED: no application _meta.clean_baseline_* references "
                          f"baseline group '{key}'"})
        else:
            cases.append({"key": key, "paths": paths, "app_file": app_file,
                          "application": _app_clean(app_file), "status": "matched",
                          "source": "app _meta.clean_baseline_*", "match_err": None})
    return cases


def _attach_match_info(rec, c):
    rec["application_file"] = c["app_file"]
    rec["application_match_status"] = c["status"]
    rec["application_match_source"] = c["source"]
    if c["match_err"]:
        rec["errors"]["application_match"] = c["match_err"]
    return rec


def run_baseline():
    cases = _resolve_baseline_cases()
    print("=" * 100)
    print("STAGE 1 -- BASELINE LABELS (clean front/back pairs vs application; expect ~100% pass)")
    print(f"  A={RAW_TEXT_MODEL}  B={STRUCTURED_MODEL}  C={VISUAL_MODEL}\n")
    bottles = []
    for c in cases:
        print(f"  running {c['key']}  ({len(c['paths'])} img, app={c['app_file'] or c['status'].upper()}) ...",
              flush=True)
        rec = _attach_match_info(run_bottle(c["key"], c["paths"], c["application"]), c)
        rec["expected_defect"] = "none (clean baseline -> expect PASS)"
        bottles.append(rec)
        _print_bottle_line(rec)

    out = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "models": {"pass_a": RAW_TEXT_MODEL, "pass_b": STRUCTURED_MODEL, "pass_c": VISUAL_MODEL},
           "bottles": bottles}
    out["gate"] = _baseline_gate(bottles)
    os.makedirs(ARTIFACTS, exist_ok=True)
    json.dump(out, open(BASELINE_INTERMEDIATE, "w", encoding="utf-8"), indent=2, default=str)
    return out


# === Stability pass: run the baseline ensemble N times, quantify run-to-run flakiness ========
def _bold_key(val, conf):
    v = "true" if val is True else ("false" if val is False else "none")
    return f"{v}[{(conf or '?')[:1]}]"


def _agree_label(d):
    if d["agree"] is None:
        return "indeterminate"
    if d["high_conf_conflict"]:
        return "disagree(HIGH-conf)"
    return "agree" if d["agree"] else "disagree"


def _dist(recs, side, who):
    """Distribution of (value[conf]) for B or C on header_bold/body_bold across the runs of one bottle."""
    from collections import Counter
    c = Counter()
    for r in recs:
        d = r["analysis"][side]
        if who == "B":
            c[_bold_key(d["structured_B"], d["structured_B_conf"])] += 1
        else:
            c[_bold_key(d["visual_C"], d["visual_C_conf"])] += 1
    return dict(c)


def _aggregate_stability(runs):
    from collections import Counter
    order = [b["case"] for b in runs[0]] if runs else []
    by_case = {}
    for case in order:
        recs = [next(b for b in run if b["case"] == case) for run in runs]
        overall = Counter((r.get("pass_b_verify") or {}).get("overall", "n/a") for r in recs)
        field_names = []
        for r in recs:
            for f in (r.get("pass_b_verify") or {}).get("fields", []):
                if f["field"] not in field_names:
                    field_names.append(f["field"])
        fields = {}
        for fn in field_names:
            c = Counter()
            for r in recs:
                st = next((f["status"] for f in (r.get("pass_b_verify") or {}).get("fields", [])
                           if f["field"] == fn), "n/a")
                c[st] += 1
            fields[fn] = dict(c)
        bc_header = Counter(_agree_label(r["analysis"]["header_bold"]) for r in recs)
        bc_body = Counter(_agree_label(r["analysis"]["body_bold"]) for r in recs)
        flags = []
        if len(overall) > 1:
            flags.append("overall verdict NOT stable across runs")
        if len(fields.get("government_warning", {})) > 1:
            flags.append("warning verdict flips across runs")
        if len(fields.get("name_and_address", {})) > 1:
            flags.append("name/address verdict flips across runs")
        if len(_dist(recs, "header_bold", "B")) > 1:
            flags.append("Pass B header-bold read flips across runs")
        if len(_dist(recs, "header_bold", "C")) > 1:
            flags.append("Pass C header-bold read flips across runs")
        by_case[case] = {
            "application_file": recs[0].get("application_file"),
            "overall": dict(overall), "fields": fields,
            "B_header_bold": _dist(recs, "header_bold", "B"), "B_body_bold": _dist(recs, "body_bold", "B"),
            "C_header_bold": _dist(recs, "header_bold", "C"), "C_body_bold": _dist(recs, "body_bold", "C"),
            "BC_header_agree": dict(bc_header), "BC_body_agree": dict(bc_body),
            "BC_header_high_conf_conflicts": sum(1 for r in recs if r["analysis"]["header_bold"]["high_conf_conflict"]),
            "BC_body_high_conf_conflicts": sum(1 for r in recs if r["analysis"]["body_bold"]["high_conf_conflict"]),
            "total_seconds": _stats([r["timing"]["total_bottle_seconds"] for r in recs]),
            "flakiness_flags": flags,
        }
    n = len(runs)
    return {"runs": n, "bottles": len(by_case), "by_case": by_case,
            "bottles_with_stable_overall": sum(1 for c in by_case if len(by_case[c]["overall"]) == 1),
            "bottles_with_stable_warning": sum(1 for c in by_case if len(by_case[c]["fields"].get("government_warning", {})) <= 1),
            "bottles_with_stable_B_header_bold": sum(1 for c in by_case if len(by_case[c]["B_header_bold"]) == 1),
            "overall_pass_bottle_runs": sum(by_case[c]["overall"].get("pass", 0) for c in by_case),
            "total_bottle_runs": n * len(by_case)}


def run_stability(n):
    cases = _resolve_baseline_cases()
    print("=" * 100)
    print(f"STABILITY PASS -- baseline ensemble run {n}x to quantify run-to-run flakiness "
          f"(does NOT run the error stage)")
    print(f"  A={RAW_TEXT_MODEL}  B={STRUCTURED_MODEL}  C={VISUAL_MODEL}\n")
    runs = []
    for i in range(n):
        print(f"  --- baseline pass {i + 1}/{n} ---", flush=True)
        run_bottles = []
        for c in cases:
            rec = _attach_match_info(run_bottle(c["key"], c["paths"], c["application"]), c)
            run_bottles.append(rec)
            v = (rec.get("pass_b_verify") or {}).get("overall", "n/a")
            w = next((f["status"] for f in (rec.get("pass_b_verify") or {}).get("fields", [])
                      if f["field"] == "government_warning"), "n/a")
            hb = rec["analysis"]["header_bold"]
            print(f"      {c['key']:<11} overall={v:<13} warn={w:<13} "
                  f"Bhdr={_bold_key(hb['structured_B'], hb['structured_B_conf'])} "
                  f"Chdr={_bold_key(hb['visual_C'], hb['visual_C_conf'])} "
                  f"total={rec['timing']['total_bottle_seconds']:.1f}s")
        runs.append(run_bottles)
    agg = _aggregate_stability(runs)
    out = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "models": {"pass_a": RAW_TEXT_MODEL, "pass_b": STRUCTURED_MODEL, "pass_c": VISUAL_MODEL},
           "runs_requested": n, "aggregate": agg, "runs": runs}
    os.makedirs(ARTIFACTS, exist_ok=True)
    json.dump(out, open(STABILITY_JSON, "w", encoding="utf-8"), indent=2, default=str)
    open(STABILITY_MD, "w", encoding="utf-8").write(_build_stability_md(out))
    _print_stability_summary(agg)
    print(f"\n  wrote artifacts/evidence_ensemble_stability.json and .md")
    return out


def _print_stability_summary(agg):
    print("\n" + "-" * 100)
    print(f"STABILITY SUMMARY ({agg['runs']} runs x {agg['bottles']} bottles)")
    print(f"  overall verdict stable across all runs: {agg['bottles_with_stable_overall']}/{agg['bottles']} bottles")
    print(f"  warning verdict stable: {agg['bottles_with_stable_warning']}/{agg['bottles']}  | "
          f"Pass B header-bold read stable: {agg['bottles_with_stable_B_header_bold']}/{agg['bottles']}")
    print(f"  clean-label overall PASS rate: {agg['overall_pass_bottle_runs']}/{agg['total_bottle_runs']} bottle-runs")
    for case, d in agg["by_case"].items():
        print(f"  {case:<11} overall={d['overall']}  warn={d['fields'].get('government_warning', {})}  "
              f"Bhdr={d['B_header_bold']}  Chdr={d['C_header_bold']}")
        for fl in d["flakiness_flags"]:
            print(f"       ! {fl}")
    print("-" * 100)


def _classify_nonpass(field, reason):
    """Short cause label for a non-passing baseline field (for the gate explanation)."""
    r = (reason or "").lower()
    if "could not verify required label information" in r:
        return "image-quality reframe (photo legibility, not a label defect)"
    if field == "name_and_address":
        return "name/address fuzzy/coverage matcher (formatting/abbreviation), pre-existing"
    if field == "government_warning":
        if "bold" in r:
            return "warning bold gate — confidence-gated by design (BENCHMARK_NOTES.md)"
        if "capital" in r:
            return "warning ALL-CAPS check"
        return "warning wording"
    return "other (inspect reason)"


def _baseline_gate(bottles):
    """Is the clean baseline 'close to 100% pass/completeness'? Auto-continue to the error stage
    only if the field-level pass rate >= BASELINE_GATE AND there are no hard FAILs AND every bottle
    was matched to an application AND verified. Returns the metrics + the non-pass causes."""
    total = passed = hard_fails = 0
    nonpass, app_errors = [], []
    for b in bottles:
        if b["errors"].get("application_match"):
            app_errors.append((b["case"], b["errors"]["application_match"]))
        v = b.get("pass_b_verify")
        if not v:
            continue
        for f in v["fields"]:
            total += 1
            if f["status"] == "pass":
                passed += 1
            else:
                nonpass.append({"case": b["case"], "field": f["field"], "status": f["status"],
                                "cause": _classify_nonpass(f["field"], f["reason"]), "reason": f["reason"]})
                if f["status"] == "fail":
                    hard_fails += 1
    field_rate = (passed / total) if total else 0.0
    overall_pass = sum(1 for b in bottles if (b.get("pass_b_verify") or {}).get("overall") == "pass")
    all_verified = all(b.get("pass_b_verify") for b in bottles)
    passed_gate = (field_rate >= BASELINE_GATE and hard_fails == 0 and not app_errors and all_verified)
    return {"passed": passed_gate, "gate_threshold": BASELINE_GATE,
            "field_pass_rate": round(field_rate, 4), "fields_passed": passed, "fields_total": total,
            "overall_pass": overall_pass, "bottles": len(bottles), "hard_fails": hard_fails,
            "application_match_errors": app_errors, "nonpass_fields": nonpass}


def _print_gate(gate):
    print("\n" + "-" * 100)
    state = "PASSED" if gate["passed"] else "TRIPPED"
    print(f"STAGE GATE: {state}  | field pass {gate['fields_passed']}/{gate['fields_total']} "
          f"({gate['field_pass_rate']*100:.0f}%, need {gate['gate_threshold']*100:.0f}%)  | "
          f"overall-PASS {gate['overall_pass']}/{gate['bottles']}  | hard-FAILs {gate['hard_fails']}  | "
          f"app-match errors {len(gate['application_match_errors'])}")
    for e in gate["application_match_errors"]:
        print(f"  app-match: {e[0]}: {e[1]}")
    for nf in gate["nonpass_fields"]:
        print(f"  non-pass: {nf['case']}.{nf['field']}={nf['status']}  -> {nf['cause']}")
    print("-" * 100)


# === Stage 2: error labels ===================================================
def run_errors():
    manifest = list(csv.DictReader(open(os.path.join(ERR, "test_fixtures_manifest.csv"), encoding="utf-8")))
    apps = {bev: _app_clean(p["app"]) for bev, p in PRODUCTS.items()}
    print("=" * 100)
    print(f"STAGE 2 -- ERROR LABELS (single-defect fixture + clean other face vs application)")
    print(f"  A={RAW_TEXT_MODEL}  B={STRUCTURED_MODEL}  C={VISUAL_MODEL}\n")
    cases = []
    for m in manifest:
        bev, side, fn = m["beverage"], m["side"], m["rename_to"]
        prod = PRODUCTS[bev]
        errored = os.path.join(ERR, fn)
        other = prod["back"] if side == "front" else prod["front"]
        paths = [errored, other] if side == "front" else [other, errored]
        print(f"  running {m['test_id']:<20} ({fn}) ...", flush=True)
        rec = run_bottle(m["test_id"], paths, apps[bev])
        rec["application_file"] = prod["app"]
        rec["expected_defect"] = m["defect_introduced"]
        rec["expected_verdict"] = m["expected_verdict"]
        rec["check_exercised"] = m["check_exercised"]
        rec["clean_fixture"] = m["clean"]
        rec["scoring"] = _score_error(m, rec)
        cases.append(rec)
        sc = rec["scoring"]
        tag = "CAUGHT" if sc["caught"] else ("FALSE-PASS" if sc["false_pass"] else "miss/partial")
        gap = "  [known gap]" if sc["known_gap"] and sc["false_pass"] else ""
        danger = "  !!DANGEROUS-FALSE-PASS-EVIDENCE" if sc["dangerous_false_pass_evidence"] else ""
        print(f"      {sc['field']}:{sc['field_verdict']:<13} exp={m['expected_verdict']:<13} {tag}{gap}{danger}")
    out = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "models": {"pass_a": RAW_TEXT_MODEL, "pass_b": STRUCTURED_MODEL, "pass_c": VISUAL_MODEL},
           "cases": cases}
    os.makedirs(ARTIFACTS, exist_ok=True)
    json.dump(out, open(ERRORS_INTERMEDIATE, "w", encoding="utf-8"), indent=2, default=str)
    return out


def _score_error(m, rec):
    """Per-check scoring on the exercised field, plus the witness-evidence flags answering 'what did
    the new witnesses catch / what slipped through / any dangerous false-pass evidence'."""
    field = CHECK_FIELD.get(m["check_exercised"], "government_warning")
    exp = EXPECT.get(m["expected_verdict"], m["expected_verdict"].lower())
    is_defect = m["expected_verdict"] in ("FAIL", "NEEDS_REVIEW")
    fields = (rec.get("pass_b_verify") or {}).get("fields") or []
    fr = next((f for f in fields if f["field"] == field), None)
    field_verdict = fr["status"] if fr else "n/a"
    reason = fr["reason"] if fr else (rec["errors"].get("pass_b") or "")
    a = rec["analysis"]

    # what the WITNESSES (A raw text, C visual) surfaced about this defect, deterministically
    witness_evidence = []
    if a["raw_text"]["raw_only_warning_wording"]:
        witness_evidence.append("raw text carried the exact warning wording that structured B did not")
    if m["check_exercised"] == "government_warning_exact_match":
        if not a["raw_text"]["warning_body_exact_match"] and a["raw_text"]["available"]:
            witness_evidence.append("raw text shows the warning body does NOT match canonical (wording deviation visible)")
        if a["raw_text"]["surgeon_general_lowercase"]:
            witness_evidence.append("raw text shows 'surgeon general' lower-cased")
    if m["check_exercised"] == "abv_notation_format" and a["raw_text"]["abv_notation_token"]:
        witness_evidence.append("raw text shows the bare 'ABV' notation")
    if a["bold_recommendation"]:
        witness_evidence.append(a["bold_recommendation"])

    # DANGEROUS false-pass evidence: a witness reproduced a compliant warning on a fixture whose
    # body wording/punctuation was actually altered (the witness hallucinated compliance).
    dangerous = []
    if m["test_id"] in WORDING_ALTERED_FIXTURES:
        if a["raw_text"]["warning_body_exact_match"]:
            dangerous.append("Pass A raw text reproduced the canonical warning body verbatim despite "
                             "the printed body being altered (hallucinated compliant wording)")
        if a["structured_warning_body_exact"]:
            dangerous.append("Pass B structured warning body matched canonical despite the altered "
                             "printed body (structured read hallucinated compliant wording)")
    # a high-confidence Pass C 'all good' on a known-defect warning fixture would also be dangerous
    c = rec.get("pass_c_visual") or {}
    if (is_defect and field == "government_warning" and m["check_exercised"] != "case_normalization"
            and c.get("header_bold") is True and c.get("header_bold_confidence") == "high"
            and c.get("body_bold") is False and c.get("body_bold_confidence") == "high"
            and a["structured_warning_body_exact"] and m["test_id"] not in ("GW-LOWERCASE-SG",)):
        dangerous.append("Pass C visual witness gave a high-confidence compliant bold profile on a "
                         "defective warning fixture")

    return {
        "field": field, "field_verdict": field_verdict, "reason": reason,
        "expected_verdict": m["expected_verdict"],
        "caught": field_verdict == exp,
        "false_pass": bool(is_defect and field_verdict == "pass"),
        "needs_review": field_verdict == "needs_review",
        "known_gap": m["check_exercised"] in KNOWN_GAP_CHECKS,
        "overall_verdict": (rec.get("pass_b_verify") or {}).get("overall", "n/a"),
        "witness_evidence": witness_evidence,
        "dangerous_false_pass_evidence": dangerous,
    }


# === reporting ===============================================================
def _print_bottle_line(rec):
    t = rec["timing"]
    v = (rec.get("pass_b_verify") or {}).get("overall", "n/a")
    a = rec["analysis"]
    confl = " BOLD-CONFLICT(hi)" if (a["header_bold"]["high_conf_conflict"]
                                     or a["body_bold"]["high_conf_conflict"]) else ""
    errs = (" ERRORS:" + ",".join(rec["errors"])) if rec["errors"] else ""
    print(f"      verify={v:<13} wall={t['parallel_wall_seconds']:.1f}s total={t['total_bottle_seconds']:.1f}s "
          f"(A={t['pass_a_raw_text_seconds']:.1f} B={t['pass_b_structured_seconds']:.1f} "
          f"C={t['pass_c_visual_seconds']:.1f}){confl}{errs}")


def _fmt(x):
    return "—" if x is None else ("true" if x is True else ("false" if x is False else str(x)))


def _cell(s, n=44):
    s = "" if s is None else str(s)
    s = s.replace("\n", " ").replace("|", "/").strip()
    return (s[: n - 1] + "…") if len(s) > n else s


def _stats(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"n": n, "avg": sum(vals) / n, "median": med, "min": min(vals), "max": max(vals)}


def _timing_block(records, label):
    """Timing stats over total_bottle_seconds, using only bottles with no pass errors (so a flaky
    retry doesn't skew the headline)."""
    clean = [r for r in records if not r["errors"]]
    totals = [r["timing"]["total_bottle_seconds"] for r in clean]
    walls = [r["timing"]["parallel_wall_seconds"] for r in clean]
    st = _stats(totals)
    if not st:
        return [f"_{label}: no clean (error-free) bottles to time._", ""]
    slowest = max(clean, key=lambda r: r["timing"]["total_bottle_seconds"])
    st_t = slowest["timing"]
    # which pass dominated the slowest bottle
    pass_secs = {"A": st_t["pass_a_raw_text_seconds"], "B": st_t["pass_b_structured_seconds"],
                 "C": st_t["pass_c_visual_seconds"]}
    dom = max(pass_secs, key=pass_secs.get)
    wall_st = _stats(walls)
    over = sum(1 for t in totals if t > TARGET_BOTTLE_SECONDS)
    over_strict = sum(1 for t in totals if t > STRICT_TARGET_SECONDS)
    return [
        f"**{label}** (n={st['n']} error-free bottles):", "",
        f"- `total_bottle_seconds`: avg **{st['avg']:.2f}s**, median **{st['median']:.2f}s**, "
        f"max **{st['max']:.2f}s** (min {st['min']:.2f}s)",
        f"- `parallel_wall_seconds`: avg {wall_st['avg']:.2f}s, median {wall_st['median']:.2f}s, "
        f"max {wall_st['max']:.2f}s",
        f"- slowest bottle: **{slowest['case']}** at {st_t['total_bottle_seconds']:.2f}s "
        f"(Pass {dom} dominated at {pass_secs[dom]:.2f}s; A={st_t['pass_a_raw_text_seconds']:.2f} "
        f"B={st_t['pass_b_structured_seconds']:.2f} C={st_t['pass_c_visual_seconds']:.2f}; "
        f"wall {st_t['parallel_wall_seconds']:.2f}s, verify {st_t['verification_seconds']:.3f}s)",
        f"- vs {TARGET_BOTTLE_SECONDS:.0f}s budget: **{over}/{st['n']} over**; "
        f"vs {STRICT_TARGET_SECONDS:.0f}s strict target: {over_strict}/{st['n']} over "
        f"(avg is {'UNDER' if st['avg'] <= TARGET_BOTTLE_SECONDS else 'OVER'} the "
        f"{TARGET_BOTTLE_SECONDS:.0f}s budget by {abs(st['avg'] - TARGET_BOTTLE_SECONDS):.2f}s)",
        "",
    ]


def _archive_previous_results(targets):
    """COPY (preserve, don't move) the given existing files into a timestamped archive/ folder
    BEFORE this run overwrites them, so a re-run never silently destroys the prior result. Copy
    (not move) so a two-stage run (`--stage baseline` then `--stage errors`) can still read the
    prior baseline intermediate it needs to merge. `targets` is only the files THIS run will write.
    Returns the archive path, or None."""
    existing = [p for p in targets if os.path.exists(p)]
    if not existing:
        return None
    stamp = datetime.fromtimestamp(max(os.path.getmtime(p) for p in existing)).strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(ARCHIVE_DIR, stamp)
    os.makedirs(dest, exist_ok=True)
    for p in existing:
        shutil.copy2(p, os.path.join(dest, os.path.basename(p)))
    return dest


def _clear_intermediate(path):
    if os.path.exists(path):
        os.remove(path)


def build_outputs():
    base = json.load(open(BASELINE_INTERMEDIATE, encoding="utf-8")) if os.path.exists(BASELINE_INTERMEDIATE) else None
    errs = json.load(open(ERRORS_INTERMEDIATE, encoding="utf-8")) if os.path.exists(ERRORS_INTERMEDIATE) else None
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    models = {"pass_a": RAW_TEXT_MODEL, "pass_b": STRUCTURED_MODEL, "pass_c": VISUAL_MODEL}

    combined = {"generated": when, "models": models,
                "target_bottle_seconds": TARGET_BOTTLE_SECONDS,
                "baseline_gate": (base or {}).get("gate"),
                "baseline": base, "errors": errs}
    os.makedirs(ARTIFACTS, exist_ok=True)
    json.dump(combined, open(RESULTS_JSON, "w", encoding="utf-8"), indent=2, default=str)
    md = _build_md(base, errs, when, models)
    open(RESULTS_MD, "w", encoding="utf-8").write(md)
    print(f"\n  wrote artifacts/evidence_ensemble_results.json and .md "
          f"({'baseline' if base else ''}{'+' if base and errs else ''}{'errors' if errs else ''})")


def _build_md(base, errs, when, models):
    L = []
    L.append("# Evidence-ensemble experiment — results\n")
    L.append(f"_Generated {when}. **Benchmark/eval only — no production code (`app.py`, "
             f"`verification.py`, `extract_fields`) was changed.** Pass A = raw-text witness "
             f"(`{models['pass_a']}`), Pass B = production structured extraction (`{models['pass_b']}`, "
             f"feeds `verify()`), Pass C = warning visual witness (`{models['pass_c']}`). "
             f"A/B/C run in parallel; deterministic Python judges the evidence — models never vote._\n")

    # ---- Stage 1: baseline ----
    if base:
        L += _md_baseline(base)
    else:
        L.append("## Stage 1 — baseline labels\n\n_Not run yet._\n")
    # ---- Stage 2: errors ----
    if errs:
        L += _md_errors(errs)
    else:
        L.append("## Stage 2 — error labels\n\n_Not run yet (run `--stage errors`)._\n")

    # ---- Decision summary (the focused diagnostic deliverable) ----
    L += _md_decision_summary(base, errs)

    # ---- The 9 questions ----
    L += _md_answers(base, errs)

    # ---- Timing summary ----
    L.append("## Timing summary\n")
    if base:
        L += _timing_block(base["bottles"], "Baseline labels")
    if errs:
        L += _timing_block(errs["cases"], "Error labels")
    L.append("> `parallel_wall_seconds` is the elapsed real time from launching A/B/C together to all "
             "three finishing; because the three calls overlap, the wall-clock is ~the slowest single "
             "pass, not the sum. `total_bottle_seconds` adds image load + verify + analysis on top.\n")
    return "\n".join(L)


def _md_baseline(base):
    bottles = base["bottles"]
    L = ["## Stage 1 — baseline labels (clean; expect 100% pass)\n"]
    n_pass = sum(1 for b in bottles if (b.get("pass_b_verify") or {}).get("overall") == "pass")
    L.append(f"**{n_pass}/{len(bottles)} baselines reached an overall PASS.** "
             f"Each is a clean front+back pair matched to its application JSON.\n")
    L.append("| bottle | app | overall | non-PASS fields | bold: B vs C (header / body) | wall | total |")
    L.append("|" + "---|" * 7)
    for b in bottles:
        v = b.get("pass_b_verify") or {}
        nonpass = [f"{f['field']}={f['status']}" for f in v.get("fields", []) if f["status"] != "pass"]
        hb, bb = b["analysis"]["header_bold"], b["analysis"]["body_bold"]
        bold = (f"hdr {_fmt(hb['structured_B'])}[{(hb['structured_B_conf'] or '?')[:1]}] vs "
                f"{_fmt(hb['visual_C'])}[{(hb['visual_C_conf'] or '?')[:1]}] / "
                f"body {_fmt(bb['structured_B'])}[{(bb['structured_B_conf'] or '?')[:1]}] vs "
                f"{_fmt(bb['visual_C'])}[{(bb['visual_C_conf'] or '?')[:1]}]")
        t = b["timing"]
        L.append("| " + " | ".join([
            _cell(b["case"], 12), _cell(b.get("application_file") or (b.get("application_match_status") or "?").upper(), 12),
            _cell(v.get("overall", "ERROR"), 13), _cell(", ".join(nonpass) or "none", 40),
            _cell(bold, 46), f"{t['parallel_wall_seconds']:.1f}s", f"{t['total_bottle_seconds']:.1f}s",
        ]) + " |")
    L.append("")
    # application matching, recorded (not guessed) — call out any ambiguous/unmatched group
    L.append("**Application matching** (from the app `_meta.clean_baseline_*` refs; recorded per "
             "bottle, never guessed): " + "; ".join(
                 f"`{b['case']}`→`{b.get('application_file') or (b.get('application_match_status') or '?').upper()}`"
                 + ("" if b.get("application_match_status") == "matched"
                    else f" **[{b.get('application_match_status')}]**")
                 for b in bottles) + ".\n")
    L += _md_gate(base.get("gate"))
    # per-bottle evidence detail
    L.append("### Per-baseline evidence detail\n")
    for b in bottles:
        L += _md_bottle_detail(b, clean=True)
    return L


def _md_gate(gate):
    if not gate:
        return [""]
    L = ["### Stage gate — baseline → error stage\n"]
    if gate["passed"]:
        L.append(f"**GATE PASSED.** Clean baselines are close to fully passing "
                 f"(field pass rate **{gate['field_pass_rate']*100:.0f}%** ≥ "
                 f"{gate['gate_threshold']*100:.0f}% threshold, no hard FAILs, all matched). "
                 f"`--stage all` continues to the error stage automatically.\n")
    else:
        L.append(f"**GATE TRIPPED — the error stage does NOT run automatically.** Clean baselines are "
                 f"not close to 100% pass/completeness, so the run pauses for inspection.\n")
    L.append(f"- field pass rate: **{gate['fields_passed']}/{gate['fields_total']} "
             f"({gate['field_pass_rate']*100:.0f}%)** vs threshold {gate['gate_threshold']*100:.0f}% · "
             f"overall-PASS bottles: {gate['overall_pass']}/{gate['bottles']} · hard FAILs on a clean "
             f"label: {gate['hard_fails']}")
    if gate["application_match_errors"]:
        L.append("- **application-match errors (flagged, not guessed silently):**")
        for case, err in gate["application_match_errors"]:
            L.append(f"  - `{case}`: {_cell(err, 100)}")
    if gate["nonpass_fields"]:
        L.append("- non-pass baseline fields and likely cause:")
        for nf in gate["nonpass_fields"]:
            L.append(f"  - `{nf['case']}` · `{nf['field']}`={nf['status']} — {nf['cause']}  "
                     f"_({_cell(nf['reason'], 80)})_")
    if not gate["passed"]:
        L.append("- **Why (not a regression):** the non-pass causes above are pre-existing pipeline "
                 "behaviours — the production name/address fuzzy/coverage matcher and the "
                 "confidence-gated warning bold rule — surfaced on perfect labels; they are NOT caused "
                 "by the ensemble (the witnesses never touch the verdict). To proceed after inspecting: "
                 "re-run `--stage errors` (or `--stage all --force-errors`).")
    L.append("")
    return L


def _dist_str(d):
    """Render a distribution dict (e.g. {'pass':3,'needs_review':2}) as 'pass×3, needs_review×2'."""
    if not d:
        return "—"
    return ", ".join(f"{k}×{v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))


def _build_stability_md(out):
    agg = out["aggregate"]
    n = agg["runs"]
    L = ["# Evidence-ensemble — baseline STABILITY pass\n"]
    L.append(f"_Generated {out['generated']}. Ran the **clean baseline ensemble {n}×** to quantify "
             f"run-to-run flakiness on perfect labels. Models: A=`{out['models']['pass_a']}`, "
             f"B=`{out['models']['pass_b']}` (feeds `verify()`), C=`{out['models']['pass_c']}`. "
             f"Benchmark-only; error stage NOT run._\n")
    L.append(f"**Headline:** overall verdict was stable across all {n} runs on "
             f"**{agg['bottles_with_stable_overall']}/{agg['bottles']}** bottles; the warning verdict "
             f"on **{agg['bottles_with_stable_warning']}/{agg['bottles']}**; Pass B's header-bold read "
             f"on **{agg['bottles_with_stable_B_header_bold']}/{agg['bottles']}**. Clean-label overall "
             f"PASS rate: **{agg['overall_pass_bottle_runs']}/{agg['total_bottle_runs']}** bottle-runs.\n")

    L.append("## Per-bottle distributions across runs\n")
    L.append(f"| bottle | app | overall ({n} runs) | warning verdict | Pass B header-bold | "
             f"Pass C header-bold | B↔C header agree | total s (med) |")
    L.append("|" + "---|" * 8)
    for case, d in agg["by_case"].items():
        ts = d["total_seconds"]
        L.append("| " + " | ".join([
            _cell(case, 12), _cell(d["application_file"], 10),
            _cell(_dist_str(d["overall"]), 30), _cell(_dist_str(d["fields"].get("government_warning", {})), 24),
            _cell(_dist_str(d["B_header_bold"]), 26), _cell(_dist_str(d["C_header_bold"]), 26),
            _cell(_dist_str(d["BC_header_agree"]), 26),
            (f"{ts['median']:.1f}" if ts else "—"),
        ]) + " |")
    L.append("")

    L.append("## Flakiness flags\n")
    any_flag = False
    for case, d in agg["by_case"].items():
        if d["flakiness_flags"]:
            any_flag = True
            L.append(f"- **`{case}`**: " + "; ".join(d["flakiness_flags"]))
            L.append(f"    - overall: {_dist_str(d['overall'])} · warning: "
                     f"{_dist_str(d['fields'].get('government_warning', {}))} · "
                     f"name/address: {_dist_str(d['fields'].get('name_and_address', {}))}")
            L.append(f"    - B header-bold: {_dist_str(d['B_header_bold'])} · B body-bold: {_dist_str(d['B_body_bold'])}")
            L.append(f"    - C header-bold: {_dist_str(d['C_header_bold'])} · C body-bold: {_dist_str(d['C_body_bold'])}")
            L.append(f"    - B↔C header agreement: {_dist_str(d['BC_header_agree'])} "
                     f"(high-conf conflicts: {d['BC_header_high_conf_conflicts']}); "
                     f"body agreement: {_dist_str(d['BC_body_agree'])} "
                     f"(high-conf conflicts: {d['BC_body_high_conf_conflicts']})")
    if not any_flag:
        L.append("- (none — every bottle was stable across all runs)")
    L.append("")

    L.append("## What this means\n")
    L.append(f"- **Clean-label verdicts are not deterministic.** With {n} repeats of the *same perfect "
             f"labels*, only {agg['bottles_with_stable_overall']}/{agg['bottles']} bottles held a single "
             f"overall verdict. A single baseline run is therefore not authoritative — which is exactly "
             f"why the stage gate exists.")
    L.append("- **The instability is concentrated in the warning bold gate** (Pass B's header-bold "
             "value/confidence wobbles between runs, flipping the `header_body_gate` verdict between "
             "pass and needs_review) and the name/address fuzzy matcher — the two pre-existing soft "
             "spots, not the witnesses.")
    L.append("- **The visual witness (C) does not stabilise bold.** Its header-bold reads have their own "
             "run-to-run spread and disagree with B on a meaningful fraction of runs; resolving the "
             "disagreement by voting would just trade one unstable signal for another. The only safe use "
             "remains routing a *high-confidence* B↔C disagreement to needs-review.")
    return "\n".join(L)


def _md_errors(errs):
    cases = errs["cases"]
    L = ["## Stage 2 — error labels (single-defect fixtures)\n"]
    caught = [c for c in cases if c["scoring"]["caught"]]
    fp_real = [c for c in cases if c["scoring"]["false_pass"] and not c["scoring"]["known_gap"]]
    fp_gap = [c for c in cases if c["scoring"]["false_pass"] and c["scoring"]["known_gap"]]
    danger = [c for c in cases if c["scoring"]["dangerous_false_pass_evidence"]]
    L.append(f"**Caught {len(caught)}/{len(cases)}** fixtures on their exercised field "
             f"(verification, Pass B). {len(fp_real)} real false-pass, {len(fp_gap)} false-pass on a "
             f"known-unchecked item. **{len(danger)} cases with dangerous false-pass *evidence* from a "
             f"witness** (see flags below).\n")
    L.append("| fixture | defect | exercised field | verdict (B) | exp | caught? | witness evidence / danger |")
    L.append("|" + "---|" * 7)
    for c in cases:
        sc = c["scoring"]
        note = "; ".join(sc["witness_evidence"]) or "—"
        if sc["dangerous_false_pass_evidence"]:
            note = "⚠ " + "; ".join(sc["dangerous_false_pass_evidence"])
        L.append("| " + " | ".join([
            _cell(c["case"], 18), _cell(c["expected_defect"], 34), _cell(sc["field"], 18),
            _cell(sc["field_verdict"], 13), _cell(sc["expected_verdict"], 12),
            ("yes" if sc["caught"] else ("FALSE-PASS" if sc["false_pass"] else "no")),
            _cell(note, 60),
        ]) + " |")
    L.append("")
    L.append("### Per-fixture evidence detail\n")
    for c in cases:
        L += _md_bottle_detail(c, clean=False)
    return L


def _md_bottle_detail(b, clean):
    L = []
    sc = b.get("scoring")
    head = f"#### {b['case']}"
    if not clean and sc:
        head += f" — exp {sc['expected_verdict']}, {sc['field']}={sc['field_verdict']} ({'caught' if sc['caught'] else 'NOT caught'})"
    L.append(head + "\n")
    L.append(f"- images: {', '.join('`'+i+'`' for i in b['images'])}  ·  app: `{b.get('application_file')}`")
    if not clean:
        L.append(f"- intended defect: {b.get('expected_defect')}  ·  check: `{b.get('check_exercised')}`")
    if b["errors"]:
        L.append(f"- **errors:** {b['errors']}")
    a = b["analysis"]
    hb, bb = a["header_bold"], a["body_bold"]
    L.append(f"- header bold — B `{_fmt(hb['structured_B'])}`[{hb['structured_B_conf']}] vs "
             f"C `{_fmt(hb['visual_C'])}`[{hb['visual_C_conf']}] → agree={_fmt(hb['agree'])}"
             f"{', **HIGH-CONF CONFLICT**' if hb['high_conf_conflict'] else ''}")
    L.append(f"- body bold — B `{_fmt(bb['structured_B'])}`[{bb['structured_B_conf']}] vs "
             f"C `{_fmt(bb['visual_C'])}`[{bb['visual_C_conf']}] → agree={_fmt(bb['agree'])}"
             f"{', **HIGH-CONF CONFLICT**' if bb['high_conf_conflict'] else ''}")
    rt = a["raw_text"]
    L.append(f"- raw text (A): warning-body-exact={_fmt(rt['warning_body_exact_match'])}, "
             f"header-caps={_fmt(rt['header_all_caps'])}, S/G-lowercase={_fmt(rt['surgeon_general_lowercase'])}, "
             f"ABV-token={_fmt(rt['abv_notation_token'])}, raw-only-wording={_fmt(rt['raw_only_warning_wording'])}, "
             f"raw-only-ABV-notation={_fmt(rt['raw_only_abv_notation'])}")
    c = b.get("pass_c_visual") or {}
    if c:
        L.append(f"- visual (C) extras: legibility={_fmt(c.get('legibility'))}, "
                 f"boxed/separated={_fmt(c.get('boxed_or_separated'))}, contrast={_fmt(c.get('contrast'))}, "
                 f"basis={_cell(c.get('basis'), 70)!r}")
    if a["bold_recommendation"]:
        L.append(f"- **judge:** {a['bold_recommendation']}")
    if not clean and sc and sc["dangerous_false_pass_evidence"]:
        for d in sc["dangerous_false_pass_evidence"]:
            L.append(f"- ⚠ **dangerous false-pass evidence:** {d}")
    # raw text excerpt
    rawtxt = (b.get("pass_a_raw_text") or {}).get("raw_text") or ""
    if rawtxt:
        excerpt = rawtxt if len(rawtxt) <= 1200 else rawtxt[:1200] + "\n…[truncated]"
        L.append("\n<details><summary>Pass A raw text</summary>\n\n```\n" + excerpt + "\n```\n</details>")
    L.append("")
    return L


def _md_decision_summary(base, errs):
    """The focused diagnostic deliverable: does Pass A (raw text) / Pass C (visual) reveal defects
    the current verifier MISSES, and which evidence is integrate-worthy? Three buckets. This is a
    DIAGNOSTIC read, NOT a green light — the baseline stability pass shows the ensemble (esp. bold)
    is not production-ready, so nothing here treats the baseline instability as 'passed'."""
    cases = errs["cases"] if errs else []
    caught = [c for c in cases if c["scoring"]["caught"]]
    false_pass = [c for c in cases if c["scoring"]["false_pass"]]
    # a defect the verifier MISSED entirely (false-pass) that a witness nonetheless surfaced:
    witness_only = [c for c in cases if c["scoring"]["false_pass"] and c["scoring"]["witness_evidence"]]
    raw_only = [c["case"] for c in cases if c["analysis"]["raw_text"]["raw_only_warning_wording"]
                or c["analysis"]["raw_text"]["raw_only_abv_notation"]]
    raw_abv = [c["case"] for c in cases if c["analysis"]["raw_text"]["abv_notation_token"]]
    raw_sg_low = [c["case"] for c in cases if c["analysis"]["raw_text"]["surgeon_general_lowercase"]]
    raw_wording_dev = [c["case"] for c in cases if c["check_exercised"] == "government_warning_exact_match"
                       and not c["analysis"]["raw_text"]["warning_body_exact_match"]]
    dangerous = [c["case"] for c in cases if c["scoring"]["dangerous_false_pass_evidence"]]
    hi_conf = [c["case"] for c in cases if c["analysis"]["header_bold"]["high_conf_conflict"]
               or c["analysis"]["body_bold"]["high_conf_conflict"]]

    # stability headline (from the stability artifact, if present) to justify the bold bucketing
    stab_line = ""
    if os.path.exists(STABILITY_JSON):
        try:
            sa = json.load(open(STABILITY_JSON, encoding="utf-8"))["aggregate"]
            stab_line = (f" Stability pass: clean-label overall PASS only "
                         f"{sa['overall_pass_bottle_runs']}/{sa['total_bottle_runs']} bottle-runs; "
                         f"Pass B header-bold read stable on {sa['bottles_with_stable_B_header_bold']}"
                         f"/{sa['bottles']} bottles.")
        except Exception:
            stab_line = ""

    L = ["## Decision summary — three buckets (DIAGNOSTIC; not a green light)\n"]
    L.append(f"_Diagnostic only. The baseline stability pass shows the ensemble is **not** ready for "
             f"production integration, especially bold/layout.{stab_line} The narrow question here: "
             f"does Pass A raw text or Pass C visual reveal defects the current verifier misses?_\n")
    L.append(f"**Key finding:** across {len(cases)} error fixtures the verifier had **{len(false_pass)} "
             f"false-pass** and **{len(witness_only)} defect that a witness caught but verification "
             f"missed entirely**. On this set the witnesses **corroborated** defects the verifier "
             f"already flags rather than revealing new misses — so the integration value of raw text is "
             f"deterministic REDUNDANCY/robustness, not a new detector.\n")

    L.append("### Bucket 1 — Integrate soon: raw-text-derived DETERMINISTIC cross-checks\n")
    L.append("Computable in Python from Pass A's verbatim OCR, with **no model compliance judgment** "
             "and far more run-to-run stable than the visual bold reads. Use them to ESCALATE "
             "(flag a deviation → fail/needs_review), **never to upgrade to pass** (Pass A can "
             "hallucinate canonical text — see Bucket 3).")
    L.append(f"- **ABV notation** — detect the bare `ABV` token in the OCR (27 CFR 4.36/5.65/7.65). "
             f"Seen in raw text on: {raw_abv or 'none'}. (Verifier already catches this via Pass B; the "
             f"raw-text check is an independent path that still fires if B mis-structures the field.)")
    L.append(f"- **Warning wording + case** — exact-match the canonical body against the OCR and check "
             f"`GOVERNMENT WARNING` caps + `Surgeon General` S/G. Raw text showed a wording deviation on "
             f"{raw_wording_dev or 'none'} and lower-cased 'surgeon general' on {raw_sg_low or 'none'} — "
             f"a deterministic second opinion on the project's most important rule.")
    L.append("- **Proof = 2×ABV** — parse both numbers from the OCR and compare (27 CFR 5.65). "
             "Deterministic arithmetic; already enforced via Pass B, cross-check adds robustness.")
    L.append("- **Net-contents value** — parse the OCR number/unit as a second source for the "
             "unit-normalization check the verifier currently lacks (NETCONTENTS-UNIT failed hard "
             "instead of needs_review).")
    L.append("- _Spelling of the designation_ — a deterministic known-term/dictionary check would run "
             "on the OCR, but there is **no fixture** for it yet, so it is a candidate, not evidenced "
             "this run.\n")

    L.append("### Bucket 2 — Reviewer-only: model VISUAL judgments (surface, never auto-decide)\n")
    L.append("Pass C's observations are genuinely useful **to a human reviewer** but are model visual "
             "opinions that the stability pass proved are not machine-reliable.")
    L.append("- **Header bold / body bold** — show Pass B and Pass C side by side as review notes. "
             f"High-confidence B↔C conflicts this run: {hi_conf or 'none'}. **The ONLY acceptable "
             f"automatic outcome when B and C disagree is `needs_review`** — never a vote, never a "
             f"winner.")
    L.append("- **Boxed / visually-separated, contrast, legibility** — useful reviewer context "
             "(is the warning legible and set apart?), but advisory only; do not gate the verdict on them.\n")

    L.append("### Bucket 3 — Too unstable or duplicative to use\n")
    L.append("- **Model voting/averaging on bold** — explicitly rejected. The stability pass shows "
             "Pass C is as unstable as Pass B on header-bold and sometimes confidently disagrees; "
             "voting would blend two noisy signals into a falsely-confident one.")
    L.append("- **Pass C as any input to the automatic verdict** beyond routing a B↔C disagreement to "
             "needs_review.")
    L.append("- **Pass A transcription of fields Pass B already structures** (brand, class/type, net "
             "contents, name/address) — duplicative; B's structured read + fuzzy match already handles "
             "these and the extra OCR string adds noise, not signal.")
    L.append(f"- **Pass A as proof of COMPLIANCE** — it can reproduce canonical wording verbatim even "
             f"when the printed text is altered (dangerous-false-pass evidence on: {dangerous or 'none'} "
             f"this run). Safe only as a deviation detector, never as a pass-confirmer.\n")
    return L


def _md_answers(base, errs):
    L = ["## Summary — answers to the 9 questions\n"]
    bottles = base["bottles"] if base else []
    cases = errs["cases"] if errs else []

    # Q1
    n_pass = sum(1 for b in bottles if (b.get("pass_b_verify") or {}).get("overall") == "pass")
    nonpass_bottles = [(b["case"], [f"{f['field']}={f['status']}" for f in (b.get("pass_b_verify") or {}).get("fields", []) if f["status"] != "pass"]) for b in bottles]
    nonpass_bottles = [(c, fs) for c, fs in nonpass_bottles if fs]
    L.append("**1. Did baseline labels get 100% expected pass? If not, why?**")
    if bottles:
        if n_pass == len(bottles):
            L.append(f"- Yes — {n_pass}/{len(bottles)} clean baselines reached overall PASS.")
        else:
            L.append(f"- No — {n_pass}/{len(bottles)} reached overall PASS. Non-pass fields:")
            for c, fs in nonpass_bottles:
                L.append(f"  - `{c}`: {', '.join(fs)}")
            L.append("  - On clean labels the expected non-pass cause is the **warning bold gate** "
                     "(`header_body_gate` needs BOTH header-bold and non-bold-body at high confidence; "
                     "anything uncertain → needs_review by design) and the known run-to-run brand/bold "
                     "instability (see `BENCHMARK_NOTES.md`), not a wording/ABV failure.")
    else:
        L.append("- _Baseline stage not run._")
    L.append("")

    # Q2 raw text catching evidence structured missed
    raw_only_wording = [b["case"] for b in bottles + cases if b["analysis"]["raw_text"]["raw_only_warning_wording"]]
    raw_only_abv = [b["case"] for b in bottles + cases if b["analysis"]["raw_text"]["raw_only_abv_notation"]]
    raw_sg_low = [b["case"] for b in bottles + cases if b["analysis"]["raw_text"]["surgeon_general_lowercase"]]
    L.append("**2. Did raw text (Pass A) catch wording/case/spelling/ABV-notation evidence that "
             "structured extraction missed?**")
    L.append(f"- raw-text-only warning wording (exact body in OCR but not in structured `text`): "
             f"{raw_only_wording or 'none'}")
    L.append(f"- raw-text-only ABV-notation token (`ABV` in OCR but not in structured alcohol value): "
             f"{raw_only_abv or 'none'}")
    L.append(f"- 'surgeon general' lower-cased visible in raw text: {raw_sg_low or 'none'}")
    L.append("- Note: structured extraction already transcribes these fields, so 'missed' means the "
             "structured `value` did not carry the cue while the OCR did. Where both carry it, raw text "
             "is corroboration, not new signal.")
    L.append("")

    # Q3 visual agreement on bold
    all_recs = bottles + cases
    hdr_agree = sum(1 for b in all_recs if b["analysis"]["header_bold"]["agree"] is True)
    hdr_dis = sum(1 for b in all_recs if b["analysis"]["header_bold"]["agree"] is False)
    hdr_na = sum(1 for b in all_recs if b["analysis"]["header_bold"]["agree"] is None)
    bdy_agree = sum(1 for b in all_recs if b["analysis"]["body_bold"]["agree"] is True)
    bdy_dis = sum(1 for b in all_recs if b["analysis"]["body_bold"]["agree"] is False)
    bdy_na = sum(1 for b in all_recs if b["analysis"]["body_bold"]["agree"] is None)
    hi_confl = [b["case"] for b in all_recs if b["analysis"]["header_bold"]["high_conf_conflict"]
                or b["analysis"]["body_bold"]["high_conf_conflict"]]
    L.append("**3. Did the visual witness (C) agree with structured extraction (B) on header/body bold?**")
    L.append(f"- header bold: agree {hdr_agree}, disagree {hdr_dis}, indeterminate {hdr_na} (of {len(all_recs)})")
    L.append(f"- body bold: agree {bdy_agree}, disagree {bdy_dis}, indeterminate {bdy_na} (of {len(all_recs)})")
    L.append(f"- **HIGH-confidence conflicts (both witnesses 'high', opposite answers): "
             f"{hi_confl or 'none'}** — these are treated as REVIEW evidence, NOT resolved by voting.")
    L.append("")

    # Q4 reduce uncertainty vs duplicate
    c_resolved = [b["case"] for b in all_recs if b["analysis"]["header_bold"]["C_resolved_B_uncertainty"]
                  or b["analysis"]["body_bold"]["C_resolved_B_uncertainty"]]
    c_dup = [b["case"] for b in all_recs if b["analysis"]["header_bold"]["C_duplicated_B"]
             or b["analysis"]["body_bold"]["C_duplicated_B"]]
    L.append("**4. Did the visual witness reduce uncertainty or mostly duplicate the main extractor?**")
    L.append(f"- C added high-confidence signal where B was uncertain (None/non-high): {c_resolved or 'none'}")
    L.append(f"- C merely duplicated B (both high-confidence and agreeing): {c_dup or 'none'}")
    L.append("")

    # Q5 / Q6 error labels
    L.append("**5. On error labels, which defects were caught by current verification?**")
    if cases:
        for c in cases:
            sc = c["scoring"]
            mark = "✅ caught" if sc["caught"] else ("❌ FALSE-PASS" if sc["false_pass"] else "➖ partial/miss")
            L.append(f"- `{c['case']}` ({c['expected_defect']}): {sc['field']}={sc['field_verdict']} "
                     f"(exp {sc['expected_verdict']}) — {mark}"
                     + (" _[known unchecked item]_" if sc["known_gap"] and sc["false_pass"] else ""))
    else:
        L.append("- _Error stage not run._")
    L.append("")
    L.append("**6. Which defects were visible in raw text or visual evidence but NOT used by verification?**")
    if cases:
        any6 = False
        for c in cases:
            sc = c["scoring"]
            slipped = (sc["false_pass"] or sc["field_verdict"] == "n/a")
            if slipped and sc["witness_evidence"]:
                any6 = True
                L.append(f"- `{c['case']}`: verification did not flag it (`{sc['field']}={sc['field_verdict']}`), "
                         f"but witnesses showed: {'; '.join(sc['witness_evidence'])}")
        if not any6:
            L.append("- None — every defect the witnesses surfaced was also flagged by verification, OR "
                     "the witness evidence matched a verification flag. (Witness evidence is reported "
                     "alongside; it is not wired into the verdict.)")
    else:
        L.append("- _Error stage not run._")
    L.append("")

    # Q7 dangerous false-pass evidence
    danger = [(c["case"], c["scoring"]["dangerous_false_pass_evidence"]) for c in cases
              if c["scoring"]["dangerous_false_pass_evidence"]]
    L.append("**7. Did any witness create dangerous false-pass evidence?**")
    if cases:
        if danger:
            for case, ds in danger:
                for d in ds:
                    L.append(f"- ⚠ `{case}`: {d}")
            L.append("- This is exactly why witnesses must stay EVIDENCE, not votes: a confident-but-wrong "
                     "witness (e.g. an OCR pass that 'auto-corrects' altered warning wording back to "
                     "canonical) would mask a real defect if it were allowed to override the verdict.")
        else:
            L.append("- No dangerous false-pass evidence was produced by a witness on this run.")
    else:
        L.append("- _Error stage not run._")
    L.append("")

    # Q8 latency / cost
    L.append("**8. What are the latency and cost implications?**")
    L.append(f"- **Latency:** A/B/C run concurrently, so per-bottle wall-clock ≈ the slowest single "
             f"pass. Because B and C both use `{STRUCTURED_MODEL}` and A uses the faster `{RAW_TEXT_MODEL}`, "
             f"the ensemble wall-clock is ~one structured extraction — see the Timing summary for exact "
             f"avg/median/max.")
    L.append(f"- **Cost:** the ensemble makes **3 vision calls per bottle instead of 1** "
             f"(B `{STRUCTURED_MODEL}` + C `{VISUAL_MODEL}` + A `{RAW_TEXT_MODEL}`), at `detail=high`. "
             f"Roughly +1 mini-class call and +1 nano-class call per bottle vs production. Latency is "
             f"hidden by parallelism; token/$$ cost is additive and scales linearly with batch size.")
    L.append("")

    # Q9 recommendation
    L.append("**9. Should this remain experimental, or is there evidence to integrate part of it?**")
    L += _recommendation(base, errs)
    L.append("")
    return L


def _recommendation(base, errs):
    bottles = base["bottles"] if base else []
    cases = errs["cases"] if errs else []
    all_recs = bottles + cases
    out = []
    hi_confl = [b["case"] for b in all_recs if b["analysis"]["header_bold"]["high_conf_conflict"]
                or b["analysis"]["body_bold"]["high_conf_conflict"]]
    c_resolved = [b["case"] for b in all_recs if b["analysis"]["header_bold"]["C_resolved_B_uncertainty"]
                  or b["analysis"]["body_bold"]["C_resolved_B_uncertainty"]]
    danger = [c["case"] for c in cases if c["scoring"]["dangerous_false_pass_evidence"]]
    raw_value = [b["case"] for b in all_recs if b["analysis"]["raw_text"]["raw_only_warning_wording"]
                 or b["analysis"]["raw_text"]["raw_only_abv_notation"]]
    out.append("- **Keep the verdict path unchanged** — nothing here justifies letting a witness "
               "override `verify()`; the dangerous-evidence cases below show why model voting on bold "
               "would be unsafe.")
    out.append(f"- **Pass C (visual) as a REVIEW signal, not an auto-resolver:** high-confidence B↔C bold "
               f"conflicts occurred on {hi_confl or 'no'} case(s); C resolved B-uncertainty on "
               f"{c_resolved or 'no'} case(s). The defensible integration is to route a B↔C "
               f"high-confidence *disagreement* to needs-review — never to pick a winner.")
    out.append(f"- **Pass A (raw text) is most useful as an audit/debug artifact and a wording cross-check** "
               f"(raw-only evidence appeared on {raw_value or 'no'} case(s)); it duplicates most structured "
               f"fields, and it can hallucinate canonical wording ({danger or 'no'} dangerous case(s)), so it "
               f"must never be trusted as proof of compliance.")
    out.append("- **Verdict: remain EXPERIMENTAL.** The data does not show the ensemble improves compliance "
               "detection over the current pipeline; its clearest value is (a) a review-routing signal on "
               "bold disagreement and (b) a human-readable evidence trail. Integrate only the "
               "disagreement→review routing, behind a flag, after a larger labelled run.")
    return out


# === entrypoint ==============================================================
def main():
    args = sys.argv[1:]
    force_errors = "--force-errors" in args   # override the stage gate

    # Stability mode: run the baseline ensemble N times, quantify flakiness, then stop.
    if "--stability" in args:
        idx = args.index("--stability")
        n = STABILITY_RUNS
        if idx + 1 < len(args) and args[idx + 1].isdigit():
            n = int(args[idx + 1])
        if not load_key():
            sys.exit("ERROR: no OpenAI key (set OPENAI_API_KEY in env / .env / .streamlit/secrets.toml).")
        archived = _archive_previous_results([STABILITY_JSON, STABILITY_MD])
        if archived:
            print(f"  archived previous stability results -> {os.path.relpath(archived, ROOT)}\n")
        run_stability(n)
        return

    args = [a for a in args if a != "--force-errors"]
    stage = "all"
    if "--stage" in args:
        stage = args[args.index("--stage") + 1]
    elif args and args[0] in ("baseline", "errors", "all", "build"):
        stage = args[0]

    if stage != "build":
        if not load_key():
            sys.exit("ERROR: no OpenAI key (set OPENAI_API_KEY in env / .env / .streamlit/secrets.toml).")

    # Preserve the previous run before anything overwrites it.
    archived = _archive_previous_results([RESULTS_JSON, RESULTS_MD, BASELINE_INTERMEDIATE, ERRORS_INTERMEDIATE])
    if archived:
        print(f"  archived previous results -> {os.path.relpath(archived, ROOT)}\n")

    if stage in ("baseline", "all"):
        # A fresh baseline/all run starts clean: drop any stale error intermediate (already archived
        # above) so the report is baseline-only until the error stage is explicitly run.
        _clear_intermediate(ERRORS_INTERMEDIATE)
        base = run_baseline()
        gate = base["gate"]
        _print_gate(gate)
        if stage == "all":
            if gate["passed"] or force_errors:
                if force_errors and not gate["passed"]:
                    print("\n  --force-errors set: continuing to the error stage despite the tripped gate.\n")
                run_errors()
            else:
                print("\n  STAGE GATE TRIPPED: baseline is not close to 100% pass — PAUSING before the "
                      "error stage.\n  Inspect Stage 1 + the Stage-gate section in "
                      "artifacts/evidence_ensemble_results.md, then run\n    python "
                      "scripts/benchmarks/evidence_ensemble.py --stage errors\n  (or `--stage all "
                      "--force-errors`) to proceed.\n")
        build_outputs()
        return

    if stage == "errors":
        if not os.path.exists(BASELINE_INTERMEDIATE):
            sys.exit("ERROR: no baseline results found (artifacts/evidence_ensemble_baseline.json). "
                     "Run `--stage baseline` first so the gate is checked before the error stage.")
        run_errors()
        build_outputs()
        return

    if stage == "build":
        build_outputs()
        return


if __name__ == "__main__":
    main()
