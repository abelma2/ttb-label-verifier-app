"""Benchmark candidate fixes for the government-warning BOLD judgment, BEFORE any of them
touch production code.

Background (see BENCHMARK_NOTES.md): bold is the one warning property the model can't read
reliably -- and the scary failure mode is *confident wrongness* (it called the known
regular-weight 03_notbold header "bold" with high confidence). So before we add fields to the
real extraction schema or wire an escalation/crop pass into the app, we test here whether any
candidate actually:

  1. keeps 01_compliant  -> PASS
  2. keeps 03_notbold    -> FAIL   (the bold trap)
  3. makes realistic baselines FALSE-FAIL LESS
  4. introduces NO new false-passes (02_titlecase / 03_notbold / 04_reworded must never PASS)

This script changes NOTHING in extraction.py / verification.py. It elicits the richer schema
with its OWN warning-only prompt and scores it here. Only a variant that wins these four
criteria should be promoted into production (with tests).

The variants differ ONLY in the bold gate (wording/caps/Surgeon-General reuse the real
verification logic so the comparison is fair):

  A  baseline      production schema (header_bold + header_bold_confidence), confidence_gate
  B  rich          + header_body_comparable + body_not_bold + formatting_quality; AND-of-evidence
                   gate. stroke_weight_observation is TELEMETRY ONLY -- never gated (it just
                   mirrors the boolean; that's the trap we're avoiding).
  C  rich + crops  same schema/gate as B, but an overlapping grid of the back-label image is
                   appended to the input (more pixels on the small warning text -- the only
                   lever likely to help, per BENCHMARK_NOTES: bold is perception, not reasoning).
  D  candidate     production schema + formatting_quality only (the one field that proved stable
                   on real labels). Gate routes legibility, not self-confidence: marginal/unusable
                   -> REVIEW (human verifies a clearer image), clear -> trust the boolean
                   (True->PASS, False->FAIL). Turns confident-wrong bold false-fails on
                   blurry-but-compliant labels into honest reviews, without auto-passing violations.

Usage (calls the real model -- needs an API key and costs money):
  python scripts/benchmarks/bold_variant_benchmark.py                      # A,B,C on adversarial + baselines, 3x
  python scripts/benchmarks/bold_variant_benchmark.py --variants a,b       # subset of variants
  python scripts/benchmarks/bold_variant_benchmark.py --adv --runs 5       # adversarial only, 5x stability
  python scripts/benchmarks/bold_variant_benchmark.py --real --runs 2      # 13 realistic photographed labels
  python scripts/benchmarks/bold_variant_benchmark.py --adv --variants a,d --runs 5   # validate candidate gate D
  python scripts/benchmarks/bold_variant_benchmark.py --grid 3x3 --detail high
  python scripts/benchmarks/bold_variant_benchmark.py --model gpt-5.4-mini --escalate-model gpt-5.5
                                                                 # B/C use the escalate model; A uses --model

Each crop adds an image block, so variant C with a 3x3 grid sends ~9 extra images PER run --
mind the token cost; that's why --runs defaults low and the grid is configurable.
"""
import base64
import io
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime

import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT
OUT_DIR = os.path.join(ROOT, "output")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rapidfuzz import fuzz

from config import EXTRACTION_MODEL
from extraction import _get_client, _model_params, _create_with_fallbacks
from verification import (
    _normalize, _warning_body, _CANONICAL_WARNING_BODY_NORM,
    WARNING_WORDING_REVIEW_FLOOR, PASS, REVIEW, FAIL,
)

ADV = os.path.join(ROOT, "adversarial")
BASE = os.path.join(ROOT, "test_labels", "baseline_labels")

# Ground truth (same cases the warning-check benchmark uses). expected = the WHOLE-warning
# verdict we want; None = a realistic compliant label (no certifiable bold GT) that simply
# must not FALSE-FAIL.
ADV_CASES = [
    ("01_compliant", [os.path.join(ADV, "01_compliant.png")], PASS),
    ("02_titlecase", [os.path.join(ADV, "02_titlecase.png")], FAIL),   # caps violation
    ("03_notbold",   [os.path.join(ADV, "03_notbold.png")],   FAIL),   # the bold trap
    ("04_reworded",  [os.path.join(ADV, "04_reworded.png")],  FAIL),   # wording violation
]
BASE_CASES = [
    ("baseline_1", [os.path.join(BASE, "baseline_1_Front.png"), os.path.join(BASE, "baseline_1_Other.png")], None),
    ("baseline_2", [os.path.join(BASE, "baseline_2_Front.png"), os.path.join(BASE, "baseline_2_Other.png")], None),
    ("baseline_3", [os.path.join(BASE, "baseline_3_Front.png"), os.path.join(BASE, "baseline_3_Other.png")], None),
]
# Realistic photographed labels (test_1..test_13, _Front/_Other .jpeg). expected=None: compliant,
# so they must not FALSE-FAIL -- a bigger sample for the false-fail-rate question than the 3 baselines.
REAL = os.path.join(ROOT, "test_labels", "real_labels")
REAL_CASES = [(f"test_{n}", [os.path.join(REAL, f"test_{n}_Front.jpeg"),
                             os.path.join(REAL, f"test_{n}_Other.jpeg")], None)
              for n in range(1, 14)]
_AB = {PASS: "PASS", REVIEW: "REVIEW", FAIL: "FAIL"}


# --- warning-only prompts + strict schemas (Structured Outputs) ---------------

_COMMON = (
    "You are reading ONLY the U.S. government health warning on an alcohol label. Look at the "
    "image(s) and report what you SEE. Do NOT judge legal compliance -- just transcribe and "
    "describe. The warning is often on the back/'other' label and printed small.\n"
)
PROMPT_A = _COMMON + (
    "Report: present (is the GOVERNMENT WARNING statement on the label?); text (transcribe the "
    "full warning VERBATIM, including the 'GOVERNMENT WARNING:' header if you can see it); "
    "header_all_caps (are the words 'GOVERNMENT WARNING' in ALL CAPITAL letters?); header_bold "
    "(is the 'GOVERNMENT WARNING' header in BOLD -- heavier strokes than the warning body?); "
    "header_bold_confidence (high/medium/low). Use null for header_bold if you truly cannot tell."
)
PROMPT_RICH = _COMMON + (
    "Compare the stroke weight of the 'GOVERNMENT WARNING' header against the body text of the "
    "SAME warning. Report: present; text (VERBATIM, with the header if visible); header_all_caps; "
    "header_bold (is the header in BOLD/heavier strokes?); header_body_comparable (can you see "
    "BOTH the header and the body clearly enough to actually compare their stroke weights?); "
    "body_not_bold (is the body/remainder NOT bold -- i.e. lighter than the header?); "
    "stroke_weight_observation (one short sentence describing what you literally see about the "
    "relative stroke weights); formatting_quality ('clear' = weights clearly legible, 'marginal' "
    "= small/blurry but partly judgable, 'unusable' = too small/blurry/glare to judge weight). "
    "Use null for any boolean you genuinely cannot determine."
)
# Variant D -- the candidate production change: production bold question PLUS the ONE field that
# proved stable in the real-label run (formatting_quality). No body-comparison / stroke prose
# (those were as noisy as the bare boolean).
PROMPT_D = _COMMON + (
    "Report: present; text (transcribe the warning VERBATIM, with the 'GOVERNMENT WARNING:' "
    "header if visible); header_all_caps (are the words 'GOVERNMENT WARNING' in ALL CAPITAL "
    "letters?); header_bold (is the 'GOVERNMENT WARNING' header in BOLD -- heavier strokes than "
    "the warning body?); formatting_quality ('clear' = the header's stroke weight is clearly "
    "legible in this image; 'marginal' = small/blurry/low-contrast so the weight is hard to "
    "judge; 'unusable' = too small/blurry/glare to judge stroke weight at all). Use null for "
    "header_bold if you truly cannot tell."
)

# Variant E -- the MULTI-PROPERTY de-priming idea: classify several ORTHOGONAL typographic
# properties (bold / italic / underline) on the header AND the body, as strict binary yes/no.
# Hypothesis: a model forced to distinguish bold from italic from underline (rather than answer a
# lone "is it bold?") has to actually LOOK and can't fall back on the "warnings are bold" prior.
# Bold flags are non-null binaries (per the proposal); abstention is carried by a SEPARATE
# formatting_legibility field so genuine uncertainty still routes to REVIEW. italic/underline are
# de-priming DISTRACTORS -- captured, never gated (not in 27 CFR 16.22).
PROMPT_E = _COMMON + (
    "Judge the TYPOGRAPHY of the government warning, looking at TWO parts separately: the HEADER "
    "(the words 'GOVERNMENT WARNING') and the BODY (the sentences after it). For EACH part decide "
    "three INDEPENDENT properties as a strict yes/no from what you literally see:\n"
    "  bold = letter strokes visibly THICKER / heavier than a normal book-weight font;\n"
    "  italic = letters visibly SLANTED;\n"
    "  underline = a visible LINE under the text.\n"
    "Judge bold, italic and underline as SEPARATE things -- do not infer one from another, and do "
    "NOT infer bold from capitalization, size, darkness, or expectation (a header is not bold just "
    "because warnings are usually bold). Report: present; text (VERBATIM, with the 'GOVERNMENT "
    "WARNING:' header if visible); header_all_caps; header_bold; header_italic; header_underline; "
    "body_bold; body_italic; body_underline; and formatting_legibility ('clear' = strokes/slant "
    "clearly legible; 'marginal' = small/blurry/low-contrast so formatting is hard to judge; "
    "'unusable' = too small/blurry/glare to judge formatting at all)."
)

# ---- Workflow-designed approaches (design-bold-approaches; ranked by an adversarial critic) ----
# F relative_scale  : remove the word "bold" from the answer space; rate RELATIVE stroke weight on a
#                     symmetric ordinal scale; derive BOTH rules in Python (header_vs_body, body_vs_surround).
# G describe_first  : force a stroke DESCRIPTION + magnitude before any judgment; gate keys off the
#                     ordinal header-vs-body comparison (the benchmark-proven anti-false-pass move), not a boolean.
# H weight_gap      : QUANTITATIVE twin stroke-weight estimates (200-800 CSS scale); gate reads only the
#                     signed gap + an independent body-weight FLOOR (the all-bold-body structural rule).
# I self_consistency: 5 INDEPENDENT in-call bold reads; empirical agreement (computed, not self-reported)
#                     is the confidence; body-anchored PASS. Targets gpt-5.x run-to-run noise.

PROMPT_F = _COMMON + (
    "Do NOT decide whether anything is \"bold\" and do NOT judge compliance -- only RATE and COMPARE the "
    "visible thickness (stroke weight) of the printed letters, independent of LETTER SIZE, CAPITALIZATION, "
    "darkness, contrast, or expectation. Big or capital letters are NOT automatically heavier.\n"
    "Three regions: HEADER = the words 'GOVERNMENT WARNING'; BODY = the sentences after it ('(1) According "
    "to the Surgeon General...'); SURROUNDING PRINT = ordinary nearby text that is NOT the warning (net "
    "contents, address, 'PRODUCT OF...', barcode numerals) -- your reference for ordinary print.\n"
    "Rate each comparison on: \"much_lighter\" | \"lighter\" | \"same\" | \"heavier\" | \"much_heavier\" | \"uncertain\".\n"
    "  - header_vs_body: how thick are the HEADER strokes vs the BODY strokes?\n"
    "  - body_vs_surround: how thick are the BODY strokes vs the SURROUNDING ordinary print? If you cannot "
    "find ordinary non-warning print, set body_vs_surround='same' and surround_available=false.\n"
    "Then: scale_confidence (high/medium/low -- how reliably you could SEE and compare the stroke widths); "
    "comparison_basis (one short phrase of what you actually saw, or null); surround_available (bool). "
    "If header and body look equally thick, say 'same' -- do not invent a difference. Never report a "
    "difference you do not actually see.\n"
    "Also return present, text (verbatim, header included, no reconstruction), header_all_caps (true/false/null)."
)

PROMPT_G = _COMMON + (
    "BOLD FORMATTING -- DESCRIBE WHAT YOU SEE BEFORE YOU JUDGE. Do NOT decide 'bold/not bold' up front and "
    "do not assume the header is bold because warnings usually are -- look at the actual strokes. Fill IN ORDER:\n"
    "1. header_sample_word: copy ONE short word from the printed 'GOVERNMENT WARNING' header (null if unreadable).\n"
    "2. body_sample_word: copy ONE short word from the body sentences (null if unreadable).\n"
    "3. header_stroke_desc: a few words describing the LETTER STROKES of header_sample_word (e.g. 'thick heavy "
    "filled-in', 'medium even', 'thin light') -- pixels, not expectation.\n"
    "4. body_stroke_desc: same, for body_sample_word.\n"
    "5. header_stroke_weight: \"hairline\"|\"light\"|\"medium\"|\"heavy\"|\"very_heavy\"|\"indeterminate\".\n"
    "6. body_stroke_weight: same scale.\n"
    "7. header_vs_body_weight (THE most important field): \"header_much_heavier\" | \"header_slightly_heavier\" | "
    "\"equal\" | \"body_heavier\" | \"indeterminate\" -- base it on the descriptions above, NOT on which text is a header.\n"
    "8. comparison_confidence: high/medium/low -- how reliably you could SEE and compare the two stroke weights.\n"
    "9. header_bold: DERIVE -- true only if header_stroke_weight heavy/very_heavy AND header_vs_body_weight is "
    "header_much/slightly_heavier; false if equal/body_heavier; null if indeterminate.\n"
    "10. body_bold: DERIVE -- true only if body heavy/very_heavy and as thick as/thicker than the header; false "
    "if body light/medium or thinner; null if indeterminate.\n"
    "Also return present, text (verbatim, header included), header_all_caps (true/false/null). Report only what you SEE."
)

PROMPT_H = _COMMON + (
    "You are measuring STROKE WEIGHT (how thick/heavy the letter strokes are), not size, color, or "
    "capitalization -- stroke weight is the only thing that makes text 'bold'. Do not assume the header is "
    "heavy because warnings usually are; judge only the actual thickness of the strokes.\n"
    "Use this 7-step CSS-style scale: 200=Hairline/Light (thinner than normal); 400=Regular (normal body "
    "text -- the NEUTRAL ANCHOR); 500=Medium; 600=Semibold; 700=Bold (distinctly thick); 800=Heavy/Black.\n"
    "Procedure, in order (the body is your on-image yardstick):\n"
    "  1. Rate the BODY/remainder text (the sentences after 'GOVERNMENT WARNING:') -> body_weight_class.\n"
    "  2. Compare the header 'GOVERNMENT WARNING' strokes DIRECTLY against that body and rate it -> header_weight_class.\n"
    "  3. weight_gap_steps = how many 100-steps heavier the HEADER is than the BODY (header minus body). A bold "
    "header over a regular body is about +3; SAME weight is 0 (do not invent a difference); header lighter is negative.\n"
    "  4. weight_legibility: 'clear' (can compare strokes confidently) | 'marginal' (close/degraded) | "
    "'unreadable' (cannot resolve stroke thickness).\n"
    "Also transcribe the full warning text exactly as printed and report header_all_caps (read from characters, not "
    "weight), and present. If you cannot resolve the strokes, say 'marginal'/'unreadable' -- do not manufacture a gap."
)

PROMPT_I = _COMMON + (
    "Judge the BOLDNESS of the warning in a deliberately repeated, INDEPENDENT way. Stroke weight is a fine "
    "distinction, so instead of deciding once, make several SEPARATE assessments and report every one honestly, "
    "including when they disagree. Do NOT make them agree on purpose and do NOT copy one into the next.\n"
    "Two regulatory weights, OPPOSITE expectations -- judge each on what you SEE: the HEADER 'GOVERNMENT WARNING', "
    "and the BODY (sentences after the header).\n"
    "Produce an array bold_trials of 5 independent trials. For EACH trial, look afresh and fill:\n"
    "  - header_bold: true if the HEADER strokes are visibly thicker/heavier than the body strokes; false if same/"
    "lighter/not heavy; null if you cannot compare this look.\n"
    "  - body_bold: true if the BODY's OWN strokes are visibly thick/heavy; false if normal weight; null if you cannot tell.\n"
    "  - basis: one short phrase of what you ACTUALLY saw this trial (not a rule or expectation).\n"
    "Every trial: compare STROKE WEIGHT only -- do NOT infer bold from capitalization, size, color, darkness, "
    "contrast, or expectation; all-caps is NOT bold; a larger header is NOT bold. If a look is ambiguous, the honest "
    "answer is null -- use it; do not round up to look consistent. Disagreement across trials is the truthful signal.\n"
    "Also return present, text (verbatim, header included, no reconstruction), header_all_caps (true/false/null), "
    "and confidence (high/medium/low transcription confidence)."
)

_SCHEMA_A = {
    "type": "object", "additionalProperties": False,
    "required": ["present", "text", "header_all_caps", "header_bold", "header_bold_confidence"],
    "properties": {
        "present": {"type": "boolean"},
        "text": {"type": ["string", "null"]},
        "header_all_caps": {"type": ["boolean", "null"]},
        "header_bold": {"type": ["boolean", "null"]},
        "header_bold_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
}
_SCHEMA_RICH = {
    "type": "object", "additionalProperties": False,
    "required": ["present", "text", "header_all_caps", "header_bold", "header_body_comparable",
                 "body_not_bold", "stroke_weight_observation", "formatting_quality"],
    "properties": {
        "present": {"type": "boolean"},
        "text": {"type": ["string", "null"]},
        "header_all_caps": {"type": ["boolean", "null"]},
        "header_bold": {"type": ["boolean", "null"]},
        "header_body_comparable": {"type": ["boolean", "null"]},
        "body_not_bold": {"type": ["boolean", "null"]},
        "stroke_weight_observation": {"type": ["string", "null"]},
        "formatting_quality": {"type": "string", "enum": ["clear", "marginal", "unusable"]},
    },
}
_SCHEMA_D = {
    "type": "object", "additionalProperties": False,
    "required": ["present", "text", "header_all_caps", "header_bold", "formatting_quality"],
    "properties": {
        "present": {"type": "boolean"},
        "text": {"type": ["string", "null"]},
        "header_all_caps": {"type": ["boolean", "null"]},
        "header_bold": {"type": ["boolean", "null"]},
        "formatting_quality": {"type": "string", "enum": ["clear", "marginal", "unusable"]},
    },
}

_SCHEMA_E = {
    "type": "object", "additionalProperties": False,
    "required": ["present", "text", "header_all_caps", "header_bold", "header_italic",
                 "header_underline", "body_bold", "body_italic", "body_underline",
                 "formatting_legibility"],
    "properties": {
        "present": {"type": "boolean"},
        "text": {"type": ["string", "null"]},
        "header_all_caps": {"type": ["boolean", "null"]},
        # bold flags are strict binaries (no null) -- the proposal's "0/1, no unsure"
        "header_bold": {"type": "boolean"},
        "header_italic": {"type": "boolean"},
        "header_underline": {"type": "boolean"},
        "body_bold": {"type": "boolean"},
        "body_italic": {"type": "boolean"},
        "body_underline": {"type": "boolean"},
        # abstention lives HERE, not on the per-property flags
        "formatting_legibility": {"type": "string", "enum": ["clear", "marginal", "unusable"]},
    },
}


_ORD5 = {"type": "string",
         "enum": ["much_lighter", "lighter", "same", "heavier", "much_heavier", "uncertain"]}
_SCHEMA_F = {
    "type": "object", "additionalProperties": False,
    "required": ["present", "text", "header_all_caps", "header_vs_body", "body_vs_surround",
                 "surround_available", "scale_confidence", "comparison_basis"],
    "properties": {
        "present": {"type": "boolean"}, "text": {"type": ["string", "null"]},
        "header_all_caps": {"type": ["boolean", "null"]},
        "header_vs_body": _ORD5, "body_vs_surround": _ORD5,
        "surround_available": {"type": "boolean"},
        "scale_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "comparison_basis": {"type": ["string", "null"]},
    },
}
_WEIGHT5 = {"type": "string", "enum": ["hairline", "light", "medium", "heavy", "very_heavy", "indeterminate"]}
_SCHEMA_G = {
    "type": "object", "additionalProperties": False,
    "required": ["present", "text", "header_all_caps", "header_sample_word", "body_sample_word",
                 "header_stroke_desc", "body_stroke_desc", "header_stroke_weight", "body_stroke_weight",
                 "header_vs_body_weight", "comparison_confidence", "header_bold", "body_bold"],
    "properties": {
        "present": {"type": "boolean"}, "text": {"type": ["string", "null"]},
        "header_all_caps": {"type": ["boolean", "null"]},
        "header_sample_word": {"type": ["string", "null"]}, "body_sample_word": {"type": ["string", "null"]},
        "header_stroke_desc": {"type": ["string", "null"]}, "body_stroke_desc": {"type": ["string", "null"]},
        "header_stroke_weight": _WEIGHT5, "body_stroke_weight": _WEIGHT5,
        "header_vs_body_weight": {"type": "string",
            "enum": ["header_much_heavier", "header_slightly_heavier", "equal", "body_heavier", "indeterminate"]},
        "comparison_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "header_bold": {"type": ["boolean", "null"]}, "body_bold": {"type": ["boolean", "null"]},
    },
}
_SCHEMA_H = {
    "type": "object", "additionalProperties": False,
    "required": ["present", "text", "header_all_caps", "header_weight_class", "body_weight_class",
                 "weight_gap_steps", "weight_legibility"],
    "properties": {
        "present": {"type": "boolean"}, "text": {"type": ["string", "null"]},
        "header_all_caps": {"type": ["boolean", "null"]},
        "header_weight_class": {"type": ["integer", "null"]},
        "body_weight_class": {"type": ["integer", "null"]},
        "weight_gap_steps": {"type": ["integer", "null"]},
        "weight_legibility": {"type": "string", "enum": ["clear", "marginal", "unreadable"]},
    },
}
_SCHEMA_I = {
    "type": "object", "additionalProperties": False,
    "required": ["present", "text", "header_all_caps", "bold_trials", "confidence"],
    "properties": {
        "present": {"type": "boolean"}, "text": {"type": ["string", "null"]},
        "header_all_caps": {"type": ["boolean", "null"]},
        # strict Structured Outputs does not support minItems/maxItems -- the prompt asks for 5 and the
        # gate is robust to any length (>= MIN_USABLE trials to act).
        "bold_trials": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["header_bold", "body_bold", "basis"],
            "properties": {"header_bold": {"type": ["boolean", "null"]},
                           "body_bold": {"type": ["boolean", "null"]},
                           "basis": {"type": ["string", "null"]}}}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
}


def _rf(name, schema):
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


# --- bold gates (the ONLY thing that differs between variants) -----------------

def _gate_a(f):
    """Production confidence_gate: trust header_bold only when the model is confident."""
    bold = f.get("header_bold")
    confident = (f.get("header_bold_confidence") or "low") in ("medium", "high")
    if bold is True and confident:
        return PASS, "bold verified (confident)"
    if bold is False and confident:
        return FAIL, "header not bold"
    return FAIL, "cannot verify bold (fail-closed)"

def _gate_rich(f):
    """AND-of-evidence: header bold AND body-comparable AND body-not-bold AND image judgable.
    stroke_weight_observation is NOT consulted -- telemetry only."""
    q = f.get("formatting_quality")
    if q == "unusable":
        return FAIL, "image unusable for bold (fail-closed)"
    bold, comp, bnb = f.get("header_bold"), f.get("header_body_comparable"), f.get("body_not_bold")
    if bold is False:
        return FAIL, "header not bold"
    if bnb is False:
        return FAIL, "body is also bold (header not distinct)"
    if bold is True and comp is True and bnb is True and q in ("clear", "marginal"):
        return PASS, "bold verified (header heavier, body not bold)"
    return FAIL, "cannot verify bold (fail-closed)"

def _gate_d(f):
    """Candidate production gate: route on image legibility, not self-confidence.
    marginal/unusable -> REVIEW (a human verifies a clearer image, instead of auto-failing a
    compliant-but-blurry label). Only a CLEAR image hard-decides: bold True->PASS, False->FAIL.
    This converts the ~18%% real-label bold false-fails into honest reviews WITHOUT auto-passing
    any violation, while keeping the hard FAIL for a confidently-not-bold header on a legible image."""
    q = f.get("formatting_quality")
    if q in ("marginal", "unusable"):
        return REVIEW, f"warning legibility {q}: cannot confirm bold from this image -- verify a clearer image"
    bold = f.get("header_bold")
    if bold is True:
        return PASS, "bold verified (header heavier; image clear)"
    if bold is False:
        return FAIL, "header not bold (image clear)"
    return REVIEW, "image clear but bold not determinable -- please verify"


def _gate_e(f):
    """Multi-property binary gate -- 27 CFR 16.22's TWO rules from the binary header_bold +
    body_bold flags (header must be bold AND the body/remainder must NOT be), with abstention
    carried by formatting_legibility (marginal/unusable -> REVIEW). italic / underline are
    de-priming distractors: captured for telemetry but NEVER gated (not regulatory)."""
    q = f.get("formatting_legibility")
    if q in ("marginal", "unusable"):
        return REVIEW, f"formatting legibility {q}: verify a clearer image"
    hb, bb = f.get("header_bold"), f.get("body_bold")
    if hb is False:
        return FAIL, "header not bold (clear image)"
    if bb is True:
        return FAIL, "body is bold -- remainder must not be bold (clear image)"
    if hb is True and bb is False:
        return PASS, "header bold, body not bold (clear image)"
    return REVIEW, "bold not determinable -- verify"


_HEAVIER = {"heavier", "much_heavier"}
_NOT_HEAVIER = {"much_lighter", "lighter", "same"}


def _gate_f(f):
    """relative_scale: derive BOTH rules from two ordinal comparisons; only high scale_confidence acts."""
    hvb, bvs = f.get("header_vs_body"), f.get("body_vs_surround")
    surround = bool(f.get("surround_available"))
    if (f.get("scale_confidence") or "low") != "high":
        return REVIEW, "could not compare the warning's stroke weights with high confidence"
    if hvb in (None, "uncertain"):
        return REVIEW, "could not compare header vs body stroke weight"
    header_not_bold = hvb in _NOT_HEAVIER
    if surround and bvs not in (None, "uncertain"):
        body_ok, body_bold, body_resolved = bvs in _NOT_HEAVIER, bvs in _HEAVIER, True
    else:
        body_ok = body_bold = body_resolved = False
    if header_not_bold and body_bold:
        return FAIL, "header not heavier than body (not bold) AND body heavier than surrounding print (body bold)"
    if header_not_bold:
        return FAIL, "'GOVERNMENT WARNING' is not heavier than the body (header does not appear bold)"
    if body_bold:
        return FAIL, "the body is heavier than the surrounding print (the remainder appears bold)"
    if hvb in _HEAVIER and body_resolved and body_ok:
        return PASS, "bold header (heavier than body) and non-bold body (not heavier than surrounding print) verified"
    return REVIEW, "could not confirm BOTH bold rules from the stroke-weight comparison -- please verify"


def _gate_g(f):
    """describe_first: gate keys off the ordinal header_vs_body_weight comparison (not the boolean)."""
    cmp, hw, bw = f.get("header_vs_body_weight"), f.get("header_stroke_weight"), f.get("body_stroke_weight")
    conf = f.get("comparison_confidence")
    HEAVY = {"heavy", "very_heavy"}
    if cmp in (None, "indeterminate") or hw == "indeterminate" or bw == "indeterminate" or conf == "low":
        return REVIEW, "could not confirm bold formatting with high confidence -- please verify"
    header_not_bold = cmp in ("equal", "body_heavier") or hw in ("hairline", "light")
    body_is_bold = (bw in HEAVY) and cmp in ("equal", "body_heavier")
    if header_not_bold and body_is_bold:
        return FAIL, "header not bold (not heavier than body) AND body appears bold"
    if header_not_bold:
        return FAIL, "'GOVERNMENT WARNING' does not appear bold (not heavier than the body)"
    if body_is_bold:
        return FAIL, "the body/remainder appears to be in bold"
    header_bold_ok = cmp in ("header_much_heavier", "header_slightly_heavier") and hw in HEAVY
    consistent = (f.get("header_bold") is not False) and (f.get("body_bold") is not True)
    if header_bold_ok and bw not in HEAVY and consistent and conf == "high":
        return PASS, "bold header and non-bold body verified (header heavier than body, body not heavy)"
    return REVIEW, "could not confirm BOTH bold rules with high confidence -- please verify"


# weight_gap thresholds (would live in config in production)
_BODY_BOLD_FLOOR, _HEADER_BOLD_MIN, _MIN_BOLD_GAP, _GAP_SLACK = 700, 700, 2, 1


def _gate_h(f):
    """weight_gap: signed gap (relative) for rule 1; independent body-weight floor for rule 2."""
    leg, hw, bw, gap = (f.get("weight_legibility"), f.get("header_weight_class"),
                        f.get("body_weight_class"), f.get("weight_gap_steps"))
    if leg != "clear":
        return REVIEW, f"stroke weight not clearly legible ({leg}) -- please verify"
    if hw is None or bw is None or gap is None:
        return REVIEW, "model abstained on a stroke-weight estimate -- please verify"
    if abs((hw - bw) / 100 - gap) > _GAP_SLACK:
        return REVIEW, "the two weight estimates contradict the reported gap -- please verify"
    if bw >= _BODY_BOLD_FLOOR:
        return FAIL, "the body/remainder is itself bold (body weight at/above bold)"
    if hw >= _HEADER_BOLD_MIN and gap >= _MIN_BOLD_GAP:
        return PASS, "bold header clearly heavier than a confirmed non-bold body (weight gap verified)"
    if gap <= 0 and hw < _HEADER_BOLD_MIN:
        return FAIL, "'GOVERNMENT WARNING' is not heavier than the body and not bold"
    return REVIEW, "stroke weights measured but inconclusive -- please verify"


_AGREE_DECIDE, _MIN_USABLE = 4, 3


def _gate_i(f):
    """self_consistency: empirical agreement across 5 independent in-call reads; body-anchored PASS."""
    trials = [t for t in (f.get("bold_trials") or []) if isinstance(t, dict)]
    if len(trials) < _MIN_USABLE:
        return REVIEW, "not enough independent bold reads to judge -- please verify"
    H = [t.get("header_bold") for t in trials if t.get("header_bold") is not None]
    Bv = [t.get("body_bold") for t in trials if t.get("body_bold") is not None]
    h_bold, h_not = H.count(True), H.count(False)
    b_bold, b_not = Bv.count(True), Bv.count(False)
    header_not_bold = len(H) >= _MIN_USABLE and h_not >= _AGREE_DECIDE
    body_bold = len(Bv) >= _MIN_USABLE and b_bold >= _AGREE_DECIDE
    if header_not_bold and body_bold:
        return FAIL, "5-read agreement: header not bold AND body bold"
    if header_not_bold:
        return FAIL, "'GOVERNMENT WARNING' does not appear bold (independent reads agreed)"
    if body_bold:
        return FAIL, "the body appears bold (independent reads agreed)"
    if (len(H) >= _MIN_USABLE and h_bold >= _AGREE_DECIDE) and (len(Bv) >= _MIN_USABLE and b_not >= _AGREE_DECIDE):
        return PASS, "bold header and non-bold body verified (5 independent reads agreed)"
    return REVIEW, "independent bold reads did not agree -- please verify"


def _judge(f, gate):
    """Full warning verdict reusing the real wording/caps/Surgeon-General logic, then the
    variant's bold gate. Returns (status, reason, cause)."""
    text = f.get("text")
    if not f.get("present") or not text:
        return FAIL, "no warning found", "presence"
    body_norm = _normalize(_warning_body(text))
    if body_norm != _CANONICAL_WARNING_BODY_NORM:
        score = fuzz.ratio(body_norm, _CANONICAL_WARNING_BODY_NORM)
        if score >= WARNING_WORDING_REVIEW_FLOOR:
            return REVIEW, f"wording near-miss ({score:.0f})", "wording"
        return FAIL, f"wording mismatch ({score:.0f})", "wording"
    caps = f.get("header_all_caps")
    m = re.search(r"government\s+warning", text, re.IGNORECASE)
    header_caps = text[m.start():m.end()].isupper() if m else caps
    if header_caps is False or caps is False:
        return FAIL, "header not in capital letters", "caps"
    if header_caps is None:
        return REVIEW, "caps unconfirmed", "caps"
    for w in ("surgeon", "general"):
        mw = re.search(w, text, re.IGNORECASE)
        if mw and text[mw.start()].islower():
            return FAIL, "Surgeon/General not capitalized", "surgeon"
    status, why = gate(f)
    return status, why, "bold"


# --- image plumbing -----------------------------------------------------------

def _media_type(path):
    return "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"


def _block(b, media_type, detail):
    return {"type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{base64.b64encode(b).decode()}",
                          "detail": detail}}

def _grid_crops(b, rows, cols, overlap=0.2):
    """Overlapping grid crops of one image (PNG bytes out). More pixels on the warning band
    without needing to know where it is."""
    from PIL import Image
    im = Image.open(io.BytesIO(b)).convert("RGB")
    W, H = im.size
    cw, ch = W / cols, H / rows
    ow, oh = cw * overlap, ch * overlap
    out = []
    for r in range(rows):
        for c in range(cols):
            box = (max(0, int(c * cw - ow)), max(0, int(r * ch - oh)),
                   min(W, int((c + 1) * cw + ow)), min(H, int((r + 1) * ch + oh)))
            buf = io.BytesIO()
            im.crop(box).save(buf, format="PNG")
            out.append(buf.getvalue())
    return out


# --- one model call -----------------------------------------------------------

def _call(model, prompt, rf, image_blocks):
    params = _model_params(model, response_format=rf)
    content = [{"type": "text", "text": prompt}] + image_blocks
    t = time.perf_counter()
    resp = _create_with_fallbacks(_get_client(), content, params)
    dt = time.perf_counter() - t
    raw = resp.choices[0].message.content
    return json.loads(raw), dt


def _run(variant, paths, grid, detail, models):
    """Run one variant on one case once. Returns a record dict."""
    imgs = [(open(p, "rb").read(), _media_type(p)) for p in paths]
    blocks = [_block(b, mt, detail) for b, mt in imgs]
    if variant == "a":
        model, prompt, rf, gate, crops = models["a"], PROMPT_A, _rf("warn_a", _SCHEMA_A), _gate_a, False
    elif variant == "d":   # candidate production gate (formatting_quality -> REVIEW)
        model, prompt, rf, gate, crops = models["a"], PROMPT_D, _rf("warn_d", _SCHEMA_D), _gate_d, False
    elif variant == "e":   # multi-property (bold/italic/underline x header/body) binary
        model, prompt, rf, gate, crops = models["a"], PROMPT_E, _rf("warn_e", _SCHEMA_E), _gate_e, False
    elif variant == "f":   # relative_scale (ordinal comparisons, both rules derived)
        model, prompt, rf, gate, crops = models["a"], PROMPT_F, _rf("warn_f", _SCHEMA_F), _gate_f, False
    elif variant == "g":   # describe_first (description -> ordinal comparison is the gated signal)
        model, prompt, rf, gate, crops = models["a"], PROMPT_G, _rf("warn_g", _SCHEMA_G), _gate_g, False
    elif variant == "h":   # weight_gap (quantitative signed gap + body-weight floor)
        model, prompt, rf, gate, crops = models["a"], PROMPT_H, _rf("warn_h", _SCHEMA_H), _gate_h, False
    elif variant == "i":   # self_consistency (5 in-call reads, agreement = confidence)
        model, prompt, rf, gate, crops = models["a"], PROMPT_I, _rf("warn_i", _SCHEMA_I), _gate_i, False
    else:                  # b (rich) / c (rich + crops)
        model, prompt, rf, gate, crops = models["rich"], PROMPT_RICH, _rf("warn_rich", _SCHEMA_RICH), _gate_rich, (variant == "c")
    if crops:              # append crops of the LAST image (the back/Other label)
        rows, cols = grid
        blocks += [_block(cb, "image/png", detail) for cb in _grid_crops(imgs[-1][0], rows, cols)]
    fields, dt = _call(model, prompt, rf, blocks)
    status, reason, cause = _judge(fields, gate)
    return {"status": status, "reason": reason, "cause": cause, "seconds": round(dt, 2),
            "fields": fields, "model": model, "n_blocks": len(blocks)}


# --- main ---------------------------------------------------------------------

def _load_key():
    if os.environ.get("OPENAI_API_KEY"):
        return
    secrets = os.path.join(ROOT, ".streamlit", "secrets.toml")
    if os.path.exists(secrets):
        with open(secrets, encoding="utf-8") as fh:
            m = re.search(r'OPENAI_API_KEY\s*=\s*"([^"]+)"', fh.read())
        if m and m.group(1) and m.group(1) != "sk-...":
            os.environ["OPENAI_API_KEY"] = m.group(1)


def _arg(args, flag, default):
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else default


def main():
    args = sys.argv[1:]
    runs = int(_arg(args, "--runs", "3"))
    detail = _arg(args, "--detail", "high")
    grid = tuple(int(x) for x in _arg(args, "--grid", "3x3").lower().split("x"))
    variants = [v.strip() for v in _arg(args, "--variants", "a,b,c").split(",") if v.strip()]
    base_model = _arg(args, "--model", EXTRACTION_MODEL)
    models = {"a": base_model, "rich": _arg(args, "--escalate-model", base_model)}

    if "--adv" in args:
        cases = ADV_CASES
    elif "--baseline" in args:
        cases = BASE_CASES
    elif "--real" in args:
        cases = REAL_CASES
    else:
        cases = ADV_CASES + BASE_CASES

    _load_key()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: no OpenAI key (env OPENAI_API_KEY or .streamlit/secrets.toml).")

    print(f"variants={variants}  runs={runs}  detail={detail}  grid={grid[0]}x{grid[1]}  "
          f"A-model={models['a']}  B/C-model={models['rich']}\n")

    # records[variant][case_id] = list of run records
    records = {v: {} for v in variants}
    for v in variants:
        for cid, paths, expected in cases:
            runs_out = []
            for i in range(runs):
                print(f"  variant {v.upper()}  {cid}  run {i + 1}/{runs} ...", flush=True)
                try:
                    rec = _run(v, paths, grid, detail, models)
                except Exception as exc:
                    rec = {"status": "ERR", "reason": str(exc)[:160], "cause": "error",
                           "seconds": None, "fields": {}, "model": models.get("rich")}
                runs_out.append(rec)
            records[v][cid] = {"expected": expected, "runs": runs_out}

    # ---- scorecard ----
    exp = {cid: e for cid, _, e in cases}
    lines = ["", "=" * 96, "BOLD-VARIANT BENCHMARK", "=" * 96,
             f"runs={runs}  detail={detail}  grid={grid[0]}x{grid[1]}  "
             f"A={models['a']}  B/C={models['rich']}",
             "criteria: 1) 01_compliant PASS  2) 03_notbold FAIL  3) baselines false-fail less  "
             "4) NO new false-passes", ""]

    def rate(v, cid, want):
        rs = records[v].get(cid, {}).get("runs", [])
        n = sum(1 for r in rs if r["status"] == want)
        return n, len(rs)

    hdr = f"{'variant':9s} {'01 PASS':9s} {'03 FAIL':9s} {'02 FAIL':9s} {'04 FAIL':9s} {'base FAILs':11s} {'false-pass':11s} {'avg_s':>6s}"
    lines += [hdr, "-" * len(hdr)]
    summary = {}
    for v in variants:
        c1 = rate(v, "01_compliant", PASS)
        c3 = rate(v, "03_notbold", FAIL)
        c2 = rate(v, "02_titlecase", FAIL)
        c4 = rate(v, "04_reworded", FAIL)
        # baseline false-fails: any baseline run that FAILed
        base_runs = [r for cid in records[v] if cid.startswith("baseline")
                     for r in records[v][cid]["runs"]]
        base_fail = sum(1 for r in base_runs if r["status"] == FAIL)
        # false-passes: any should-FAIL case that PASSed
        fp = sum(1 for cid in records[v] if exp.get(cid) == FAIL
                 for r in records[v][cid]["runs"] if r["status"] == PASS)
        allr = [r for cid in records[v] for r in records[v][cid]["runs"] if r["seconds"] is not None]
        avg = sum(r["seconds"] for r in allr) / len(allr) if allr else 0
        summary[v] = {"c1": c1, "c2": c2, "c3": c3, "c4": c4,
                      "base_fail": base_fail, "base_n": len(base_runs), "false_pass": fp, "avg_s": round(avg, 1)}
        lines.append(f"{v.upper():9s} {f'{c1[0]}/{c1[1]}':9s} {f'{c3[0]}/{c3[1]}':9s} "
                     f"{f'{c2[0]}/{c2[1]}':9s} {f'{c4[0]}/{c4[1]}':9s} "
                     f"{f'{base_fail}/{len(base_runs)}':11s} {str(fp):11s} {avg:>6.1f}")

    # ---- bold-cause + telemetry detail per variant ----
    lines.append("")
    for v in variants:
        lines.append(f"--- variant {v.upper()} detail ---")
        for cid in records[v]:
            rs = records[v][cid]["runs"]
            verdicts = Counter(_AB.get(r["status"], r["status"]) for r in rs)
            causes = Counter(r["cause"] for r in rs if r["status"] in (FAIL, REVIEW))
            line = f"  {cid:14s} {dict(verdicts)}"
            if causes:
                line += f"  fail/review-cause={dict(causes)}"
            lines.append(line)
            # show the model's raw bold evidence on the bold trap and one baseline
            if cid in ("03_notbold", "baseline_1"):
                for r in rs[:2]:
                    f = r.get("fields", {})
                    ev = {k: f.get(k) for k in ("header_bold", "header_bold_confidence",
                          "header_body_comparable", "body_not_bold", "formatting_quality",
                          "header_italic", "header_underline", "body_bold", "body_italic",
                          "body_underline", "formatting_legibility") if k in f}
                    obs = f.get("stroke_weight_observation")
                    lines.append(f"      {ev}" + (f"  obs={obs!r}" if obs else ""))

    # ---- verdict ----
    lines += ["", "VERDICT (vs criteria; A is the incumbent baseline):"]
    a = summary.get("a")
    for v in variants:
        s = summary[v]
        ok1 = s["c1"][0] == s["c1"][1] and s["c1"][1] > 0
        ok2 = s["c3"][0] == s["c3"][1] and s["c3"][1] > 0
        ok4 = s["false_pass"] == 0
        better3 = (a is not None and v != "a"
                   and s["base_fail"] / max(1, s["base_n"]) < a["base_fail"] / max(1, a["base_n"]))
        tag = "PASSES all 4" if (ok1 and ok2 and ok4) else "FAILS a criterion"
        extra = ""
        if v != "a":
            extra = "  (fewer baseline false-fails than A)" if better3 else "  (no baseline-false-fail improvement vs A)"
        lines.append(f"  {v.upper()}: crit1(01=PASS)={'Y' if ok1 else 'N'} "
                     f"crit2(03=FAIL)={'Y' if ok2 else 'N'} crit4(no false-pass)={'Y' if ok4 else 'N'} "
                     f"-> {tag}{extra}")
    lines.append("")
    lines.append("Promote a richer/crop variant into production ONLY if it PASSES crit1/2/4 AND "
                 "reduces baseline false-fails vs A. Otherwise keep the current single-boolean gate.")

    report = "\n".join(lines)
    print(report)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(OUT_DIR, f"bold_variant_benchmark_{stamp}.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    with open(os.path.join(OUT_DIR, f"bold_variant_benchmark_{stamp}.json"), "w", encoding="utf-8") as fh:
        json.dump({"config": {"runs": runs, "detail": detail, "grid": list(grid), "models": models},
                   "summary": summary, "records": records}, fh, indent=2, ensure_ascii=False)
    print(f"\nWritten to: output/bold_variant_benchmark_{stamp}.txt / .json")


if __name__ == "__main__":
    main()
