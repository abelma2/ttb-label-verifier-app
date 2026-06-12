"""Configuration: regulatory constants and matching thresholds.

Regulatory text and rules are grounded in the TTB Beverage Alcohol Manuals (BAM):
  - Distilled Spirits: TTB P 5110.7  (Vol 2, 04/2007)
  - Wine:              TTB-G-2018-7  (Vol 1, 08/2018)
  - Malt Beverages:    TTB P 5130.3  (Vol 3, 07/2001)
"""
import os

# --- Government Health Warning Statement (27 CFR part 16; ABLA of 1988) -------
# Verbatim text, confirmed identical across all three BAMs (Spirits Ch.1 §15, Wine Ch.1
# §10, Malt Ch.1 §10). The wording check is an exact match, so this string must be exact.
# Format rule (same cites): "GOVERNMENT WARNING" must appear in CAPITAL letters AND bold;
# the remainder may NOT be bold.
GOVERNMENT_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
    "drink alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
    "operate machinery, and may cause health problems."
)

# The literal header that must be in caps + bold (used in reviewer-facing messages).
GOVERNMENT_WARNING_HEADER = "GOVERNMENT WARNING"

# A near-but-not-exact warning transcription (>= this similarity, 0-100) goes to
# needs-review instead of a hard fail; nothing non-exact ever auto-passes.
WARNING_WORDING_REVIEW_FLOOR = 90

# Bold handling policy for the government warning (implemented in
# verification._check_warning; measured rationale in BENCHMARK_NOTES.md, dev-archive).
#   "supplement_gate" -- DEFAULT: judges the merged bold observation (the warning
#       supplement's read when enabled, else the main read): True -> pass, False ->
#       review, null -> review. Confidence ignored; bold can never FAIL a label.
# Legacy single-model modes, kept env-selectable for comparison:
#   "note_null_review"   -- a determinate observation passes with a note; only null -> review.
#   "header_simple_gate" -- True -> pass, False -> review, null -> FAIL (clearer image).
#   "note"               -- bold never gates; observations recorded on the reason.
#   "header_medium_gate" -- header only: True at medium+ conf -> pass, False at high -> FAIL,
#                           else review. body_bold is a note.
#   "medium_pass_gate"   -- two rules (header bold AND body NOT bold, 27 CFR 16.22): both at
#                           medium+ conf -> pass; only a high-confidence violation FAILs.
#   "header_body_gate"   -- same two rules, but PASS requires high confidence on both.
#   "confidence_gate"    -- header only, fail-closed on null/low confidence.
#   "review"             -- always hand an otherwise-valid warning to a human.
#   "trust_model"        -- judge header_bold alone, ignoring confidence. Not recommended.
WARNING_BOLD_POLICY = os.environ.get("WARNING_BOLD_POLICY", "supplement_gate")

# --- Fuzzy-match thresholds (0-100) for text fields (brand, class/type) -------
#   score >= FUZZY_PASS          -> pass
#   FUZZY_REVIEW_FLOOR..PASS     -> needs review
#   below FUZZY_REVIEW_FLOOR     -> fail
# Brand/class are scored with a containment-aware token_set_ratio against the UNION of the
# application's {brand_name, fanciful_name} / {class_type, statement_of_composition} --
# see verification._check_text / _candidates.
FUZZY_PASS = 95
FUZZY_REVIEW_FLOOR = 85

# Name & address and country of origin vary in formatting (line breaks, abbreviations,
# embedding in a phrase like "PRODUCT OF SCOTLAND"), so they use a more forgiving
# subset/partial ratio and a lower review floor.
NAME_ADDRESS_PASS = 90
NAME_ADDRESS_REVIEW_FLOOR = 70

# An otherwise-passing brand/class read that differs from the application by at most this
# many character edits (a likely typo: "JON'S" vs "JOHN'S" scores ~96) is routed to review
# instead of auto-passing. A superset read has a large edit distance and is unaffected.
TEXT_NEAR_MISS_EDIT_DISTANCE = 2

# --- ABV matching tolerance, in percentage points (label vs application) ------
# A *matching* tolerance (does the label agree with the application?), not the regulatory
# label-vs-actual-product tolerance in the BAMs.
ABV_PASS_TOLERANCE = 0.1     # within this -> pass
ABV_REVIEW_TOLERANCE = 0.5   # within this -> needs review; beyond -> fail

# --- ABV label-only regulatory checks (independent of the application) --------
# US proof is by definition 2x ABV (27 CFR 5.65). Tolerance in proof points (allows rounding).
PROOF_ABV_TOLERANCE = 1.0
# TTB prescribes "alcohol __% by volume" / "alc. __% by vol." (27 CFR 4.36 / 5.65 / 7.65);
# the bare abbreviation "ABV" is not a prescribed form. Matched as a whole word.
NONCOMPLIANT_ABV_NOTATIONS = ("abv",)

# --- Net contents: unit-aware volume comparison (label vs application) --------
# Both sides parsing to the same volume within this fractional tolerance -> needs-review
# (same volume, different unit/format -- a human verifies the unit and standard of fill),
# never auto-pass; beyond it -> fail. 2% absorbs cross-unit rounding (a 30 mL miniature
# legally printed "1 FL OZ" is ~1.4% off) while staying well below the gap between adjacent
# standard-of-fill sizes (355 vs 375 mL is ~5.6%). The standard-of-fill table itself is
# NOT enforced.
NET_CONTENTS_VOLUME_TOLERANCE = 0.02

# Low-confidence reads are downgraded from pass to "needs review".
ESCALATE_LOW_CONFIDENCE = True

# --- Extraction / runtime ----------------------------------------------------
# Vision model for the main extraction, chosen by a 5x stability benchmark
# (BENCHMARK_NOTES.md, dev-archive). Do NOT default to gpt-4o/-mini, gpt-4.1-mini, o4-mini,
# or gpt-5.4-nano -- they mishandle the bold gate. Env-overridable for A/B testing.
EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", "gpt-5.4-mini")

# Warning-only second reader, run IN PARALLEL with the main extraction and merged by
# extraction.py: its present/text/caps/bold read becomes THE warning the verifier judges;
# the main model's read is kept alongside as main_* evidence. Cross-family on purpose (a
# gpt-5.4-mini supplement repeated the main model's own misreads); measured at 100%
# warning-verdict accuracy on ground truth vs 70% for the main full-extraction read
# (BENCHMARK_NOTES.md). Non-blocking: any failure falls back to the main read with a note.
# Set to "" to disable entirely (single-model behavior, no second API call).
WARNING_SUPPLEMENT_MODEL = os.environ.get("WARNING_SUPPLEMENT_MODEL", "gpt-4.1")

# Hard ceiling on a single extraction call so a hung request fails fast instead of
# blocking the UI.
REQUEST_TIMEOUT_SECONDS = 30

# Batch upload: max concurrent extraction calls.
BATCH_MAX_WORKERS = 8

# A batch bursts up to BATCH_MAX_WORKERS concurrent calls, which can trip HTTP 429 on lower
# account tiers. 429s return immediately, so a short backoff retry (in
# extraction._create_with_fallbacks) does not undermine the REQUEST_TIMEOUT_SECONDS ceiling
# the way SDK-level retries would (the client keeps max_retries=0).
RATE_LIMIT_MAX_RETRIES = int(os.environ.get("RATE_LIMIT_MAX_RETRIES", "2"))
