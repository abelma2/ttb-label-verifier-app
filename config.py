"""Configuration: regulatory constants and matching thresholds.

Regulatory text and rules are grounded in the TTB Beverage Alcohol Manuals (BAM):
  - Distilled Spirits: TTB P 5110.7  (Vol 2, 04/2007)
  - Wine:              TTB-G-2018-7  (Vol 1, 08/2018)
  - Malt Beverages:    TTB P 5130.3  (Vol 3, 07/2001)

Tune the matching thresholds against a small labeled test set before trusting them.
"""
import os

# --- Government Health Warning Statement (27 CFR part 16; ABLA of 1988) -------
# Verbatim text, confirmed IDENTICAL across all three BAMs:
#   Spirits BAM Ch.1 §15 (p.1-17), Wine BAM Ch.1 §10 (p.1-14), Malt BAM Ch.1 §10 (p.1-11).
# The wording check is an exact (case-insensitive) match, so this string must be exact.
# Format rule (same cites): the words "GOVERNMENT WARNING" must appear in CAPITAL
# letters AND bold; the remainder may NOT be bold; it must be one continuous paragraph.
GOVERNMENT_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
    "drink alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
    "operate machinery, and may cause health problems."
)

# The literal header that must be in caps + bold (used in reviewer-facing messages).
GOVERNMENT_WARNING_HEADER = "GOVERNMENT WARNING"

# The warning must match word-for-word. Because the vision model can misread small print,
# a near-but-not-exact transcription (>= this similarity, 0-100) goes to "needs review" so
# a human verifies against the label, instead of a hard fail. Nothing non-exact ever
# auto-passes; only a large deviation fails outright.
WARNING_WORDING_REVIEW_FLOOR = 90

# Bold handling policy for the government warning. Modes:
#   "header_medium_gate" -- DEFAULT (since 2026-06-11, product decision): only the HEADER bold
#                    rule gates the verdict. PASS when header_bold True at MEDIUM-or-high
#                    confidence (on top of wording + ALL-CAPS); FAIL only on a HIGH-confidence
#                    header_bold False; anything else (null / low, or a medium-confidence
#                    violation) -> needs-review. body_bold is TRACKED -- a med/high-confidence
#                    bold-body observation is appended to the reason as a note (and always stays
#                    in the raw extraction) -- but it never decides pass/review/fail. Known cost:
#                    an all-bold-body label with a bold header now PASSES; the two-rule gates
#                    below were added precisely because the old header-only gate auto-passed ~93%
#                    of all-bold-body violations, and accepting that gap again is the explicit
#                    product call here (the note keeps the observation visible to reviewers).
#   "medium_pass_gate" -- the prior default (2026-06-11, per course-staff guidance that the bold
#                    PASS gate need not demand high confidence). 27 CFR 16.22 has TWO visual rules:
#                    "GOVERNMENT WARNING" must be bold, AND the remainder/body may NOT be bold.
#                    PASS when header_bold True AND body_bold False, each at MEDIUM-or-high
#                    confidence (on top of wording + ALL-CAPS). FAIL stays strict -- only a
#                    HIGH-confidence violation of either rule fails (header_bold False+high, or
#                    body_bold True+high); everything else (null, low, or a medium-confidence
#                    violation) -> needs-review. The only relaxation vs header_body_gate is that
#                    medium-confidence, both-rules-satisfied reads move from REVIEW to PASS (fewer
#                    false reviews on clean labels); because FAIL is unchanged, it cannot auto-pass
#                    any high-confidence violation header_body_gate catches. Known cost (see
#                    BENCHMARK_NOTES.md): a medium-confidence MISREAD of a not-bold header as bold
#                    now passes instead of going to review.
#   "header_body_gate" -- the STRICTER prior default (kept selectable via the env var). Same two
#                    rules, but PASS requires HIGH confidence on both fields; anything uncertain
#                    (null / medium / low on either) goes to needs-REVIEW. header_bold True by
#                    itself can never pass -- the body/remainder is checked too. The benchmark
#                    series (BENCHMARK_NOTES.md) showed the old header-only gate auto-passed
#                    ~93% of all-bold-body violations; both two-rule gates close that gap.
#   "confidence_gate" -- older default. Header only: header_bold True + medium/high -> pass; False +
#                    medium/high -> fail; null/low -> fail-closed. Does NOT check the body (all-bold
#                    warnings auto-pass). Kept for comparison/benchmarking.
#   "note"        -- bold is telemetry only; an otherwise-valid warning PASSES with a bold note.
#   "review"      -- an otherwise-valid warning always goes to needs-review for a human.
#   "trust_model" -- judge from header_bold alone (True->pass, False->FAIL, None->review),
#                    ignoring confidence. Not recommended.
# See BENCHMARK_NOTES.md (dev-archive branch; kept locally for dev) for the bold
# experiments behind these choices.
WARNING_BOLD_POLICY = os.environ.get("WARNING_BOLD_POLICY", "header_medium_gate")

# --- Fuzzy-match thresholds (0-100) for text fields (brand, class/type) -------
#   score >= FUZZY_PASS          -> pass
#   FUZZY_REVIEW_FLOOR..PASS     -> needs review
#   below FUZZY_REVIEW_FLOOR     -> fail
# brand/class are scored with a containment-aware token_set_ratio against the UNION of the
# application's {brand_name, fanciful_name} and {class_type, statement_of_composition}
# (see verification._check_text / _candidates), so a more-verbose label read, or the model
# tagging the fanciful name as the brand, still matches the legitimate value. A genuine
# mismatch still scores well below the floor (the union does not mask wrong reads).
FUZZY_PASS = 95
FUZZY_REVIEW_FLOOR = 85

# Name & address and country of origin vary in formatting (line breaks, abbreviations,
# embedding in a phrase like "PRODUCT OF SCOTLAND"), so they use a more forgiving
# subset/partial ratio and a lower review floor.
NAME_ADDRESS_PASS = 90
NAME_ADDRESS_REVIEW_FLOOR = 70

# A high fuzzy score can still hide a 1-2 character difference in a SHORT identity field
# (e.g. brand "JON'S" vs "JOHN'S" scores ~96 -> would auto-pass). When an otherwise-passing
# brand/class read differs from the application by at most this many character edits (after
# normalization, so case/whitespace/apostrophe differences are already 0), route it to review
# instead of passing. A superset read (e.g. "Captain John's Spiced Rum" vs "Captain John's")
# has a large edit distance and is NOT caught by this guard.
TEXT_NEAR_MISS_EDIT_DISTANCE = 2

# --- ABV matching tolerance, in percentage points (label vs application) ------
# NOTE: this is a *matching* tolerance (does the label agree with the application?),
# not the regulatory label-vs-actual-product tolerance in the BAMs.
ABV_PASS_TOLERANCE = 0.1     # within this -> pass
ABV_REVIEW_TOLERANCE = 0.5   # within this -> needs review; beyond -> fail

# --- ABV label-only regulatory checks (independent of the application) --------
# US proof is by definition 2x ABV (27 CFR 5.65), so a proof that disagrees with the stated
# ABV is an internally inconsistent label. Tolerance in proof points (allows rounding).
PROOF_ABV_TOLERANCE = 1.0
# TTB prescribes the alcohol statement as "alcohol __% by volume" / "alc. __% by vol."
# (27 CFR 4.36 / 5.65 / 7.65); the bare abbreviation "ABV" is not a prescribed form. Matched
# as a whole word against the transcribed alcohol-content value, case-insensitively.
NONCOMPLIANT_ABV_NOTATIONS = ("abv",)

# --- Net contents: unit-aware volume comparison (label vs application) --------
# The net-contents check compares VOLUME, not just the printed string: "16.9 FL. OZ." and
# "1 PINT 0.9 FL. OZ." are the same volume in a different unit/format. When both sides parse to a
# volume within this FRACTIONAL tolerance of each other they are treated as the SAME volume and
# routed to needs-review (the unit/format differs from the application — a human verifies the unit
# and standard of fill), NOT auto-passed; a larger difference is a genuine mismatch -> fail. The
# tolerance (2%) absorbs cross-unit rounding (e.g. a 30 mL miniature legally printed as "1 FL OZ",
# 29.57 mL, ~1.4% off) while staying well below the gap between adjacent standard-of-fill sizes (the
# closest, 355 mL vs 375 mL, is ~5.6% apart). Unparseable values fall back to the fuzzy string
# compare. This is a *matching* tolerance, not the standard-of-fill table (which is NOT enforced).
NET_CONTENTS_VOLUME_TOLERANCE = 0.02

# Low-confidence reads from the vision model are downgraded from pass to "needs review"
# so a human double-checks them rather than the app trusting a shaky read.
ESCALATE_LOW_CONFIDENCE = True

# --- Extraction / runtime ----------------------------------------------------
# Vision model used for extraction. gpt-5.4-mini is the default, chosen by a 5x stability pass
# (scripts/benchmarks/stability_benchmark.py; BENCHMARK_NOTES.md -- dev-archive branch). Under WARNING_BOLD_POLICY
# "confidence_gate" it caught the NOT-BOLD adversarial (03_notbold) 5/5, passed compliant (01)
# 5/5, failed title-case/reworded 5/5, AND passed the realistic baselines 14/15 -- whereas
# gpt-4.1 (the prior default) caught 03_notbold but FALSE-FAILED every realistic baseline
# (0/15: it reads their bold headers as not-bold). gpt-5.4-mini is also the fastest accurate
# model (~4.2s/label). gpt-5.5 is the accuracy ceiling (same behavior, ~40% slower). Do NOT use
# gpt-4o/-mini, gpt-4.1-mini, o4-mini, or gpt-5.4-nano as the default -- they mishandle the bold
# gate. Override at runtime with the EXTRACTION_MODEL env var (handy for A/B testing).
EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", "gpt-5.4-mini")

# Hard ceiling on a single extraction call so a hung request fails fast instead of
# blocking the UI (the per-label target is ~5s; this is a safety bound, not the target).
REQUEST_TIMEOUT_SECONDS = 30

# Batch upload: max concurrent extraction calls. Each label is an independent,
# I/O-bound API call, so a small thread pool keeps total time bounded by the API
# rate limit rather than the file count.
BATCH_MAX_WORKERS = 8

# A batch bursts up to BATCH_MAX_WORKERS concurrent calls, which can trip the API rate
# limit (HTTP 429) on lower account tiers. 429s come back immediately -- nothing hangs --
# so a short in-code backoff retry does not undermine the REQUEST_TIMEOUT_SECONDS ceiling
# the way SDK-level retries would (those re-issue timed-out requests; the client keeps
# max_retries=0). Retried in extraction._create_with_fallbacks.
RATE_LIMIT_MAX_RETRIES = int(os.environ.get("RATE_LIMIT_MAX_RETRIES", "2"))
