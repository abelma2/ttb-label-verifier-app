"""Deterministic verification: compare the extracted label fields against the
application data and the fixed regulatory rules. The model reads; this code judges.

Two kinds of check:
  - consistency checks -> label field vs the value the applicant submitted
  - rules checks       -> label field vs fixed regulation (e.g. the government warning)

Regulatory behaviour is grounded in the TTB Beverage Alcohol Manuals (see config.py):
  - Government warning: exact wording + "GOVERNMENT WARNING" in CAPS and BOLD, judged
    fail-closed (an unknown caps/bold observation goes to needs-review, never an auto-pass).
  - Alcohol content is REQUIRED for spirits, CONDITIONAL for wine (<=14% "table"/"light"
    wine may omit it), and OPTIONAL for malt beverages.

Each check consumes the extractor's field objects
``{"present": bool, "value": str|null, "confidence": ...}`` and returns a FieldResult.
"""
import re
from dataclasses import dataclass

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from config import (
    GOVERNMENT_WARNING,
    GOVERNMENT_WARNING_HEADER,
    WARNING_WORDING_REVIEW_FLOOR,
    WARNING_BOLD_POLICY,
    FUZZY_PASS,
    FUZZY_REVIEW_FLOOR,
    NAME_ADDRESS_PASS,
    NAME_ADDRESS_REVIEW_FLOOR,
    TEXT_NEAR_MISS_EDIT_DISTANCE,
    ABV_PASS_TOLERANCE,
    ABV_REVIEW_TOLERANCE,
    PROOF_ABV_TOLERANCE,
    NONCOMPLIANT_ABV_NOTATIONS,
    NET_CONTENTS_VOLUME_TOLERANCE,
    ESCALATE_LOW_CONFIDENCE,
)

PASS, REVIEW, FAIL = "pass", "needs_review", "fail"
_SEVERITY = {PASS: 0, REVIEW: 1, FAIL: 2}


@dataclass
class FieldResult:
    field: str
    extracted: str
    expected: str
    status: str
    reason: str
    # Machine-readable cause of the verdict, set where downstream logic must branch on WHY
    # (currently the government-warning checks: "absence" / "wording" / "caps" / "bold" /
    # "low_confidence"). It is for programmatic branching, never on the user-facing
    # reason string — reasons are display text and may be reworded freely.
    cause: str | None = None


# --- helpers -----------------------------------------------------------------

def _normalize(s) -> str:
    """Casefold, standardize apostrophes, and collapse whitespace."""
    if not s:
        return ""
    s = str(s).casefold().strip()
    s = s.replace("’", "'").replace("`", "'")
    return re.sub(r"\s+", " ", s)


def _get(field_obj):
    """Pull (present, value, confidence) from a normalized field object."""
    if not isinstance(field_obj, dict):
        return False, None, "low"
    return (bool(field_obj.get("present")), field_obj.get("value"),
            field_obj.get("confidence", "low"))


def _escalate(result: FieldResult, confidence) -> FieldResult:
    """Downgrade a PASS to needs-review when the read was low-confidence. The downgraded
    result's cause is "low_confidence" (the checks all SUCCEEDED — the doubt is the read's
    overall confidence), so cause-driven logic never mistakes it for a substantive finding."""
    if ESCALATE_LOW_CONFIDENCE and result.status == PASS and confidence == "low":
        return FieldResult(result.field, result.extracted, result.expected, REVIEW,
                           result.reason + " (low-confidence read — please verify)",
                           cause="low_confidence")
    return result


def _parse_abv(s):
    """Parse an alcohol figure to ABV percent.

    Percent-anchored numbers win (so '45% Alc./Vol. (90 Proof)' -> 45); a proof-only
    figure is converted (US proof / 2 -> ABV, so '90 Proof' -> 45); a value that is
    essentially just a number is taken as ABV (so a user typing '45' works). Anything
    else (e.g. 'Bottled in 2021') returns None rather than grabbing a stray number."""
    if s is None or s == "" or isinstance(s, bool):
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s)
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*proof", s, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 2.0
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", s)
    if m:
        return float(m.group(1))
    return None


_ABV_RANGE_RE = re.compile(r"\d+(?:\.\d+)?\s*%?\s*(?:-|–|to)\s*\d+(?:\.\d+)?\s*%", re.IGNORECASE)


def _is_abv_range(s) -> bool:
    """True if the alcohol figure is a range (e.g. '5-6%', '5% to 6%'); routed to review
    rather than silently committing to one endpoint."""
    return bool(s) and bool(_ABV_RANGE_RE.search(str(s)))


def _fuzzy_result(field_name, value, expected, score, pass_floor, review_floor) -> FieldResult:
    if score >= pass_floor:
        return FieldResult(field_name, value, expected, PASS, "matches the application")
    if score >= review_floor:
        return FieldResult(field_name, value, expected, REVIEW,
                           f"close to the application value but not an exact match "
                           f"({score:.0f}% similar) — please verify")
    return FieldResult(field_name, value, expected, FAIL,
                       f"does not match the application value ({score:.0f}% similar)")


def _missing_value(field_name, present, expected) -> FieldResult:
    """Verdict for a matched field with no readable value: needs-review if the model says
    it IS on the label (present-but-unreadable), fail only if the model reports it absent.
    Keeps 'couldn't read it' distinct from 'not on the label'. Uses `present`, NOT
    confidence: the model returns confidence='low' for absent fields too, so confidence
    can't distinguish absent from unreadable."""
    label = field_name.replace("_", " ")
    if present:
        return FieldResult(field_name, "", expected or "", REVIEW,
                           f"{label} appears on the label but could not be read — please verify")
    return FieldResult(field_name, "", expected or "", FAIL,
                       f"required {label} not found on the label")


# --- consistency checks (label vs application) -------------------------------

def _check_text(field_name, field_obj, expected, *, pass_floor=FUZZY_PASS,
                review_floor=FUZZY_REVIEW_FLOOR, scorer=fuzz.token_sort_ratio,
                near_miss_review=False) -> FieldResult:
    """Fuzzy match for free-text fields (brand name, class/type).

    `expected` may be a single application value OR a list of acceptable values (e.g.
    the brand_name *or* the fanciful_name): the label read is scored against each and
    the best match wins, so whichever legitimate name the model happened to transcribe
    still matches. The extractor never sees these values — it stays blind; the union is
    purely a verification-side decision.

    `near_miss_review` adds an edit-distance guard for short identity fields: a fuzzy PASS
    that still differs from the matched value by 1-2 characters (a likely typo, e.g.
    "JON'S" vs "JOHN'S") is routed to review rather than auto-passing. A superset read has a
    large edit distance and is unaffected; an exact (normalized) match has distance 0."""
    present, value, conf = _get(field_obj)
    candidates = [c for c in (expected if isinstance(expected, (list, tuple)) else [expected]) if c]
    if not candidates:
        if value:
            return FieldResult(field_name, value, "", REVIEW,
                               "present on the label but no application value to compare")
        return FieldResult(field_name, value or "", "", REVIEW, "no application value provided")
    if not value:
        return _missing_value(field_name, present, candidates[0])
    best_expected, best_score = max(
        ((c, scorer(_normalize(value), _normalize(c))) for c in candidates),
        key=lambda cs: cs[1])
    result = _fuzzy_result(field_name, value, best_expected, best_score, pass_floor, review_floor)
    if near_miss_review and result.status == PASS:
        edits = Levenshtein.distance(_normalize(value), _normalize(best_expected))
        if 0 < edits <= TEXT_NEAR_MISS_EDIT_DISTANCE:
            result = FieldResult(field_name, value, best_expected, REVIEW,
                                 f"near-exact match but differs from the application by "
                                 f"{edits} character(s) (possible typo) — please verify")
    return _escalate(result, conf)


# --- net contents: unit-aware volume parsing --------------------------------
# Physical unit -> millilitre conversions (module-local: fixed physical constants, not a tunable
# knob; the tolerance that uses them lives in config.NET_CONTENTS_VOLUME_TOLERANCE). Each entry is
# (regex, mL-per-unit). Ordered most-specific first ("ml"/"cl"/"dl" before bare "l", "fl oz" before
# "oz"); each regex requires the number to immediately precede the unit, so two units never claim
# the same number.
_VOLUME_UNITS_ML = (
    (r"milliliters?|millilitres?|mls?", 1.0),
    (r"centiliters?|centilitres?|cl", 10.0),
    (r"deciliters?|decilitres?|dl", 100.0),
    (r"liters?|litres?|l", 1000.0),
    (r"fluid\s*ounces?|fl\.?\s*oz\.?|floz", 29.5735),
    (r"ounces?|oz\.?", 29.5735),   # net contents on a beverage: oz = US fluid ounce
    (r"pints?|pt\.?", 473.176),
    (r"quarts?|qt\.?", 946.353),
    (r"gallons?|gal\.?", 3785.41),
)
_VOLUME_PART_RES = tuple(
    (re.compile(rf"(\d+(?:\.\d+)?)\s*(?:{pat})(?![a-z])", re.IGNORECASE), factor)
    for pat, factor in _VOLUME_UNITS_ML)


def _parse_volume(s):
    """Parse a net-contents string to a volume in millilitres, or None if no recognized unit is
    present. Sums a genuine COMPOUND quantity ("1 PINT 0.9 FL. OZ." -> 1 pint + 0.9 fl oz), but
    treats a parenthetical or repeated RESTATEMENT of the same volume as ONE value, not a sum:
      - a parenthetical is dropped before parsing ("16.9 FL OZ (500 mL)" -> 16.9 fl oz);
      - identical (unit, value) pairs are de-duped ("750 mL 750 mL" -> 750), while a real compound
        uses DIFFERENT units and still sums.
    Commas are treated as thousands separators ("1,750 mL" -> 1750). A bare number or unknown unit
    returns None so the caller falls back to the fuzzy string compare rather than guess."""
    if not s:
        return None
    text = re.sub(r"(?<=\d),(?=\d)", "", str(s).lower())     # join thousands separators
    primary = re.sub(r"\([^)]*\)", " ", text)                # a parenthetical restates, it does not add
    if not any(rx.search(primary) for rx, _ in _VOLUME_PART_RES):
        primary = text                                       # the only declaration was inside the parens
    quantities = {(factor, float(m.group(1)))
                  for rx, factor in _VOLUME_PART_RES for m in rx.finditer(primary)}
    if not quantities:
        return None
    return sum(factor * value for factor, value in quantities)


def _fmt_ml(ml):
    return f"{ml:.0f} mL" if ml >= 10 else f"{ml:.1f} mL"


def _check_net_contents(field_obj, expected) -> FieldResult:
    """Compare net contents by VOLUME, not just the printed string. After the exact/whitespace
    match, parse both sides to millilitres: the SAME volume in a different unit/format (e.g.
    "16.9 FL. OZ." vs "1 PINT 0.9 FL. OZ.") goes to needs-review (verify the unit / standard of
    fill), NOT auto-pass; a materially different volume fails; an unparseable value falls back to
    the fuzzy string compare. The standard-of-fill table (permitted sizes) is NOT enforced here."""
    present, value, conf = _get(field_obj)
    if not expected:
        if value:
            return _escalate(FieldResult("net_contents", value, "", REVIEW,
                             "present on the label but no application value to compare"), conf)
        return FieldResult("net_contents", value or "", "", REVIEW, "no application value provided")
    if not value:
        return _missing_value("net_contents", present, expected)
    strip = lambda s: re.sub(r"\s+", "", _normalize(s))
    if strip(value) == strip(expected):
        return _escalate(FieldResult("net_contents", value, expected, PASS,
                                     "matches the application"), conf)
    v_ml, e_ml = _parse_volume(value), _parse_volume(expected)
    if v_ml is not None and e_ml is not None:   # 0.0 is a valid parsed volume, not "unparseable"
        if abs(v_ml - e_ml) <= NET_CONTENTS_VOLUME_TOLERANCE * max(v_ml, e_ml):
            return FieldResult("net_contents", value, expected, REVIEW,
                               "same net contents volume as the application but a different "
                               "unit/format — please verify the unit and standard of fill")
        return FieldResult("net_contents", value, expected, FAIL,
                           f"net contents differ from the application "
                           f"({_fmt_ml(v_ml)} vs {_fmt_ml(e_ml)})")
    return _check_text("net_contents", field_obj, expected)


# A name/address read that is a strict subset of the expected value scores ~100 under
# token_set_ratio (containment), so an extraction that captured only the city/state and
# DROPPED the producer/bottler/importer name would otherwise PASS. Require the read to cover
# at least this fraction of the expected's significant tokens; a high-scoring but low-coverage
# subset read is routed to needs-review instead of auto-passing. (Module-local, not config.py:
# this is a guard mechanic, not a regulatory knob — promote it to config on request.)
_NAME_ADDRESS_COVERAGE_FLOOR = 0.6
# Connectives PLUS the standard bottler/producer RELATIONSHIP-phrase words ("Brewed & Bottled By",
# "Distilled & Bottled By", "Produced and Bottled By", "Imported By"). These are NOT part of the
# producer name/address, so they are treated as non-significant for the coverage + producer-token
# checks — making the match invariant to whether the relationship prefix is printed. (The
# relationship TYPE — bottled vs imported — is a separate compliance check, not done here.)
_NAME_ADDRESS_STOPWORDS = frozenset({
    "by", "and", "the", "of", "for",
    "brewed", "bottled", "distilled", "produced", "manufactured",
    "packed", "blended", "vinted", "cellared", "imported",
})


def _significant_tokens(s) -> set:
    """Alphanumeric tokens of a name/address, minus ubiquitous connectives + relationship-phrase
    words, for coverage and the producer-token check."""
    return {t for t in re.findall(r"[a-z0-9]+", _normalize(s))
            if len(t) > 1 and t not in _NAME_ADDRESS_STOPWORDS}


def _normalize_address(s) -> str:
    """Name/address-specific normalization used ONLY for the match score: on top of _normalize
    (casefold + apostrophes + whitespace), map '&' -> 'and' and drop punctuation (commas/colons/
    periods) that otherwise sticks to tokens and makes token_set_ratio brittle ('Distillery,' !=
    'Distillery'). The original value/expected strings are still what gets displayed."""
    s = _normalize(s).replace("&", " and ")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def _is_short_subset_address(value, expected) -> bool:
    """True when the read is a STRICT subset of the expected name/address that covers too
    little of it (e.g. only the city/state) — the producer/bottler/importer name is missing.
    A near-complete read (drops only a connective like 'Bottled by') stays above the floor."""
    ev, ee = _significant_tokens(value), _significant_tokens(expected)
    if not ev or not ee or not ev < ee:          # only a strict subset can be a partial read
        return False
    return len(ev & ee) / len(ee) < _NAME_ADDRESS_COVERAGE_FLOOR


def _check_name_address(field_obj, expected) -> FieldResult:
    """Name & address vary in order/abbreviation/punctuation, so score with a forgiving subset
    ratio over a punctuation-normalized form (_normalize_address: '&'->'and', commas/periods
    dropped) so a comma-placement or relationship-prefix difference doesn't false-review an
    otherwise-identical address. The coverage guard (_is_short_subset_address) keeps that from
    being TOO forgiving: a short subset read that DROPPED a chunk (e.g. only the city/state — the
    producer/bottler name is missing) is routed to needs-review, never an auto-pass.

    KNOWN GAP: a producer-name word SUBSTITUTION on a long address (e.g. 'Vintners' misread as
    'Wineries') can still pass the fuzzy score. A token-level substitution guard was tried and
    removed because it false-reviewed common formatting differences (full state name vs postal
    abbreviation, an application ZIP the label omits, a dropped apostrophe) — see the xfail tests.
    A future fix should flag only a genuine swap (a distinctive token present on both sides but
    different), not any missing token, and tokenize over the same _normalize_address the score uses."""
    present, value, conf = _get(field_obj)
    if not expected:
        if value:
            return _escalate(FieldResult("name_and_address", value, "", REVIEW,
                             "present on the label but no application value to compare"), conf)
        return FieldResult("name_and_address", value or "", "", REVIEW, "no application value provided")
    if not value:
        return _missing_value("name_and_address", present, expected)
    score = fuzz.token_set_ratio(_normalize_address(value), _normalize_address(expected))
    result = _fuzzy_result("name_and_address", value, expected, score,
                           NAME_ADDRESS_PASS, NAME_ADDRESS_REVIEW_FLOOR)
    if result.status == PASS and _is_short_subset_address(value, expected):
        result = FieldResult("name_and_address", value, expected, REVIEW,
                             "label read appears to omit part of the required name/address "
                             "(e.g. the producer/bottler/importer name) — please verify")
    return _escalate(result, conf)


def _check_country(field_obj, expected) -> FieldResult:
    """Country of origin is only required for imports. If the application supplies a
    country (i.e. the product is imported) it must appear on the label and match;
    otherwise an absent statement is fine (domestic product)."""
    present, value, conf = _get(field_obj)
    if not expected:
        if value:
            return _escalate(FieldResult("country_of_origin", value, "", REVIEW,
                             "label shows a country of origin but the application lists none — please verify"), conf)
        return FieldResult("country_of_origin", "", "", PASS, "not applicable (domestic product)")
    if not value:
        if present:
            return FieldResult("country_of_origin", "", expected, REVIEW,
                               "country of origin appears present but could not be read — please verify")
        return FieldResult("country_of_origin", "", expected, FAIL,
                           "imported product must show country of origin; none found on the label")
    # the country name may be embedded in a phrase like "PRODUCT OF SCOTLAND"
    score = fuzz.partial_ratio(_normalize(value), _normalize(expected))
    return _escalate(_fuzzy_result("country_of_origin", value, expected, score,
                                   NAME_ADDRESS_PASS, NAME_ADDRESS_REVIEW_FLOOR), conf)


# --- alcohol content (class-dependent) ---------------------------------------

def _is_low_alcohol_table_wine(class_value) -> bool:
    """Per Wine BAM Ch.1 §3 (p.1-3): wine <=14% may omit the ABV statement if the
    class designation is 'table wine' or 'light wine' (both are by definition <=14%)."""
    n = _normalize(class_value)
    return "table wine" in n or "light wine" in n


def _abv_missing_by_class(beverage_type, class_obj) -> FieldResult:
    """Verdict for a label that shows NO readable ABV, by beverage class."""
    if beverage_type == "spirits":
        return FieldResult("alcohol_content", "", "", FAIL,
                           "distilled spirits must state alcohol content; none found on the label")
    if beverage_type == "wine":
        _, class_value, _ = _get(class_obj)
        if _is_low_alcohol_table_wine(class_value):
            return FieldResult("alcohol_content", "", "", PASS,
                               "alcohol content may be omitted for ≤14% table/light wine")
        return FieldResult("alcohol_content", "", "", REVIEW,
                           "no alcohol content found (required for wine unless ≤14% table/light wine)")
    if beverage_type == "beer":
        return FieldResult("alcohol_content", "", "", PASS,
                           "alcohol content not stated (optional for malt beverages)")
    return FieldResult("alcohol_content", "", "", REVIEW,
                       "no alcohol content found (beverage type unknown)")


def _label_abv(field_obj):
    """Best-effort ABV number from a field object: prefer the parsed abv_percent, then
    a percent/proof-aware parse of the verbatim value, then the structured proof / 2."""
    if not isinstance(field_obj, dict):
        return None
    if field_obj.get("abv_percent") is not None:
        return float(field_obj["abv_percent"])
    parsed = _parse_abv(field_obj.get("value"))
    if parsed is not None:
        return parsed
    if field_obj.get("proof") is not None:
        return float(field_obj["proof"]) / 2.0
    return None


def _check_abv_notation(value, expected):
    """Flag a non-compliant alcohol-content notation (e.g. the bare abbreviation 'ABV', which
    TTB does not prescribe — 27 CFR 4.36 / 5.65 / 7.65). Returns a FAIL FieldResult, or None
    when the notation is acceptable / there is no value to inspect."""
    n = _normalize(value)
    if not n:
        return None
    for bad in NONCOMPLIANT_ABV_NOTATIONS:
        if re.search(rf"\b{re.escape(bad)}\b", n):
            return FieldResult("alcohol_content", value, expected or "", FAIL,
                               f"non-compliant alcohol-content notation '{bad.upper()}'; TTB "
                               "requires 'alcohol __% by volume' / 'alc. __% by vol.'")
    return None


def _check_proof_consistency(field_obj, expected):
    """Flag a proof that disagrees with the stated ABV (US proof is by definition 2x ABV;
    27 CFR 5.65). Returns a FAIL FieldResult, or None when consistent or when proof and ABV
    are not both present on the label."""
    if not isinstance(field_obj, dict):
        return None
    abv, proof = field_obj.get("abv_percent"), field_obj.get("proof")
    if abv is None or proof is None:
        return None
    abv, proof = float(abv), float(proof)
    if abs(proof - 2.0 * abv) > PROOF_ABV_TOLERANCE:
        return FieldResult("alcohol_content", field_obj.get("value") or f"{abv:g}%", expected or "",
                           FAIL, f"label is internally inconsistent: {proof:g} proof is not "
                           f"2 × {abv:g}% ABV ({2.0 * abv:g} proof expected)")
    return None


def _check_abv(field_obj, expected, beverage_type, class_obj) -> FieldResult:
    """Numeric comparison for alcohol content, with the class-specific presence rule, plus two
    label-only regulatory checks (independent of the application): the notation must be a TTB
    form (no bare 'ABV') and the proof must equal 2x the ABV."""
    _, value, conf = _get(field_obj)
    if _is_abv_range(value) or _is_abv_range(expected):
        return _escalate(FieldResult("alcohol_content", value or "", expected or "", REVIEW,
                         "alcohol content is stated as a range — please verify the comparison"), conf)
    bad_notation = _check_abv_notation(value, expected)
    if bad_notation is not None:
        return bad_notation
    inconsistent_proof = _check_proof_consistency(field_obj, expected)
    if inconsistent_proof is not None:
        return inconsistent_proof
    label_abv = _label_abv(field_obj)
    app_abv = _parse_abv(expected)

    if label_abv is None:
        base = _abv_missing_by_class(beverage_type, class_obj)
        if app_abv is not None and base.status == PASS:
            # class doesn't require it, but the application lists one -> worth a look
            return FieldResult("alcohol_content", "", expected, REVIEW,
                               "no alcohol content on the label, but the application lists one — please verify")
        return FieldResult("alcohol_content", base.extracted, expected or base.expected,
                           base.status, base.reason)

    shown = value or f"{label_abv:g}%"
    if app_abv is None:
        return _escalate(FieldResult("alcohol_content", shown, expected or "", PASS,
                         f"alcohol content present ({label_abv:g}% ABV)"), conf)
    diff = abs(label_abv - app_abv)
    if diff <= ABV_PASS_TOLERANCE:
        r = FieldResult("alcohol_content", shown, expected, PASS, "matches the application")
    elif diff <= ABV_REVIEW_TOLERANCE:
        r = FieldResult("alcohol_content", shown, expected, REVIEW,
                        f"off by {diff:.1f} percentage points")
    else:
        r = FieldResult("alcohol_content", shown, expected, FAIL,
                        f"label says {label_abv:g}% but application says {app_abv:g}%")
    return _escalate(r, conf)


# --- appellation of origin (wine, conditionally mandatory) -------------------
# A wine that names a grape varietal, carries a vintage date, or uses a semi-generic type
# designation (among other triggers) must show an appellation of origin (27 CFR 4.25 / 4.34;
# the TTB Wine checklist). Unlike the composition-triggered disclosures (sulfites, FD&C Yellow
# #5...), these triggers ARE visible on the label, so they are checkable. Varietal detection
# uses a common-varietal set; the full list is 27 CFR 4.91, so an uncertain case is routed to
# review rather than hard-failed.
_COMMON_VARIETALS = {
    "chardonnay", "cabernet sauvignon", "cabernet franc", "merlot", "pinot noir", "pinot grigio",
    "pinot gris", "sauvignon blanc", "riesling", "zinfandel", "syrah", "shiraz", "malbec",
    "tempranillo", "sangiovese", "grenache", "chenin blanc", "viognier", "gewurztraminer",
    "gewürztraminer", "moscato", "muscat", "petite sirah", "barbera", "nebbiolo", "semillon",
    "albarino", "albariño", "verdejo", "petit verdot", "mourvedre", "mourvèdre", "carmenere",
    "carmenère", "gruner veltliner", "grüner veltliner", "torrontes", "torrontés",
}

# Semi-generic type designations (27 CFR 4.24(b)): when one of these is used as the class/type,
# the Wine checklist (27 CFR 4.34) requires an appellation of origin to accompany it — e.g.
# "California Burgundy", not a bare "Burgundy". These are region-derived names, distinct from the
# grape varietals above. Matched on WORD BOUNDARIES (not the varietals' plain substring test) so
# "port" does not fire on "Portland"/"porter" and "hock" does not fire on "shock".
_SEMI_GENERIC_DESIGNATIONS = frozenset({
    "angelica", "burgundy", "claret", "chablis", "champagne", "chianti", "malaga", "marsala",
    "madeira", "moselle", "port", "rhine", "hock", "sauterne", "sauternes", "sherry", "tokay",
})


def _wine_requires_appellation(class_value, vintage_present):
    """(required, reason) per the Wine checklist trigger: a vintage date, a grape varietal
    designation, a semi-generic type designation, or an estate-bottled claim makes an
    appellation of origin mandatory (27 CFR 4.25 / 4.34)."""
    n = _normalize(class_value)
    if vintage_present:
        return True, "a vintage date"
    if any(v in n for v in _COMMON_VARIETALS):
        return True, "a grape varietal designation"
    if any(re.search(rf"\b{re.escape(d)}\b", n) for d in _SEMI_GENERIC_DESIGNATIONS):
        return True, "a semi-generic type designation"
    if "estate" in n:
        return True, "an estate-bottled claim"
    return False, ""


# Country/state appellation terms that may be embedded in the class/type designation, e.g.
# "American Moscato" or "California Red Wine" -- in which case the appellation requirement is
# satisfied even if the model didn't break it into the appellation field. The full appellation
# set (counties, AVAs) is larger; this covers the common country/state cases confirmable from text.
_APPELLATION_TERMS = {
    "american", "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana",
    "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire",
    "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio", "oklahoma",
    "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota", "tennessee",
    "texas", "utah", "vermont", "virginia", "washington", "west virginia", "wisconsin", "wyoming",
}


def _appellation_in_text(text):
    """Return an appellation term embedded in `text` (e.g. the class/type designation), or None."""
    n = _normalize(text)
    for term in _APPELLATION_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", n):
            return term.title()
    return None


def _check_appellation(beverage_type, class_obj, vintage_obj, appellation_obj) -> FieldResult:
    """Wine only: when the label names a varietal/vintage/semi-generic type, an appellation of
    origin is mandatory. Trigger detection is imperfect (full varietal list is 27 CFR 4.91), so a
    genuinely-absent appellation FAILs while a present-but-unreadable one goes to review. An
    appellation embedded in the designation (e.g. "American Moscato") satisfies the rule."""
    if beverage_type != "wine":
        return FieldResult("appellation", "", "", PASS, "not applicable (wine requirement)")
    _, class_value, _ = _get(class_obj)
    vintage_present, _, _ = _get(vintage_obj)
    present, value, conf = _get(appellation_obj)
    required, why = _wine_requires_appellation(class_value, bool(vintage_present))
    if value:
        note = f"appellation present (required because the label shows {why})" if required \
            else "appellation present"
        return _escalate(FieldResult("appellation", value, "", PASS, note), conf)
    if not required:
        return FieldResult("appellation", "", "", PASS,
                           "not required (label shows no varietal, vintage, or semi-generic type)")
    # the appellation may be embedded in the class/type designation, e.g. "American Moscato"
    embedded = _appellation_in_text(class_value)
    if embedded:
        return FieldResult("appellation", embedded, "", PASS,
                           f"appellation '{embedded}' appears in the class/type designation "
                           f"(required because the label shows {why})")
    if present:   # present but unreadable
        return FieldResult("appellation", "", "", REVIEW,
                           f"appellation appears present but could not be read — required because "
                           f"the label shows {why}; please verify")
    return FieldResult("appellation", "", "", FAIL,
                       f"appellation of origin required (the label shows {why}) but none was found")


# --- rules check (label vs fixed regulation) ---------------------------------

# The canonical warning body (the statement after the "GOVERNMENT WARNING:" header).
# Labels commonly print the body in all-caps, and the vision model sometimes returns the
# body WITHOUT the header, so wording is matched on the body, case-insensitively.
_WARNING_HEADER_RE = re.compile(r"^\s*government\s+warning\s*:?\s*", re.IGNORECASE)


def _warning_body(text) -> str:
    return _WARNING_HEADER_RE.sub("", text or "", count=1)


_CANONICAL_WARNING_BODY = _warning_body(GOVERNMENT_WARNING)
_CANONICAL_WARNING_BODY_NORM = _normalize(_CANONICAL_WARNING_BODY)


def _check_warning(gw) -> FieldResult:
    """Wording + the caps/bold header rule for the government warning, FAIL-CLOSED.

    Grounded in 27 CFR part 16 / the TTB checklists. Because labels often print the
    warning in all-caps and the model sometimes returns the body without the
    "GOVERNMENT WARNING:" header, we:
      - match WORDING on the body (case-insensitive);
      - judge header CAPS deterministically from the verbatim text when the header is
        present, else fall back to the model's header_all_caps, with caps==False as a
        fail backstop;
      - require the "S" in Surgeon / "G" in General to be capitalized (all-caps is fine);
      - judge BOLD per config.WARNING_BOLD_POLICY (default "header_body_gate": Pass/Review/Fail
        on BOTH 27 CFR 16.22 visual rules -- the header must be bold AND the body/remainder must
        NOT be bold. PASS only when header_bold True + high confidence AND body_bold False + high
        confidence; FAIL on a high-confidence violation of either (header not bold, or body bold);
        anything uncertain (null / medium / low on either field) -> needs-review. header_bold True
        by itself can no longer pass. The inline comment at the bold block documents all policies.)
    Title case fails and an unverifiable bold read goes to review; a near-miss wording read
    goes to needs-review; nothing non-exact ever auto-passes.
    """
    gw = gw if isinstance(gw, dict) else {}
    present = bool(gw.get("present"))
    text = gw.get("text")
    caps = gw.get("header_all_caps")

    if not present or not text:
        return FieldResult("government_warning", text or "", GOVERNMENT_WARNING, FAIL,
                           "no government warning found on the label", cause="absence")

    body_norm = _normalize(_warning_body(text))
    if body_norm != _CANONICAL_WARNING_BODY_NORM:
        # Not exact. The model can misread small print, so a near match goes to review
        # (a human verifies) rather than a hard fail; nothing non-exact auto-passes.
        score = fuzz.ratio(body_norm, _CANONICAL_WARNING_BODY_NORM)
        if score >= WARNING_WORDING_REVIEW_FLOOR:
            return FieldResult("government_warning", text, GOVERNMENT_WARNING, REVIEW,
                               f"warning wording is close but not an exact match "
                               f"({score:.0f}% similar) — the read may be imperfect; "
                               f"please verify the exact text against the label", cause="wording")
        return FieldResult("government_warning", text, GOVERNMENT_WARNING, FAIL,
                           f"warning wording does not match the required text ({score:.0f}% similar)",
                           cause="wording")

    # Header caps: deterministic from the verbatim text when the header is present there (so
    # title case fails), else fall back to the model's header_all_caps observation, with an
    # explicit caps==False as a backstop. NOTE: we tried *requiring* the literal header in
    # `text`, but the model often omits it from the transcription even when it IS present
    # (still reporting header_all_caps=true), which false-reviewed compliant labels -- so we
    # trust the observation as a fallback rather than gate on the transcription quirk.
    m = re.search(r"government\s+warning", text, re.IGNORECASE)
    header_caps = text[m.start():m.end()].isupper() if m else caps
    if header_caps is False or caps is False:
        return FieldResult("government_warning", text, GOVERNMENT_WARNING, FAIL,
                           f"'{GOVERNMENT_WARNING_HEADER}' must be in capital letters", cause="caps")
    if header_caps is None:
        return FieldResult("government_warning", text, GOVERNMENT_WARNING, REVIEW,
                           f"could not confirm '{GOVERNMENT_WARNING_HEADER}' is in capital letters — please verify",
                           cause="caps")

    # "S" in Surgeon and "G" in General must be capitalized (all-caps satisfies this).
    for word in ("surgeon", "general"):
        mw = re.search(word, text, re.IGNORECASE)
        if mw and text[mw.start()].islower():
            return FieldResult("government_warning", text, GOVERNMENT_WARNING, FAIL,
                               "the 'S' in Surgeon and 'G' in General must be capitalized",
                               cause="caps")

    # Bold handling per config.WARNING_BOLD_POLICY (see config.py for the full description).
    #   "header_body_gate" -> DEFAULT. Pass/Review/Fail on BOTH visual rules (27 CFR 16.22): the
    #                    header must be bold AND the remainder/body must NOT be bold. header_bold
    #                    alone can no longer pass. PASS requires HIGH confidence on both fields.
    #   "medium_pass_gate" -> same two rules as header_body_gate, but the PASS gate accepts MEDIUM-or-
    #                    high confidence; FAIL is unchanged (high-confidence violation only). Behind
    #                    the env option, NOT the default -- benchmark before promoting.
    #   "confidence_gate" -> fail-closed using header_bold + header_bold_confidence (header only);
    #   "trust_model"     -> judge from header_bold alone (ignores confidence);
    #   "note"            -> bold is telemetry only, never gates;
    #   "review"          -> hand every otherwise-valid warning to a human.
    # NOTE: the per-policy branches below intentionally repeat the PASS/_escalate + reason pattern.
    # That duplication is kept on purpose -- an explicit, auditable branch per policy is preferred
    # over a DRY dispatch for this regulatory logic. medium_pass_gate (the 6th policy) deliberately
    # mirrors header_body_gate's structure rather than sharing a helper, so the two gates can be
    # diffed line-by-line; a shared PASS/escalate helper remains a deferred future cleanup.
    bold = gw.get("header_bold")
    bold_conf = gw.get("header_bold_confidence", "low")
    if WARNING_BOLD_POLICY == "header_body_gate":
        # 27 CFR 16.22 has TWO visual rules: "GOVERNMENT WARNING" must be bold, and the remainder
        # may NOT be bold. Both must be confidently satisfied to PASS; a high-confidence violation
        # of either FAILS; anything uncertain (null / medium / low on either field) -> REVIEW.
        body_bold = gw.get("body_bold")
        body_conf = gw.get("body_bold_confidence", "low")
        header_not_bold = bold is False and bold_conf == "high"
        body_is_bold = body_bold is True and body_conf == "high"
        header_msg = f"'{GOVERNMENT_WARNING_HEADER}' does not appear to be in bold"
        body_msg = ("the body of the warning appears to be in bold — the remainder of "
                    "the warning may not appear in bold type")
        if header_not_bold and body_is_bold:
            # both visual rules violated at high confidence: report BOTH, so a reviewer doesn't
            # "fix" only the header and resubmit a label whose body is still impermissibly bold.
            return FieldResult("government_warning", text, GOVERNMENT_WARNING, FAIL,
                               f"{header_msg}; and {body_msg}", cause="bold")
        if header_not_bold:
            return FieldResult("government_warning", text, GOVERNMENT_WARNING, FAIL, header_msg,
                               cause="bold")
        if body_is_bold:
            return FieldResult("government_warning", text, GOVERNMENT_WARNING, FAIL, body_msg,
                               cause="bold")
        if bold is True and bold_conf == "high" and body_bold is False and body_conf == "high":
            result = FieldResult("government_warning", text, GOVERNMENT_WARNING, PASS,
                                 "wording, capital letters, bold header, and non-bold body all verified",
                                 cause="bold")
            return _escalate(result, gw.get("confidence"))
        return FieldResult("government_warning", text, GOVERNMENT_WARNING, REVIEW,
                           f"could not confirm the warning's bold formatting with high confidence "
                           f"(need '{GOVERNMENT_WARNING_HEADER}' bold AND the body NOT bold) — "
                           f"please verify", cause="bold")
    if WARNING_BOLD_POLICY == "medium_pass_gate":
        # Same two-rule structure as header_body_gate, but the PASS gate accepts MEDIUM-or-high
        # confidence instead of high-only. PASS when header_bold True AND body_bold False, each at
        # medium/high confidence. FAIL is IDENTICAL to header_body_gate -- only a HIGH-confidence
        # violation of either visual rule fails (header_bold False+high, or body_bold True+high).
        # Everything else (null, low confidence, OR a medium-confidence violation such as
        # header_bold False+medium / body_bold True+medium) -> REVIEW. Because the FAIL conditions
        # are unchanged, no high-confidence violation that header_body_gate fails can pass here; the
        # only behavioral delta is that medium-confidence, both-rules-satisfied reads move REVIEW->PASS.
        body_bold = gw.get("body_bold")
        body_conf = gw.get("body_bold_confidence", "low")
        header_not_bold = bold is False and bold_conf == "high"
        body_is_bold = body_bold is True and body_conf == "high"
        header_msg = f"'{GOVERNMENT_WARNING_HEADER}' does not appear to be in bold"
        body_msg = ("the body of the warning appears to be in bold — the remainder of "
                    "the warning may not appear in bold type")
        if header_not_bold and body_is_bold:
            return FieldResult("government_warning", text, GOVERNMENT_WARNING, FAIL,
                               f"{header_msg}; and {body_msg}", cause="bold")
        if header_not_bold:
            return FieldResult("government_warning", text, GOVERNMENT_WARNING, FAIL, header_msg,
                               cause="bold")
        if body_is_bold:
            return FieldResult("government_warning", text, GOVERNMENT_WARNING, FAIL, body_msg,
                               cause="bold")
        header_bold_ok = bold is True and bold_conf in ("medium", "high")
        body_not_bold_ok = body_bold is False and body_conf in ("medium", "high")
        if header_bold_ok and body_not_bold_ok:
            result = FieldResult("government_warning", text, GOVERNMENT_WARNING, PASS,
                                 "wording, capital letters, bold header, and non-bold body all verified",
                                 cause="bold")
            return _escalate(result, gw.get("confidence"))
        return FieldResult("government_warning", text, GOVERNMENT_WARNING, REVIEW,
                           f"could not confirm the warning's bold formatting with at least medium "
                           f"confidence (need '{GOVERNMENT_WARNING_HEADER}' bold AND the body NOT "
                           f"bold) — please verify", cause="bold")
    if WARNING_BOLD_POLICY == "confidence_gate":
        confident = bold_conf in ("medium", "high")
        if bold is True and confident:
            result = FieldResult("government_warning", text, GOVERNMENT_WARNING, PASS,
                                 "wording, capital letters, and bold header all verified",
                                 cause="bold")
            return _escalate(result, gw.get("confidence"))
        if bold is False and confident:
            return FieldResult("government_warning", text, GOVERNMENT_WARNING, FAIL,
                               f"'{GOVERNMENT_WARNING_HEADER}' does not appear to be in bold",
                               cause="bold")
        # header_bold is null, or the bold read was low-confidence -> cannot verify, fail closed
        return FieldResult("government_warning", text, GOVERNMENT_WARNING, FAIL,
                           f"could not verify that '{GOVERNMENT_WARNING_HEADER}' is visibly bold "
                           f"from this image; submit a clearer label image", cause="bold")
    if WARNING_BOLD_POLICY == "trust_model":
        if bold is False:
            return FieldResult("government_warning", text, GOVERNMENT_WARNING, FAIL,
                               f"'{GOVERNMENT_WARNING_HEADER}' does not appear to be in bold",
                               cause="bold")
        if bold is None:
            return FieldResult("government_warning", text, GOVERNMENT_WARNING, REVIEW,
                               f"could not confirm '{GOVERNMENT_WARNING_HEADER}' is in bold — please verify",
                               cause="bold")
        result = FieldResult("government_warning", text, GOVERNMENT_WARNING, PASS,
                             "wording, capital letters, and bold header all verified", cause="bold")
        return _escalate(result, gw.get("confidence"))
    if WARNING_BOLD_POLICY == "note":
        # Bold is telemetry, not a gate: surface the model's observation but never fail/review
        # on it (benchmarks show bold isn't reliably machine-verifiable -- BENCHMARK_NOTES.md).
        observed = {True: "model observed bold", False: "model observed NOT bold"}.get(
            gw.get("header_bold"), "bold not determinable")
        result = FieldResult("government_warning", text, GOVERNMENT_WARNING, PASS,
                             f"wording and capital letters verified; header bold is a "
                             f"model-observed note, not a gate ({observed})", cause="bold")
        return _escalate(result, gw.get("confidence"))
    return FieldResult("government_warning", text, GOVERNMENT_WARNING, REVIEW,
                       f"wording and ALL-CAPS verified — a reviewer must confirm "
                       f"'{GOVERNMENT_WARNING_HEADER}' is in BOLD (not machine-verifiable)",
                       cause="bold")


# --- presence-only check (batch screening, no application data) --------------

def _check_presence(field_name, field_obj) -> FieldResult:
    """Is a mandatory field present and readable on the label?"""
    present, value, conf = _get(field_obj)
    label = field_name.replace("_", " ")
    if value:
        return _escalate(FieldResult(field_name, value, "", PASS, "present on the label"), conf)
    if present:
        return FieldResult(field_name, "", "", REVIEW, f"{label} appears present but is unreadable")
    return FieldResult(field_name, "", "", FAIL, f"required {label} not found on the label")


# --- image-quality-aware reframing -------------------------------------------
# When the photo itself is the problem (soft/small/glare/angle/crop/low-res), a missing field,
# an unreadable field, an unverifiable-bold read, or a near-miss wording read is a PHOTO problem
# -- not proof the physical label is noncompliant. We KEEP the fail/review verdict (the required
# info still couldn't be verified) but reword the reason so it doesn't accuse the label. Clean
# images (no quality note -- e.g. the controlled adversarial set) are never reframed, so the
# compliance logic there is unchanged.
_IMAGE_QUALITY_TERMS = (
    "glar", "blur", "soft", "small", "tiny", "faint", "obscur", "crop", "cut off", "cut-off",
    "angle", "rotat", "curv", "reflect", "resolution", "low res", "low-res", "shadow", "unclear",
    "partial", "hard to read", "difficult to read",
)
_UNVERIFIABLE_REASON = ("could not verify required label information from this image; "
                        "submit a clearer label image")
# reasons that mean "couldn't read it from this photo" (vs a definite, legible violation like
# title case, clearly-wrong wording, or a HIGH-CONFIDENCE bold violation, which are NOT reframed).
# NOTE: the confident bold-violation messages ("does not appear to be in bold" / "appears to be in
# bold") are deliberately NOT listed here -- under header_body_gate those only fire on a
# high-confidence determination, so they are definite findings, not readability problems, and
# rewording them to "submit a clearer image" would mislead the reviewer about the most important
# rule. The genuinely-unverifiable bold outcomes still reframe via "could not confirm" /
# "could not verify that" (header_body_gate REVIEW and confidence_gate's null/low fail-closed).
_READABILITY_REASON_HINTS = (
    "not found on the label", "could not be read", "appears present but", "is unreadable",
    "could not verify that", "could not confirm", "close but not an exact match",
)


def _image_is_low_quality(notes) -> bool:
    n = (notes or "").lower()
    return bool(n) and any(t in n for t in _IMAGE_QUALITY_TERMS)


def _reframe_for_image_quality(results, image_quality_notes):
    """Reword readability-driven failures to the image-quality message when the photo is flagged
    low-quality. Verdicts are unchanged; clean images are untouched."""
    if not _image_is_low_quality(image_quality_notes):
        return results
    out = []
    for r in results:
        if r.status in (FAIL, REVIEW) and any(h in r.reason for h in _READABILITY_REASON_HINTS):
            # keep .cause: it is the stable programmatic channel precisely because the
            # display reason gets reworded (here and by future copy edits)
            out.append(FieldResult(r.field, r.extracted, r.expected, r.status,
                                   _UNVERIFIABLE_REASON, cause=r.cause))
        else:
            out.append(r)
    return out


# --- roll-ups ----------------------------------------------------------------

def _rollup(results, extracted):
    overall = max(results, key=lambda r: _SEVERITY[r.status]).status
    return {
        "overall": overall,
        "fields": results,
        "beverage_type": extracted.get("beverage_type", "unknown"),
        "additional_statements": extracted.get("additional_statements", []),
        "image_quality_notes": extracted.get("image_quality_notes"),
    }


def _candidates(application, *keys):
    """Acceptable application values for a union match (e.g. brand_name OR fanciful_name).
    Drops empty/missing keys. The extractor is never shown any of these — the union is a
    verification-side allowance for the model picking a different-but-legitimate name."""
    return [application.get(k) for k in keys if application.get(k)]


def verify(extracted: dict, application: dict) -> dict:
    """Full verification: match label fields against the application AND check the rules.

    The government warning is a hard gate: if it fails, the label fails (its FAIL is the
    worst severity, so the worst-status roll-up already enforces this). Otherwise the
    overall status is the worst field status (fail > needs-review > pass).
    """
    extracted = extracted or {}
    application = application or {}
    beverage_type = extracted.get("beverage_type", "unknown")

    results = [
        _check_text("brand_name", extracted.get("brand_name"),
                    _candidates(application, "brand_name", "fanciful_name"),
                    scorer=fuzz.token_set_ratio, near_miss_review=True),
        _check_text("class_type", extracted.get("class_type"),
                    _candidates(application, "class_type", "statement_of_composition"),
                    scorer=fuzz.token_set_ratio, near_miss_review=True),
        _check_abv(extracted.get("alcohol_content"), application.get("alcohol_content"),
                   beverage_type, extracted.get("class_type")),
        _check_net_contents(extracted.get("net_contents"), application.get("net_contents")),
        _check_name_address(extracted.get("name_and_address"), application.get("name_and_address")),
        _check_country(extracted.get("country_of_origin"), application.get("country_of_origin")),
        _check_warning(extracted.get("government_warning")),
    ]
    if beverage_type == "wine":
        results.append(_check_appellation(beverage_type, extracted.get("class_type"),
                                          extracted.get("vintage"), extracted.get("appellation")))
    results = _reframe_for_image_quality(results, extracted.get("image_quality_notes"))
    return _rollup(results, extracted)


def verify_label_only(extracted: dict) -> dict:
    """Rules-only screening for batch use: check the government warning and the
    presence of mandatory fields on the label, with no application data to match
    against. Returns the same shape as verify().

    Country of origin is omitted here because import status (hence whether it is
    required) can't be known without the application."""
    extracted = extracted or {}
    beverage_type = extracted.get("beverage_type", "unknown")
    results = [
        _check_presence("brand_name", extracted.get("brand_name")),
        _check_presence("class_type", extracted.get("class_type")),
        _check_abv(extracted.get("alcohol_content"), None, beverage_type, extracted.get("class_type")),
        _check_presence("net_contents", extracted.get("net_contents")),
        _check_presence("name_and_address", extracted.get("name_and_address")),
        _check_warning(extracted.get("government_warning")),
    ]
    if beverage_type == "wine":
        results.append(_check_appellation(beverage_type, extracted.get("class_type"),
                                          extracted.get("vintage"), extracted.get("appellation")))
    results = _reframe_for_image_quality(results, extracted.get("image_quality_notes"))
    return _rollup(results, extracted)
