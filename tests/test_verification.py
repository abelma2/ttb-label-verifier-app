"""Unit tests for the deterministic verification logic and extraction coercion.

These are pure (no network/API) and cover the regulation-critical paths:
the government-warning fail-closed rule and the class-dependent ABV rule.

Run:  pytest
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import verification
import extraction
from config import GOVERNMENT_WARNING
from extraction import _coerce, _build_content, _model_params, _parse_response, ExtractionError
from verification import (
    verify, verify_label_only,
    _check_text, _check_abv, _check_net_contents, _check_country,
    _check_name_address, _without_relationship_prefix, _check_warning, _check_presence,
    _check_appellation, _parse_abv,
    PASS, REVIEW, FAIL,
)


# --- helpers -----------------------------------------------------------------

def field(value, present=None, confidence="high", **extra):
    if present is None:
        present = value is not None
    d = {"present": present, "value": value, "confidence": confidence}
    d.update(extra)
    return d


def warning(text=GOVERNMENT_WARNING, present=True, caps=True, bold=True, confidence="high",
            bold_confidence="high", body_bold=False, body_bold_confidence="high"):
    return {"present": present, "text": text, "header_all_caps": caps,
            "header_bold": bold, "header_bold_confidence": bold_confidence,
            "header_bold_basis": None, "body_bold": body_bold,
            "body_bold_confidence": body_bold_confidence, "confidence": confidence}


# --- text fields -------------------------------------------------------------

def test_brand_fuzzy_tolerates_case_and_apostrophe():
    # the canonical Dave Morrison example: "STONE'S THROW" vs "Stone's Throw"
    assert _check_text("brand_name", field("STONE'S THROW"), "Stone’s Throw").status == PASS


def test_brand_missing_on_label_fails():
    assert _check_text("brand_name", field(None, present=False), "Old Tom").status == FAIL


def test_brand_blank_application_is_review_not_fail():
    assert _check_text("brand_name", field("Old Tom"), "").status == REVIEW


def test_text_low_confidence_escalates_pass_to_review():
    assert _check_text("brand_name", field("Old Tom", confidence="low"), "Old Tom").status == REVIEW


def test_text_accepts_a_list_and_takes_the_best_candidate():
    # expected may be a union of acceptable values; the best match wins.
    r = _check_text("brand_name", field("Stormchaser White"), ["Lighthouse", "Stormchaser White"])
    assert r.status == PASS
    assert r.expected == "Stormchaser White"   # reports which candidate matched


def test_brand_union_superset_read_passes():
    """A more-verbose brand read (model merged brand + class) still matches the legal brand
    via the containment-aware token_set_ratio used for brand/class in verify()."""
    extract = _good_spirits_extract()
    extract["brand_name"] = field("CAPTAIN JOHN'S SPICED RUM")
    app = dict(_good_application(), brand_name="Captain John's")
    fields = {f.field: f.status for f in verify(extract, app)["fields"]}
    assert fields["brand_name"] == PASS


def test_brand_union_matches_fanciful_name():
    """The model may tag the fanciful name as the brand; matching the union of
    brand_name + fanciful_name still passes."""
    extract = _good_wine_extract()
    extract["brand_name"] = field("STORMCHASER WHITE")
    app = {"brand_name": "Lighthouse", "fanciful_name": "Stormchaser White",
           "class_type": "Chardonnay"}
    fields = {f.field: f.status for f in verify(extract, app)["fields"]}
    assert fields["brand_name"] == PASS


def test_class_union_matches_statement_of_composition():
    """The model may read the statement of composition as the class; the union with
    statement_of_composition still passes."""
    extract = _good_spirits_extract()
    extract["class_type"] = field("RUM WITH NATURAL FLAVORS ADDED")
    app = dict(_good_application(), class_type="Spiced Rum",
               statement_of_composition="Rum with natural flavors added")
    fields = {f.field: f.status for f in verify(extract, app)["fields"]}
    assert fields["class_type"] == PASS


def test_brand_union_does_not_mask_genuine_mismatch():
    """The union must not hide a real mismatch: a fanciful read with no matching
    application fanciful field still fails."""
    extract = _good_spirits_extract()
    extract["brand_name"] = field("Stormchaser White")
    app = dict(_good_application(), brand_name="Lighthouse")   # no fanciful_name supplied
    fields = {f.field: f.status for f in verify(extract, app)["fields"]}
    assert fields["brand_name"] == FAIL


def test_brand_one_letter_typo_reviews():
    """"Jon's" vs application "John's" scores ~96 (would auto-pass); the edit-distance guard
    routes a 1-2 character difference in a short identity field to review."""
    extract = _good_spirits_extract()
    extract["brand_name"] = field("CAPTAIN JON'S")
    app = dict(_good_application(), brand_name="Captain John's")
    fields = {f.field: f.status for f in verify(extract, app)["fields"]}
    assert fields["brand_name"] == REVIEW


# --- alcohol content: matching -----------------------------------------------

def test_abv_match_within_tolerance():
    ac = field("45% Alc./Vol. (90 Proof)", abv_percent=45.0, proof=90.0)
    assert _check_abv(ac, "45%", "spirits", field("Bourbon")).status == PASS


def test_abv_mismatch_fails():
    ac = field("40% Alc./Vol.", abv_percent=40.0, proof=None)
    assert _check_abv(ac, "45%", "spirits", field("Bourbon")).status == FAIL


def test_abv_small_difference_is_review():
    ac = field("45.3% Alc./Vol.", abv_percent=45.3, proof=None)
    assert _check_abv(ac, "45%", "spirits", field("Bourbon")).status == REVIEW


def test_abv_proof_not_parsed_as_abv():
    # abv_percent comes pre-parsed from extraction; the 90 proof must not win
    ac = field("45% Alc./Vol. (90 Proof)", abv_percent=45.0, proof=90.0)
    assert _check_abv(ac, "45%", "spirits", field("Bourbon")).status == PASS


def test_proof_inconsistent_with_abv_fails():
    # 50 proof on a 20% ABV label is impossible (US proof = 2 x ABV) — internal inconsistency
    ac = field("20% Alcohol by Volume (50 Proof)", abv_percent=20.0, proof=50.0)
    assert _check_abv(ac, "20%", "spirits", field("Spiced Rum")).status == FAIL


def test_proof_consistent_with_abv_passes():
    ac = field("45% Alc./Vol. (90 Proof)", abv_percent=45.0, proof=90.0)
    assert _check_abv(ac, "45%", "spirits", field("Bourbon")).status == PASS


def test_abv_notation_abbreviation_fails():
    # the bare abbreviation "ABV" is not a TTB-prescribed notation (27 CFR 7.65)
    ac = field("5% ABV", abv_percent=5.0, proof=None)
    assert _check_abv(ac, "5%", "beer", field("India Pale Ale")).status == FAIL


def test_abv_notation_compliant_passes():
    ac = field("5% ALC./VOL.", abv_percent=5.0, proof=None)
    assert _check_abv(ac, "5%", "beer", field("India Pale Ale")).status == PASS


# --- alcohol content: class-dependent presence rule --------------------------

def test_abv_missing_on_spirits_fails():
    ac = field(None, present=False, abv_percent=None, proof=None)
    assert _check_abv(ac, "45%", "spirits", field("Bourbon")).status == FAIL


def test_abv_missing_on_beer_passes():
    ac = field(None, present=False, abv_percent=None, proof=None)
    assert _check_abv(ac, None, "beer", field("India Pale Ale")).status == PASS


def test_abv_missing_on_beer_but_application_lists_one_is_review():
    ac = field(None, present=False, abv_percent=None, proof=None)
    assert _check_abv(ac, "5%", "beer", field("India Pale Ale")).status == REVIEW


def test_abv_missing_on_table_wine_passes():
    ac = field(None, present=False, abv_percent=None, proof=None)
    assert _check_abv(ac, None, "wine", field("California Table Wine")).status == PASS


def test_abv_missing_on_nontable_wine_is_review():
    ac = field(None, present=False, abv_percent=None, proof=None)
    assert _check_abv(ac, None, "wine", field("Cabernet Sauvignon")).status == REVIEW


# --- net contents ------------------------------------------------------------

def test_net_contents_whitespace_insensitive():
    assert _check_net_contents(field("750 mL"), "750ml").status == PASS


def test_net_contents_missing_fails():
    assert _check_net_contents(field(None, present=False), "750 mL").status == FAIL


def test_net_contents_exact_passes():
    assert _check_net_contents(field("750 mL"), "750 mL").status == PASS


def test_net_contents_same_volume_different_unit_is_review():
    # 16.9 fl oz == 1 pint 0.9 fl oz: same volume, different unit/format -> review, not pass/fail
    assert _check_net_contents(field("16.9 FL. OZ."), "1 PINT 0.9 FL. OZ.").status == REVIEW


def test_net_contents_liter_vs_ml_same_volume_is_review():
    # 0.75 L == 750 mL: same volume, different unit -> review (do not auto-pass on different unit)
    assert _check_net_contents(field("0.75 L"), "750 mL").status == REVIEW


def test_net_contents_floz_vs_ml_same_volume_is_review():
    # 12 fl oz ~= 355 mL (within the rounding tolerance) -> review
    assert _check_net_contents(field("12 FL OZ"), "355 mL").status == REVIEW


def test_net_contents_materially_different_volume_fails():
    assert _check_net_contents(field("500 mL"), "750 mL").status == FAIL


def test_net_contents_floz_materially_different_fails():
    # 8 fl oz (~237 mL) vs 750 mL is a real mismatch, not a unit-format difference
    assert _check_net_contents(field("8 FL OZ"), "750 mL").status == FAIL


def test_net_contents_unparseable_falls_back_to_fuzzy():
    # no recognized unit -> falls back to the fuzzy string compare (genuine mismatch -> FAIL)
    assert _check_net_contents(field("a dozen bottles"), "750 mL").status == FAIL


def test_net_contents_dual_declaration_not_double_counted():
    # "16.9 FL OZ (500 mL)" must NOT be summed to ~1000 mL: the two systems agree on 500 mL, so
    # vs application "500 mL" it is the same volume in a different format -> REVIEW (not FAIL).
    assert _check_net_contents(field("16.9 FL OZ (500 mL)"), "500 mL").status == REVIEW


def test_parse_volume_compound_dual_and_unparseable():
    assert abs(verification._parse_volume("1 PINT 0.9 FL. OZ.") - 499.79) < 0.5
    assert abs(verification._parse_volume("0.75 L") - 750.0) < 0.01
    assert abs(verification._parse_volume("12 FL OZ") - 354.88) < 0.5
    assert abs(verification._parse_volume("16.9 FL OZ (500 mL)") - 500.0) < 1.0  # reconciled, not summed
    assert verification._parse_volume("a dozen") is None
    assert verification._parse_volume("750") is None  # bare number, no unit


def test_parse_volume_thousands_separator_and_restatement():
    # review-finding regressions: thousands comma, parenthetical/repeated restatement (not summed)
    assert verification._parse_volume("1,750 mL") == 1750.0           # comma = thousands separator
    assert abs(verification._parse_volume("0.75 L (750 mL)") - 750.0) < 0.5   # paren restates, not +750
    assert abs(verification._parse_volume("750 mL 750 mL") - 750.0) < 0.5     # duplicate read de-duped


def test_net_contents_thousands_separator_reviews_same_volume():
    # "1,750 mL" must parse as 1750 (not 750) so it matches "1.75 L" as the same volume
    assert _check_net_contents(field("1,750 mL"), "1.75 L").status == REVIEW


def test_net_contents_same_system_restatement_not_double_counted():
    # "0.75 L (750 mL)" is ONE volume restated, not 1500 mL -> must not FAIL a correct 750 mL label
    assert _check_net_contents(field("0.75 L (750 mL)"), "750 mL").status == REVIEW


def test_net_contents_dual_declaration_rounding_reviews():
    # a compliant miniature "1 FL OZ (30 mL)" (29.57 mL ~ 30 mL) vs app "30 mL" -> same volume review
    assert _check_net_contents(field("1 FL OZ (30 mL)"), "30 mL").status == REVIEW


def test_net_contents_zero_volume_uses_volume_path_not_fuzzy():
    # 0.0 is a valid parsed volume (is-not-None guard), so two zero volumes route via the volume
    # path to REVIEW rather than being treated as "unparseable" and fuzzy-compared
    assert _check_net_contents(field("0 mL"), "0 L").status == REVIEW


# --- name & address ----------------------------------------------------------

def test_name_address_match():
    r = _check_name_address(field("BOTTLED BY OLD TOM DISTILLERY, BARDSTOWN, KY"),
                            "Old Tom Distillery, Bardstown, KY")
    assert r.status == PASS


def test_name_address_missing_when_required_fails():
    assert _check_name_address(field(None, present=False), "Old Tom Distillery").status == FAIL


def test_name_address_city_only_subset_reviews():
    # token_set_ratio would PASS the contained "Hyattsville, MD", but the read dropped the
    # bottler name — the coverage guard routes this strict short subset to review.
    r = _check_name_address(field("HYATTSVILLE, MD"),
                            "Brewed & Bottled By Malt & Hop Brewery, Hyattsville, MD")
    assert r.status == REVIEW


def test_name_address_full_read_still_passes():
    # a complete (super-set) read must not be touched by the guard
    r = _check_name_address(field("DISTILLED & BOTTLED BY: ABC DISTILLERY FREDERICK, MD"),
                            "Distilled & Bottled By ABC Distillery, Frederick, MD")
    assert r.status == PASS


def test_name_address_minor_omission_still_passes():
    # drops only the "Bottled by" connective; producer + city/state intact -> still a match,
    # coverage stays above the floor so the guard does NOT downgrade it
    r = _check_name_address(field("OLD TOM DISTILLERY, BARDSTOWN, KY"),
                            "Bottled by Old Tom Distillery, Bardstown, KY")
    assert r.status == PASS


# name/address punctuation/prefix normalization (reduces false reviews; must not pass substitutions)

def test_name_address_prefix_asymmetry_equivalent_passes():
    # production case: the label carries the "Distilled & Bottled By:" relationship prefix that the
    # application value omits. Once punctuation/prefix are normalized they are equivalent -> PASS
    # (this previously false-reviewed at token_set_ratio 74).
    r = _check_name_address(field("DISTILLED & BOTTLED BY: ABC DISTILLERY FREDERICK, MD"),
                            "ABC Distillery, Frederick, MD")
    assert r.status == PASS


def test_name_address_ampersand_equals_and():
    r = _check_name_address(field("Malt and Hop Brewery, Hyattsville, MD"),
                            "Malt & Hop Brewery, Hyattsville, MD")
    assert r.status == PASS


def test_name_address_comma_only_difference_passes():
    r = _check_name_address(field("Old Tom Distillery Bardstown KY"),
                            "Old Tom Distillery, Bardstown, KY")
    assert r.status == PASS


def test_name_address_dropped_producer_no_prefix_reviews():
    # the relationship prefix is NOT part of the name: dropping the producer name still reviews
    # even when the expected value carries no prefix (coverage guard, prefix-invariant).
    r = _check_name_address(field("HYATTSVILLE, MD"), "Malt & Hop Brewery, Hyattsville, MD")
    assert r.status == REVIEW


def test_name_address_totally_different_fails():
    r = _check_name_address(field("XYZ Brewing, Boston, MA"), "ABC Distillery, Frederick, MD")
    assert r.status == FAIL


# the LEADING relationship phrase is stripped from the DISPLAYED read (the match score was
# already phrase-invariant): the field shows just the producer name and address. Applied at
# the verify()/verify_label_only() call sites, so these go through the public entry points.

def test_address_leading_relationship_phrase_stripped_from_read():
    extract = _good_spirits_extract()
    extract["name_and_address"] = field("DISTILLED AND BOTTLED BY: ABC DISTILLERY FREDERICK,MD")
    app = dict(_good_application(), name_and_address="ABC Distillery Frederick, MD")
    na = next(f for f in verify(extract, app)["fields"] if f.field == "name_and_address")
    assert na.status == PASS
    assert na.extracted == "ABC DISTILLERY FREDERICK,MD"


def test_address_prefix_stripped_in_label_only_screening():
    extract = _good_wine_extract()
    extract["name_and_address"] = field("VINTED AND BOTTLED BY 19 CRIMES, SONOMA, CA")
    na = next(f for f in verify_label_only(extract)["fields"] if f.field == "name_and_address")
    assert na.status == PASS
    assert na.extracted == "19 CRIMES, SONOMA, CA"


def test_address_phrase_only_read_keeps_original_value():
    # a read that is ONLY the phrase must not be emptied into the missing-value path
    assert _without_relationship_prefix(field("BOTTLED BY"))["value"] == "BOTTLED BY"


def test_address_midstring_phrase_untouched():
    # only a LEADING phrase is stripped; one inside the producer statement is left alone
    untouched = "ABC FARMS, PRODUCED AND BOTTLED BY ABC, NAPA, CA"
    assert _without_relationship_prefix(field(untouched))["value"] == untouched


@pytest.mark.xfail(strict=True, reason="KNOWN GAP: a producer-name word substitution still passes "
                   "the fuzzy score. The token-level guard was removed (it false-reviewed common "
                   "formatting differences); see _check_name_address docstring.")
def test_name_address_producer_substitution_does_not_pass():
    # DESIRED: a producer-name word substitution ("Wineries" for "Vintners") should NOT silently
    # pass on a high token_set_ratio. Currently it does -> xfail documents the gap.
    r = _check_name_address(field("Lighthouse Wineries, Kingston, NY"),
                            "Lighthouse Vintners, Kingston, NY")
    assert r.status != PASS


@pytest.mark.xfail(strict=True, reason="KNOWN GAP: producer-name substitution behind the "
                   "relationship prefix still passes the fuzzy score (guard removed).")
def test_name_address_producer_substitution_with_prefix_does_not_pass():
    r = _check_name_address(field("PRODUCED AND BOTTLED BY LIGHTHOUSE WINERIES KINGSTON, NY"),
                            "Produced and Bottled By Lighthouse Vintners, Kingston, NY")
    assert r.status != PASS


def test_name_address_full_state_name_vs_abbrev_passes():
    # regression (review finding): the application typing the full state name while the label uses
    # the postal abbreviation must NOT false-review -- it is a formatting difference, not a misread.
    r = _check_name_address(field("ABC Distillery, Frederick, MD"),
                            "ABC Distillery, Frederick, Maryland")
    assert r.status == PASS


def test_name_address_app_zip_not_on_label_passes():
    # regression (review finding): an application ZIP the label doesn't carry must not false-review.
    r = _check_name_address(field("ABC Distillery, Bardstown, KY"),
                            "ABC Distillery, Bardstown, KY 40004")
    assert r.status == PASS


def test_name_address_dropped_apostrophe_passes():
    # regression (review finding): the model dropping an apostrophe in the producer name must not
    # false-review ("Johns" vs "John's").
    r = _check_name_address(field("Johns Distillery, Bardstown, KY"),
                            "John's Distillery, Bardstown, KY")
    assert r.status == PASS


# --- country of origin -------------------------------------------------------

def test_country_domestic_passes_when_absent():
    assert _check_country(field(None, present=False), "").status == PASS


def test_country_import_match_passes():
    assert _check_country(field("PRODUCT OF SCOTLAND"), "Scotland").status == PASS


def test_country_import_missing_fails():
    assert _check_country(field(None, present=False), "Scotland").status == FAIL


# --- appellation of origin (wine, conditionally mandatory) -------------------

def test_appellation_required_varietal_missing_fails():
    # a varietal designation (Chardonnay) triggers the appellation requirement
    r = _check_appellation("wine", field("Chardonnay"), field(None, present=False),
                           field(None, present=False))
    assert r.status == FAIL and "appellation" in r.reason.lower()


def test_appellation_required_vintage_missing_fails():
    # a vintage date triggers the requirement even without a varietal
    r = _check_appellation("wine", field("Red Wine"), field("2018"), field(None, present=False))
    assert r.status == FAIL


def test_appellation_required_semi_generic_missing_fails():
    # a semi-generic type designation (27 CFR 4.24(b)), e.g. "Burgundy", triggers the requirement
    r = _check_appellation("wine", field("Burgundy"), field(None, present=False),
                           field(None, present=False))
    assert r.status == FAIL and "semi-generic" in r.reason.lower()


def test_appellation_semi_generic_with_embedded_appellation_passes():
    # "California Burgundy": the embedded state appellation satisfies the triggered requirement
    r = _check_appellation("wine", field("California Burgundy"), field(None, present=False),
                           field(None, present=False))
    assert r.status == PASS


def test_appellation_semi_generic_word_boundary_no_false_trigger():
    # "port" must match on word boundaries only: "Portland" is NOT a semi-generic designation, so
    # a plain red wine with no varietal/vintage stays not-required (regression guard for substring
    # matching that would wrongly fire on "Portland"/"porter").
    r = _check_appellation("wine", field("Portland Red Wine"), field(None, present=False),
                           field(None, present=False))
    assert r.status == PASS and "not required" in r.reason.lower()


def test_appellation_present_when_required_passes():
    r = _check_appellation("wine", field("Chardonnay"), field("2018"), field("Hudson River Region"))
    assert r.status == PASS


def test_appellation_not_required_without_varietal_or_vintage():
    r = _check_appellation("wine", field("Red Wine"), field(None, present=False),
                           field(None, present=False))
    assert r.status == PASS


def test_appellation_not_applicable_for_non_wine():
    r = _check_appellation("spirits", field("Bourbon"), field(None, present=False),
                           field(None, present=False))
    assert r.status == PASS and "not applicable" in r.reason


def test_appellation_unreadable_when_required_is_review():
    r = _check_appellation("wine", field("Chardonnay"), field(None, present=False),
                           field(None, present=True, confidence="low"))
    assert r.status == REVIEW


def _good_wine_extract():
    return {
        "beverage_type": "wine",
        "brand_name": field("Lighthouse"),
        "class_type": field("Chardonnay"),
        "alcohol_content": field("13.5% Alc./Vol.", abv_percent=13.5, proof=None),
        "net_contents": field("750 mL"),
        "name_and_address": field("PRODUCED AND BOTTLED BY LIGHTHOUSE VINTNERS, KINGSTON, NY"),
        "country_of_origin": field(None, present=False),
        "appellation": field("Hudson River Region"),
        "vintage": field("2018"),
        "government_warning": warning(),
        "additional_statements": [],
        "image_quality_notes": None,
    }


def test_verify_label_only_wine_includes_appellation_and_passes():
    result = verify_label_only(_good_wine_extract())
    assert "appellation" in [f.field for f in result["fields"]]
    assert result["overall"] == PASS


def test_verify_label_only_wine_missing_appellation_fails():
    extract = _good_wine_extract()
    extract["appellation"] = field(None, present=False)   # Chardonnay + 2018 but no appellation
    assert verify_label_only(extract)["overall"] == FAIL


# --- government warning: the critical fail-closed rule -----------------------

def test_warning_compliant_with_bold_true_passes():
    # Under the default medium_pass_gate policy, wording + ALL-CAPS + a confident bold header
    # AND a confident non-bold body are all verified (warning() defaults body_bold=False/high),
    # so a compliant warning PASSES.
    assert _check_warning(warning()).status == PASS


def test_warning_note_policy_does_not_gate_on_bold(monkeypatch):
    # "note": header_bold (and its confidence) are telemetry only -> always PASS, never gate
    monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "note")
    assert _check_warning(warning(bold=True)).status == PASS
    assert _check_warning(warning(bold=False)).status == PASS
    assert _check_warning(warning(bold=None)).status == PASS
    assert _check_warning(warning(bold=True, bold_confidence="low")).status == PASS


# --- header_body_gate policy: header bold AND body-not-bold, both at HIGH confidence ------------
# (the stricter prior default; still env-selectable via WARNING_BOLD_POLICY=header_body_gate)

def test_default_warning_bold_policy_is_medium_pass_gate():
    # pins the SHIPPED default (review #8). Every policy test below monkeypatches the mode, so
    # without this nothing would catch a reverted config default. (Fails if the WARNING_BOLD_POLICY
    # env var is set in the test environment -- that is the intended signal that it is not the default.)
    # Default relaxed from header_body_gate to medium_pass_gate 2026-06-11 per course-staff guidance.
    assert verification.WARNING_BOLD_POLICY == "medium_pass_gate"


def test_default_policy_medium_confidence_compliant_passes():
    # BEHAVIORAL pin of the default (no monkeypatch): the relaxed gate's distinguishing cell —
    # both bold rules satisfied at MEDIUM confidence — PASSES under the shipped default. Under the
    # prior header_body_gate default this exact input went to REVIEW, so this catches a quiet
    # revert that the string-equality pin above alone would not surface as a verdict change.
    gw = warning(bold=True, bold_confidence="medium",
                 body_bold=False, body_bold_confidence="medium")
    assert _check_warning(gw).status == PASS


def test_header_body_gate_pass_requires_both(monkeypatch):
    monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "header_body_gate")
    assert _check_warning(warning(bold=True, bold_confidence="high",
                                  body_bold=False, body_bold_confidence="high")).status == PASS


def test_header_body_gate_header_true_alone_does_not_pass(monkeypatch):
    # the new rule: header_bold=True by itself no longer passes -- the body must be confirmed not bold
    monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "header_body_gate")
    r = _check_warning(warning(bold=True, bold_confidence="high",
                               body_bold=None, body_bold_confidence="low"))
    assert r.status == REVIEW


def test_header_body_gate_header_not_bold_high_fails(monkeypatch):
    monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "header_body_gate")
    r = _check_warning(warning(bold=False, bold_confidence="high"))
    assert r.status == FAIL and "bold" in r.reason.lower()


def test_header_body_gate_body_bold_high_fails(monkeypatch):
    # all-bold violation (27 CFR 16.22 "remainder may not be bold"): header bold but body also bold
    monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "header_body_gate")
    r = _check_warning(warning(bold=True, bold_confidence="high",
                               body_bold=True, body_bold_confidence="high"))
    assert r.status == FAIL and "body" in r.reason.lower()


def test_header_body_gate_uncertain_reviews(monkeypatch):
    monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "header_body_gate")
    # header only medium-confidence -> review
    assert _check_warning(warning(bold=True, bold_confidence="medium")).status == REVIEW
    # body bold medium-confidence -> review (not a high-confidence violation)
    assert _check_warning(warning(bold=True, bold_confidence="high",
                                  body_bold=True, body_bold_confidence="medium")).status == REVIEW


def test_header_body_gate_body_bold_high_fails_regardless_of_header(monkeypatch):
    # the body-not-bold rule fires INDEPENDENTLY of the header read: a confident bold body is a
    # FAIL (27 CFR 16.22) even when the header is unreadable (None) or itself not-bold -- pins the
    # branch ordering so a refactor can't silently demote the doubly-/body-violating label to REVIEW.
    monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "header_body_gate")
    assert _check_warning(warning(bold=None, bold_confidence="low",
                                  body_bold=True, body_bold_confidence="high")).status == FAIL
    assert _check_warning(warning(bold=False, bold_confidence="high",
                                  body_bold=True, body_bold_confidence="high")).status == FAIL


def test_header_body_gate_compliant_body_medium_confidence_reviews(monkeypatch):
    # PASS genuinely requires HIGH confidence on body_bold=False: a compliant-looking warning whose
    # non-bold body the model is only MEDIUM-confident about must go to REVIEW, not PASS (review #4).
    # Pins the body-confidence clause of the PASS gate so a loosening refactor (e.g. accepting
    # `body_bold is not True`) can't silently auto-pass this common real-world cell.
    monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "header_body_gate")
    r = _check_warning(warning(bold=True, bold_confidence="high",
                               body_bold=False, body_bold_confidence="medium"))
    assert r.status == REVIEW


def test_header_body_gate_both_violations_reported_together(monkeypatch):
    # when BOTH visual rules are violated at high confidence, the FAIL reason names BOTH the header
    # and the body, so a reviewer doesn't fix only the header and resubmit a still-bold body (review #5).
    monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "header_body_gate")
    r = _check_warning(warning(bold=False, bold_confidence="high",
                               body_bold=True, body_bold_confidence="high"))
    assert r.status == FAIL
    assert "GOVERNMENT WARNING" in r.reason and "body" in r.reason.lower()


# --- medium_pass_gate policy: like header_body_gate but the PASS gate accepts MEDIUM confidence ---
# THE PRODUCTION DEFAULT (since 2026-06-11, per course-staff guidance). FAIL is identical to
# header_body_gate (only a HIGH-confidence violation of either visual rule fails); the sole
# behavioral delta is that medium-confidence, both-rules-satisfied reads move from REVIEW
# (under header_body_gate) to PASS.

# (header_bold, header_conf, body_bold, body_conf) -> expected status under medium_pass_gate.
_MEDIUM_PASS_GATE_TABLE = [
    # PASS: header_bold True AND body_bold False, each at MEDIUM-or-high confidence.
    ((True,  "high",   False, "high"),   PASS),  # also passes header_body_gate
    ((True,  "high",   False, "medium"), PASS),  # DELTA: header_body_gate -> REVIEW
    ((True,  "medium", False, "high"),   PASS),  # DELTA: header_body_gate -> REVIEW
    ((True,  "medium", False, "medium"), PASS),  # DELTA: header_body_gate -> REVIEW
    # FAIL: a HIGH-confidence violation of either rule (IDENTICAL to header_body_gate).
    ((False, "high",   False, "high"),   FAIL),  # header not bold (high)
    ((False, "high",   True,  "high"),   FAIL),  # both rules violated (high)
    ((True,  "high",   True,  "high"),   FAIL),  # body bold (high)
    ((None,  "low",    True,  "high"),   FAIL),  # body bold (high) regardless of header
    ((False, "high",   None,  "low"),    FAIL),  # header not bold (high) regardless of body
    # REVIEW: nulls, low confidence, or MEDIUM-confidence violations (not high -> never FAIL/PASS).
    ((True,  "low",    False, "high"),   REVIEW),  # header bold but LOW conf -> can't pass
    ((True,  "high",   False, "low"),    REVIEW),  # body not-bold but LOW conf -> can't pass
    ((None,  "high",   False, "high"),   REVIEW),  # header null -> can't pass, not a violation
    ((True,  "high",   None,  "high"),   REVIEW),  # body null -> can't pass, not a violation
    ((False, "medium", False, "high"),   REVIEW),  # MEDIUM-confidence header violation -> NOT fail
    ((True,  "high",   True,  "medium"), REVIEW),  # MEDIUM-confidence body violation -> NOT fail
    ((False, "low",    False, "high"),   REVIEW),  # LOW-confidence header "violation" -> review
    ((True,  "medium", True,  "medium"), REVIEW),  # ok-direction header (med) + medium body violation
]


def test_medium_pass_gate_full_truth_table(monkeypatch):
    monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "medium_pass_gate")
    for (hb, hc, bb, bc), expected in _MEDIUM_PASS_GATE_TABLE:
        r = _check_warning(warning(bold=hb, bold_confidence=hc,
                                   body_bold=bb, body_bold_confidence=bc))
        assert r.status == expected, (
            "medium_pass_gate(header_bold=%r[%s], body_bold=%r[%s]) -> %s, expected %s"
            % (hb, hc, bb, bc, r.status, expected))


def test_medium_pass_gate_passes_medium_where_header_body_gate_reviews(monkeypatch):
    # The whole point of the variant: medium-confidence, both-rules-satisfied reads PASS here but
    # go to REVIEW under the stricter prior-default header_body_gate. SAME input, verdict differs
    # ONLY at this cell.
    for hb, hc, bb, bc in [(True, "high", False, "medium"),
                           (True, "medium", False, "high"),
                           (True, "medium", False, "medium")]:
        gw = warning(bold=hb, bold_confidence=hc, body_bold=bb, body_bold_confidence=bc)
        monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "header_body_gate")
        assert _check_warning(gw).status == REVIEW
        monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "medium_pass_gate")
        assert _check_warning(gw).status == PASS


def test_medium_pass_gate_fail_parity_with_header_body_gate(monkeypatch):
    # FAIL behavior MUST be identical to header_body_gate: the looser PASS gate may never soften a
    # high-confidence violation to PASS/REVIEW (this is the false-pass guard).
    for hb, hc, bb, bc in [(False, "high", False, "high"),
                           (False, "high", True,  "high"),
                           (True,  "high", True,  "high"),
                           (None,  "low",  True,  "high"),
                           (False, "high", None,  "low")]:
        gw = warning(bold=hb, bold_confidence=hc, body_bold=bb, body_bold_confidence=bc)
        monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "header_body_gate")
        assert _check_warning(gw).status == FAIL
        monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "medium_pass_gate")
        assert _check_warning(gw).status == FAIL


def test_medium_pass_gate_both_violations_reported_together(monkeypatch):
    # both rules violated at high confidence -> FAIL naming BOTH (same reviewer-clarity behavior).
    monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "medium_pass_gate")
    r = _check_warning(warning(bold=False, bold_confidence="high",
                               body_bold=True, body_bold_confidence="high"))
    assert r.status == FAIL
    assert "GOVERNMENT WARNING" in r.reason and "body" in r.reason.lower()


def test_medium_pass_gate_caps_fail_precedes_bold(monkeypatch):
    # the caps gate still precedes the bold gate -- the looser PASS does not bypass earlier checks.
    monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "medium_pass_gate")
    title = GOVERNMENT_WARNING.replace("GOVERNMENT WARNING:", "Government Warning:")
    assert _check_warning(warning(text=title, caps=False, bold=True)).status == FAIL


# --- confidence_gate policy: header-only, fail-closed (prior default; kept for comparison) -------

def test_confidence_gate_bold_true_confident_passes(monkeypatch):
    monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "confidence_gate")
    assert _check_warning(warning(bold=True, bold_confidence="high")).status == PASS
    assert _check_warning(warning(bold=True, bold_confidence="medium")).status == PASS


def test_confidence_gate_bold_false_confident_fails(monkeypatch):
    monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "confidence_gate")
    r = _check_warning(warning(bold=False, bold_confidence="high"))
    assert r.status == FAIL and "bold" in r.reason.lower()


def test_confidence_gate_low_confidence_fails_unverifiable(monkeypatch):
    monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "confidence_gate")
    # low bold-confidence (even with bold=True) can't be approved -> automated "could not verify"
    r = _check_warning(warning(bold=True, bold_confidence="low"))
    assert r.status == FAIL and "could not verify" in r.reason
    # bold=False but low-confidence also lands on the could-not-verify fail (not the "not bold" one)
    assert _check_warning(warning(bold=False, bold_confidence="low")).status == FAIL


def test_confidence_gate_null_bold_fails_unverifiable(monkeypatch):
    monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "confidence_gate")
    r = _check_warning(warning(bold=None, bold_confidence="high"))
    assert r.status == FAIL and "could not verify" in r.reason


def test_confidence_gate_caps_fail_precedes_bold(monkeypatch):
    monkeypatch.setattr(verification, "WARNING_BOLD_POLICY", "confidence_gate")
    # a caps violation fails on caps, before the bold gate is reached
    title = GOVERNMENT_WARNING.replace("GOVERNMENT WARNING:", "Government Warning:")
    assert _check_warning(warning(text=title, caps=False, bold=True)).status == FAIL


def test_warning_title_case_header_with_caps_false_fails():
    title = GOVERNMENT_WARNING.replace("GOVERNMENT WARNING:", "Government Warning:")
    assert _check_warning(warning(text=title, caps=False)).status == FAIL


def test_warning_title_case_header_with_caps_true_still_fails():
    # the verbatim text proves it is title case, so it fails even if the model says caps=True
    title = GOVERNMENT_WARNING.replace("GOVERNMENT WARNING:", "Government Warning:")
    assert _check_warning(warning(text=title, caps=True, bold=True)).status == FAIL


def test_warning_title_case_header_with_caps_none_fails():
    title = GOVERNMENT_WARNING.replace("GOVERNMENT WARNING:", "Government Warning:")
    assert _check_warning(warning(text=title, caps=None)).status == FAIL


def test_warning_caps_confirmed_via_text_then_bold_passes():
    # caps unknown from the model, but the verbatim text shows the caps header -> caps OK,
    # then bold=True -> PASS (rather than failing on caps)
    assert _check_warning(warning(caps=None, bold=True)).status == PASS


def test_warning_absent_fails():
    assert _check_warning(warning(text=None, present=False)).status == FAIL


def test_warning_wording_mismatch_fails():
    assert _check_warning(warning(text="GOVERNMENT WARNING: drink responsibly")).status == FAIL


def test_warning_near_miss_wording_is_review():
    # a one-word transcription slip is close to exact -> review (human verifies), not a fail
    bad = GOVERNMENT_WARNING.replace("women should not drink", "nobody should drink")
    assert _check_warning(warning(text=bad)).status == REVIEW


def test_warning_lowercase_surgeon_general_fails():
    # the TTB checklist explicitly checks the "S" in Surgeon and "G" in General are caps
    bad = GOVERNMENT_WARNING.replace("Surgeon General", "surgeon general")
    assert _check_warning(warning(text=bad)).status == FAIL


def test_warning_header_omitted_caps_via_flag_then_bold_passes():
    # model omits the header from text but reports caps=True -> caps OK via the flag, then
    # bold=True -> PASS (rather than failing on caps)
    body = GOVERNMENT_WARNING.split(": ", 1)[1].upper()
    assert _check_warning(warning(text=body, caps=True, bold=True)).status == PASS


# --- roll-up -----------------------------------------------------------------

def _good_spirits_extract():
    return {
        "beverage_type": "spirits",
        "brand_name": field("Old Tom"),
        "class_type": field("Kentucky Straight Bourbon Whiskey"),
        "alcohol_content": field("45% Alc./Vol. (90 Proof)", abv_percent=45.0, proof=90.0),
        "net_contents": field("750 mL"),
        "name_and_address": field("BOTTLED BY OLD TOM DISTILLERY, BARDSTOWN, KY"),
        "country_of_origin": field(None, present=False),
        "government_warning": warning(),
        "additional_statements": [],
        "image_quality_notes": None,
    }


def _good_application():
    return {
        "brand_name": "Old Tom",
        "class_type": "Kentucky Straight Bourbon Whiskey",
        "alcohol_content": "45%",
        "net_contents": "750 mL",
        "name_and_address": "Old Tom Distillery, Bardstown, KY",
        "country_of_origin": "",
    }


def test_verify_good_label_passes():
    # a fully-matching label with confident bold=True now PASSES overall
    assert verify(_good_spirits_extract(), _good_application())["overall"] == PASS


def test_warning_failure_is_hard_gate():
    extract = _good_spirits_extract()
    extract["government_warning"] = warning(caps=False)  # title-case header
    assert verify(extract, _good_application())["overall"] == FAIL


def test_overall_is_worst_status():
    extract = _good_spirits_extract()
    extract["alcohol_content"] = field("40% Alc./Vol.", abv_percent=40.0, proof=None)
    result = verify(extract, _good_application())
    assert result["overall"] == FAIL


def test_verify_label_only_screens_warning_and_presence():
    result = verify_label_only(_good_spirits_extract())
    assert result["overall"] == PASS  # warning bold=True -> pass; everything else passes too
    # spirits with no ABV should fail under label-only screening too
    extract = _good_spirits_extract()
    extract["alcohol_content"] = field(None, present=False, abv_percent=None, proof=None)
    assert verify_label_only(extract)["overall"] == FAIL


# --- extraction coercion -----------------------------------------------------

def test_coerce_fills_missing_keys():
    out = _coerce({})
    assert out["beverage_type"] == "unknown"
    assert out["brand_name"] == {"present": False, "value": None, "confidence": "low"}
    assert out["government_warning"]["header_all_caps"] is None
    assert out["additional_statements"] == []


def test_coerce_parses_numeric_abv_and_proof():
    out = _coerce({"alcohol_content": {"present": True, "value": "45% Alc./Vol. (90 Proof)",
                                       "abv_percent": "45", "proof": "90", "confidence": "high"}})
    assert out["alcohol_content"]["abv_percent"] == 45.0
    assert out["alcohol_content"]["proof"] == 90.0


def test_coerce_coerces_nonbool_caps_to_none():
    out = _coerce({"government_warning": {"present": True, "text": "x", "header_all_caps": "yes"}})
    assert out["government_warning"]["header_all_caps"] is None


def test_coerce_government_warning_bold_telemetry_defaults():
    # the bold telemetry keys are always present, defaulting to low / null
    gw = _coerce({})["government_warning"]
    assert gw["header_bold_confidence"] == "low"
    assert gw["header_bold_basis"] is None
    got = _coerce({"government_warning": {"present": True, "text": "x", "header_bold": True,
                                          "header_bold_confidence": "high",
                                          "header_bold_basis": "thicker strokes"}})["government_warning"]
    assert got["header_bold_confidence"] == "high"
    assert got["header_bold_basis"] == "thicker strokes"


def test_coerce_government_warning_body_bold_defaults_and_values():
    # the body_bold keys are always present, defaulting to null / low
    gw = _coerce({})["government_warning"]
    assert gw["body_bold"] is None
    assert gw["body_bold_confidence"] == "low"
    # explicit values pass through; a bogus confidence is normalized to "low"
    got = _coerce({"government_warning": {"present": True, "text": "x",
                                          "body_bold": False, "body_bold_confidence": "high"}})["government_warning"]
    assert got["body_bold"] is False and got["body_bold_confidence"] == "high"
    bad = _coerce({"government_warning": {"body_bold": "nope", "body_bold_confidence": "bogus"}})["government_warning"]
    assert bad["body_bold"] is None and bad["body_bold_confidence"] == "low"


def test_coerce_includes_appellation_and_vintage():
    out = _coerce({})
    assert out["appellation"] == {"present": False, "value": None, "confidence": "low"}
    assert out["vintage"]["present"] is False


def test_coerce_includes_new_conditional_fields_defaults():
    # fanciful_name / statement_of_composition / sulfite_declaration default to present=False/None/low
    out = _coerce({})
    for f in ("fanciful_name", "statement_of_composition", "sulfite_declaration"):
        assert out[f] == {"present": False, "value": None, "confidence": "low"}, f


def test_coerce_new_conditional_fields_pass_through_values():
    out = _coerce({
        "fanciful_name": {"present": True, "value": "Stormchaser White", "confidence": "high"},
        "statement_of_composition": {"present": True, "value": "Rum with natural flavors added",
                                     "confidence": "high"},
        "sulfite_declaration": {"present": True, "value": "CONTAINS SULFITES", "confidence": "medium"},
    })
    assert out["fanciful_name"]["value"] == "Stormchaser White"
    assert out["statement_of_composition"]["present"] is True
    assert out["sulfite_declaration"]["value"] == "CONTAINS SULFITES"
    assert out["sulfite_declaration"]["confidence"] == "medium"
    # a bogus confidence is normalized to "low" (defensive coercion still applies)
    bad = _coerce({"fanciful_name": {"present": True, "value": "X", "confidence": "bogus"}})
    assert bad["fanciful_name"]["confidence"] == "low"


def test_coerce_unhashable_confidence_normalizes_to_low():
    # a non-hashable confidence value (list/dict) must normalize to "low", not raise
    # TypeError from the set-membership test — reachable on the json_object fallback path
    out = _coerce({
        "brand_name": {"present": True, "value": "X", "confidence": []},
        "alcohol_content": {"present": True, "value": "40% ABV", "confidence": {}},
        "government_warning": {"present": True, "text": "x", "confidence": [],
                               "header_bold_confidence": {}, "body_bold_confidence": ["high"]},
        "additional_statements": [{"value": "AGED 4 YEARS", "confidence": {}}],
    })
    assert out["brand_name"]["confidence"] == "low"
    assert out["alcohol_content"]["confidence"] == "low"
    gw = out["government_warning"]
    assert gw["confidence"] == "low"
    assert gw["header_bold_confidence"] == "low"
    assert gw["body_bold_confidence"] == "low"
    assert out["additional_statements"][0]["confidence"] == "low"


def test_coerce_normalizes_string_additional_statements():
    out = _coerce({"additional_statements": ["CONTAINS SULFITES", {"value": "AGED 4 YEARS"}]})
    assert out["additional_statements"][0]["value"] == "CONTAINS SULFITES"
    assert out["additional_statements"][1]["value"] == "AGED 4 YEARS"


# --- proof / ABV parsing (regression guards from the review) -----------------

def test_abv_proof_only_label_converts_to_abv():
    # proof-only label (no % printed): abv_percent null, proof 90 -> 45% ABV
    ac = field("90 Proof", present=True, abv_percent=None, proof=90.0)
    assert _check_abv(ac, "45%", "spirits", field("Bourbon")).status == PASS


def test_abv_application_entered_as_proof_converts():
    ac = field("45% Alc./Vol.", abv_percent=45.0, proof=None)
    assert _check_abv(ac, "90 proof", "spirits", field("Bourbon")).status == PASS


def test_parse_abv_percent_wins_over_proof():
    assert _parse_abv("45% Alc./Vol. (90 Proof)") == 45.0


def test_parse_abv_proof_converts_to_abv():
    assert _parse_abv("90 Proof") == 45.0


def test_parse_abv_bare_number_is_abv():
    assert _parse_abv("45") == 45.0


def test_parse_abv_ignores_stray_embedded_numbers():
    assert _parse_abv("Bottled in 2021") is None


# --- additional branch coverage flagged by the review ------------------------

def test_abv_missing_on_unknown_type_is_review():
    ac = field(None, present=False, abv_percent=None, proof=None)
    assert _check_abv(ac, None, "unknown", field("Mystery")).status == REVIEW


def test_country_import_wrong_country_fails():
    assert _check_country(field("PRODUCT OF SCOTLAND"), "France").status == FAIL


def test_country_label_present_app_blank_is_review():
    assert _check_country(field("PRODUCT OF FRANCE"), "").status == REVIEW


def test_text_fuzzy_review_band():
    # score lands between FUZZY_REVIEW_FLOOR (85) and FUZZY_PASS (95)
    assert _check_text("brand_name", field("Old Tomm"), "Old Tom").status == REVIEW


def test_abv_low_confidence_escalates():
    ac = field("45% Alc./Vol.", abv_percent=45.0, proof=None, confidence="low")
    assert _check_abv(ac, "45%", "spirits", field("Bourbon")).status == REVIEW


def test_net_contents_low_confidence_escalates():
    assert _check_net_contents(field("750 mL", confidence="low"), "750 mL").status == REVIEW


def test_verify_label_only_omits_country_and_has_six_fields():
    field_names = [f.field for f in verify_label_only(_good_spirits_extract())["fields"]]
    assert "country_of_origin" not in field_names
    assert set(field_names) == {
        "brand_name", "class_type", "alcohol_content", "net_contents",
        "name_and_address", "government_warning",
    }


# --- multi-image extraction input (front + back of one product) --------------

def _image_blocks(content):
    return [b for b in content if b["type"] == "image_url"]


def test_build_content_single_bytes_back_compat():
    content = _build_content(b"frontbytes", "image/png")
    assert content[0]["type"] == "text"
    assert len(_image_blocks(content)) == 1


def test_build_content_multiple_images_front_and_back():
    content = _build_content([(b"front", "image/png"), (b"back", "image/jpeg")])
    blocks = _image_blocks(content)
    assert len(blocks) == 2
    assert blocks[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_build_content_list_of_bare_bytes():
    content = _build_content([b"front", b"back"])
    assert len(_image_blocks(content)) == 2


def test_model_params_gpt4_family_uses_temperature():
    p = _model_params("gpt-4o-mini")
    assert p["temperature"] == 0
    assert "max_completion_tokens" in p and "max_tokens" not in p
    assert "reasoning_effort" not in p


def test_model_params_reasoning_model_omits_temperature():
    # gpt-5 / o-series reject a non-default temperature and want minimal reasoning
    p = _model_params("gpt-5-nano")
    assert "temperature" not in p
    assert p["reasoning_effort"] == "minimal"
    assert "max_completion_tokens" in p


def test_model_params_gpt54_55_use_low_reasoning():
    # the deployed default (gpt-5.4-mini) and the ceiling (gpt-5.5) REJECT "minimal" — their
    # floor is "low". This pins the production routing the generic gpt-5 test above misses.
    for model in ("gpt-5.4-mini", "gpt-5.4", "gpt-5.5"):
        p = _model_params(model)
        assert p["reasoning_effort"] == "low", model
        assert "temperature" not in p
        assert "max_completion_tokens" in p


def test_model_params_o_series_uses_low_reasoning():
    # the o-series also rejects "minimal" and uses "low"
    p = _model_params("o4-mini")
    assert p["reasoning_effort"] == "low"
    assert "temperature" not in p


def test_model_params_uses_structured_outputs():
    # response shape is enforced by the API via strict Structured Outputs
    rf = _model_params("gpt-4o")["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    gw_required = rf["json_schema"]["schema"]["properties"]["government_warning"]["required"]
    for key in ("header_bold", "header_bold_confidence", "header_bold_basis",
                "body_bold", "body_bold_confidence"):
        assert key in gw_required


# --- fixes from the team review ----------------------------------------------

def test_warning_low_confidence_escalates_to_review():
    # the warning is the hard gate, so an otherwise-passing low-confidence read -> review
    assert _check_warning(warning(confidence="low")).status == REVIEW


def test_unreadable_field_is_review_not_fail():
    # present-but-unreadable (or low-confidence blank) -> review, not a hard FAIL
    assert _check_text("brand_name", field(None, present=True, confidence="low"), "Old Tom").status == REVIEW


def test_absent_field_still_fails():
    # genuinely absent (present=false, high confidence) -> FAIL
    assert _check_text("brand_name", field(None, present=False, confidence="high"), "Old Tom").status == FAIL


def test_presence_unreadable_is_review():
    assert _check_presence("net_contents", field(None, present=True, confidence="low")).status == REVIEW


def test_country_unreadable_is_review():
    assert _check_country(field(None, present=True, confidence="low"), "Scotland").status == REVIEW


def test_beverage_type_synonyms_normalized():
    assert _coerce({"beverage_type": "Spirits"})["beverage_type"] == "spirits"
    assert _coerce({"beverage_type": "Distilled Spirits"})["beverage_type"] == "spirits"
    assert _coerce({"beverage_type": "malt"})["beverage_type"] == "beer"
    assert _coerce({"beverage_type": "WINE"})["beverage_type"] == "wine"
    assert _coerce({"beverage_type": "soda"})["beverage_type"] == "unknown"


def test_beverage_type_synonym_preserves_spirits_abv_fail():
    # "Spirits" must normalize to spirits so a missing ABV still FAILs (not a soft review)
    out = _coerce({"beverage_type": "Spirits", "class_type": {"present": True, "value": "Bourbon"}})
    r = _check_abv(out["alcohol_content"], None, out["beverage_type"], out["class_type"])
    assert r.status == FAIL


def test_abv_range_on_label_routes_to_review():
    ac = field("5-6% ALC/VOL", present=True, abv_percent=None, proof=None)
    assert _check_abv(ac, "5%", "beer", field("Ale")).status == REVIEW


def test_abv_range_in_application_routes_to_review():
    ac = field("5.4% ALC/VOL", abv_percent=5.4, proof=None)
    assert _check_abv(ac, "5% to 6%", "beer", field("Ale")).status == REVIEW


def _fake_response(content, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason, message=SimpleNamespace(content=content))])


def test_parse_response_valid_coerces():
    assert _parse_response(_fake_response('{"beverage_type": "wine"}'))["beverage_type"] == "wine"


def test_parse_response_truncated_raises():
    with pytest.raises(ExtractionError):
        _parse_response(_fake_response('{"beverage_type": "wi', finish_reason="length"))


def test_parse_response_empty_raises():
    with pytest.raises(ExtractionError):
        _parse_response(_fake_response(None))


def test_parse_response_invalid_json_raises():
    with pytest.raises(ExtractionError):
        _parse_response(_fake_response('not json at all'))


def test_check_warning_cause_codes():
    # the machine-readable cause is what the merge layer branches on — pin the mapping
    assert _check_warning(warning(text=None, present=False)).cause == "absence"
    assert _check_warning(warning(text="GOVERNMENT WARNING: drink responsibly")).cause == "wording"
    near = GOVERNMENT_WARNING.replace("birth defects", "birth defect")
    assert _check_warning(warning(text=near)).cause == "wording"
    title = GOVERNMENT_WARNING.replace("GOVERNMENT WARNING:", "Government Warning:")
    assert _check_warning(warning(text=title, caps=True)).cause == "caps"
    assert _check_warning(warning(bold=True, bold_confidence="medium")).cause == "bold"   # pass under
    #   the medium_pass_gate default (was review under header_body_gate); cause is "bold" either way
    assert _check_warning(warning(bold=False, bold_confidence="high")).cause == "bold"    # fail
    assert _check_warning(warning()).cause == "bold"                                      # pass
    assert _check_warning(warning(confidence="low")).cause == "low_confidence"            # escalated


def test_bold_prompt_is_deprimed_variant_a_style():
    # Regression guard for the benchmark-validated variant-A wording (BENCHMARK_NOTES.md,
    # dev-archive branch):
    # the 'darker' shortcut was the measured cause of all-bold-body false-passes, and the
    # de-prime clause is what cut high-confidence false-passes 14 -> 0. The bold-instruction
    # block in the extraction prompt must keep these properties.
    prompt = extraction._PROMPT
    assert "darker" not in prompt.lower()
    assert "Do NOT infer bold from capitalization" in prompt
    assert "do not assume bold just because warning headers are usually bold" in prompt


# --- appellation embedded in the class/type designation ----------------------

def test_appellation_embedded_in_designation_passes():
    # "American Moscato" -> "American" is the appellation even if not separately extracted
    r = _check_appellation("wine", field("American Moscato"), field(None, present=False),
                           field(None, present=False))
    assert r.status == PASS and "American" in r.reason
    # "California Red Wine" likewise
    r2 = _check_appellation("wine", field("California Red Wine"), field("2021"),
                            field(None, present=False))
    assert r2.status == PASS and "California" in r2.reason


def test_appellation_no_embedded_term_still_fails():
    # a bare varietal with no appellation anywhere still fails
    r = _check_appellation("wine", field("Chardonnay"), field(None, present=False),
                           field(None, present=False))
    assert r.status == FAIL


# --- image-quality-aware reframing -------------------------------------------

def _spirits_with(name_obj, notes):
    return {
        "beverage_type": "spirits",
        "brand_name": field("X"), "class_type": field("Bourbon"),
        "alcohol_content": field("45% Alc./Vol.", abv_percent=45.0, proof=None),
        "net_contents": field("750 mL"), "name_and_address": name_obj,
        "government_warning": warning(), "additional_statements": [], "image_quality_notes": notes,
    }


def test_reframe_rewords_missing_field_on_low_quality_image():
    extract = _spirits_with(field(None, present=False), "Small text is soft and glared.")
    na = next(f for f in verify_label_only(extract)["fields"] if f.field == "name_and_address")
    assert na.status == FAIL                                    # verdict unchanged
    assert "could not verify required label information" in na.reason


def test_reframe_keeps_specific_reason_on_clean_image():
    extract = _spirits_with(field(None, present=False), None)   # no quality note
    na = next(f for f in verify_label_only(extract)["fields"] if f.field == "name_and_address")
    assert na.status == FAIL
    assert "not found on the label" in na.reason                # not reframed


def test_reframe_does_not_touch_caps_violation():
    # a title-case header is a DEFINITE, legible violation -> never reframed, even on a soft photo
    title = GOVERNMENT_WARNING.replace("GOVERNMENT WARNING:", "Government Warning:")
    extract = _spirits_with(field("Old Tom Distillery, KY"), "glare and blur on the back label")
    extract["government_warning"] = warning(text=title, caps=False)
    gw = next(f for f in verify_label_only(extract)["fields"] if f.field == "government_warning")
    assert gw.status == FAIL and "capital letters" in gw.reason


def test_reframe_does_not_fire_on_adversarial_clean_bold_fail():
    # source-of-truth guard: a not-bold warning on a CLEAN image keeps the genuine bold message
    extract = _spirits_with(field("Old Tom Distillery, KY"), None)
    extract["government_warning"] = warning(bold=False, bold_confidence="high")
    gw = next(f for f in verify_label_only(extract)["fields"] if f.field == "government_warning")
    assert gw.status == FAIL and "bold" in gw.reason            # genuine finding preserved


def test_reframe_does_not_touch_header_not_bold_on_low_quality_image():
    # review #1: a HIGH-confidence header-not-bold FAIL is a definite finding, not a readability
    # problem -- it must keep its specific bold message even on a soft/glared photo (the reframer
    # previously overwrote it with the generic "submit a clearer image" text).
    extract = _spirits_with(field("Old Tom Distillery, KY"), "glare and blur on the back label")
    extract["government_warning"] = warning(bold=False, bold_confidence="high")
    gw = next(f for f in verify_label_only(extract)["fields"] if f.field == "government_warning")
    assert gw.status == FAIL
    assert "bold" in gw.reason.lower()
    assert "could not verify required label information" not in gw.reason


def test_reframe_does_not_touch_body_bold_violation_on_low_quality_image():
    # review #1: a HIGH-confidence body-bold violation (27 CFR 16.22 "remainder may not be bold") is
    # a definite finding -> never masked into the generic photo message, even on a flagged image.
    extract = _spirits_with(field("Old Tom Distillery, KY"), "soft, low-resolution back label")
    extract["government_warning"] = warning(bold=True, bold_confidence="high",
                                            body_bold=True, body_bold_confidence="high")
    gw = next(f for f in verify_label_only(extract)["fields"] if f.field == "government_warning")
    assert gw.status == FAIL
    assert "body" in gw.reason.lower()
    assert "could not verify required label information" not in gw.reason


def test_reframe_still_rewords_unverifiable_bold_on_low_quality_image():
    # the flip side of review #1: a genuinely UNVERIFIABLE bold read (uncertain -> REVIEW) on a
    # low-quality image IS still reframed to the photo message -- only confident violations are kept.
    # (bold_confidence="low": under the medium_pass_gate default a MEDIUM-confidence bold-header
    # read now PASSES, so "low" is what keeps this read genuinely unverifiable.)
    extract = _spirits_with(field("Old Tom Distillery, KY"), "glare obscures the small print")
    extract["government_warning"] = warning(bold=True, bold_confidence="low")
    gw = next(f for f in verify_label_only(extract)["fields"] if f.field == "government_warning")
    assert gw.status == REVIEW
    assert "could not verify required label information" in gw.reason
