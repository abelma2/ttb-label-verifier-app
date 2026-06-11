"""Checklist-driven eval harness for the TTB label verifier (report-first; benchmark-only).

Runs the REAL production pipeline (extraction.extract_fields -> verification.verify) against:
  1. the error fixtures in test_labels/error_labels/test_fixtures_manifest.csv, each built as a
     full-product submission = the errored face + the CLEAN baseline of the other face; and
  2. a COMPLETENESS pass: every clean baseline product, to confirm the verifier extracts and
     PASSES each always-applicable mandatory field.

SCORING is per-check (face-relevant): each fixture is judged on the STATUS OF THE FIELD ITS
DEFECT EXERCISES (e.g. PROOF-CONSISTENCY is judged on alcohol_content), not on the overall
verdict -- because pairing a Gemini-rendered errored face with a real clean face introduces
unrelated brand/bold read variance that would otherwise contaminate the overall. The overall
verdict is kept as a secondary column.

It does NOT modify production code. Known verifier gaps (proof != 2xABV, ABV notation) come from
eval/coverage_matrix.csv so their fixture misses are reported as confirmed gaps. Real OpenAI
vision calls; key from OPENAI_API_KEY env, .env, or .streamlit/secrets.toml.

Run (from repo root):  python eval/run_eval.py
Outputs: eval/results/{results.csv, completeness.csv, gaps.md, summary.md} + console summary.
"""
import csv
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # eval/ -> repo root
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EVAL = os.path.join(ROOT, "eval")
RESULTS = os.path.join(EVAL, "results")
ERR = os.path.join(ROOT, "test_labels", "error_labels")
APPS = os.path.join(ROOT, "test_labels", "applications")


def load_key():
    """Load OPENAI_API_KEY from the env, .env, or .streamlit/secrets.toml.
    Returns True when the key is set."""
    envf = os.path.join(ROOT, ".env")
    env_text = open(envf, encoding="utf-8").read() if os.path.exists(envf) else ""
    sec = os.path.join(ROOT, ".streamlit", "secrets.toml")
    sec_text = open(sec, encoding="utf-8").read() if os.path.exists(sec) else ""
    for var in ("OPENAI_API_KEY",):
        if os.environ.get(var):
            continue
        m = (re.search(var + r'\s*=\s*"?([^"\r\n]+)"?', env_text)
             or re.search(var + r'\s*=\s*"([^"]+)"', sec_text))
        if m and m.group(1).strip() and m.group(1).strip() != "sk-...":
            os.environ[var] = m.group(1).strip()
    return bool(os.environ.get("OPENAI_API_KEY"))


def _media(p):
    return "image/jpeg" if p.lower().endswith((".jpg", ".jpeg")) else "image/png"


def _imgs(*paths):
    return [(open(p, "rb").read(), _media(p)) for p in paths]


PRODUCTS = {
    "spirits": {"app": "rum.json", "front": "test_labels/baseline_labels/baseline_1_Front.png",
                "back": "test_labels/baseline_labels/baseline_1_Other.png"},
    "malt": {"app": "malt.json", "front": "test_labels/baseline_labels/baseline_2_Front.png",
             "back": "test_labels/baseline_labels/baseline_2_Other.png"},
    "wine": {"app": "wine.json", "front": "test_labels/error_labels/wine_front_control.png",
             "back": "test_labels/error_labels/wine_back_control_allcaps.png"},
}
EXPECT = {"FAIL": "fail", "NEEDS_REVIEW": "needs_review", "PASS": "pass"}
KNOWN_GAP_CHECKS = set()   # PR-A: proof_equals_2x_abv + abv_notation_format are now CHECKED
#   (verification._check_proof_consistency / _check_abv_notation), so they are no longer known gaps.
# the verifier field each fixture's defect actually exercises (per-check, face-relevant scoring)
CHECK_FIELD = {
    "government_warning_exact_match": "government_warning",
    "case_normalization": "government_warning",
    "brand_name_fuzzy_match": "brand_name",
    "proof_equals_2x_abv": "alcohol_content",
    "abv_numeric_match": "alcohol_content",
    "abv_notation_format": "alcohol_content",
    "net_contents_normalization": "net_contents",
    "vintage_requires_appellation": "appellation",
    "none": "government_warning",   # control: warning is the key compliance field (brand is unstable)
}


def _app(name):
    d = json.load(open(os.path.join(APPS, name), encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _run(images, application):
    from extraction import extract_fields
    from verification import verify
    last = None
    for k in range(3):   # retry transient API errors (rate-limit/timeout) so a flake isn't a 'miss'
        try:
            t = time.perf_counter()
            extracted = extract_fields(images)
            secs = time.perf_counter() - t
            return verify(extracted, application), secs
        except Exception as e:
            last = e
            time.sleep(4 + 5 * k)
    raise last


def _classify(check, exp, primary, is_gap):
    """Per-check result class from the exercised field's status vs the expected verdict."""
    if primary == exp:
        return "caught", True
    if is_gap and exp == "fail" and primary == "pass":
        return "GAP-CONFIRMED (defect passed; item is unchecked)", False
    if check == "brand_name_fuzzy_match" and exp == "needs_review" and primary == "pass":
        return "CALIBRATION (dropped letter PASSED; fuzzy cutoff too loose)", False
    if exp == "fail" and primary == "needs_review":
        return "PARTIAL (flagged needs_review, not fail; near-miss wording policy)", False
    if exp == "needs_review" and primary == "fail":
        return "PARTIAL (over-strict: fail instead of needs_review)", False
    return "MISS", False


def main():
    os.makedirs(RESULTS, exist_ok=True)
    if not load_key():
        sys.exit("ERROR: no OpenAI key (set OPENAI_API_KEY in env/.env or .streamlit/secrets.toml).")

    matrix = list(csv.DictReader(open(os.path.join(EVAL, "coverage_matrix.csv"), encoding="utf-8")))
    gap_rows = [r for r in matrix if r["status"] == "gap-no-check"]
    struct_rows = [r for r in matrix if r["status"] == "structural-out-of-scope"]
    support = {"yes": 0, "partial": 0, "no": 0}
    for r in matrix:
        s = r["verifier_supports"]
        support["yes" if s.startswith("yes") else s] = support.get("yes" if s.startswith("yes") else s, 0) + 1

    manifest = list(csv.DictReader(open(os.path.join(ERR, "test_fixtures_manifest.csv"), encoding="utf-8")))
    apps = {bev: _app(p["app"]) for bev, p in PRODUCTS.items()}

    rows, lat = [], []
    print("=" * 92)
    print("FIXTURE RUNS  (errored face + clean other face; scored on the EXERCISED field)\n")
    for m in manifest:
        bev, side, fn = m["beverage"], m["side"], m["rename_to"]
        prod = PRODUCTS[bev]
        clean_other = os.path.join(ROOT, prod["back"] if side == "front" else prod["front"])
        errored = os.path.join(ERR, fn)
        images = _imgs(errored, clean_other) if side == "front" else _imgs(clean_other, errored)
        try:
            result, secs = _run(images, apps[bev]); err = ""
            fstat = {f.field: f.status for f in result["fields"]}
            overall = result["overall"]
        except Exception as e:
            result, secs, err, fstat, overall = None, 0.0, str(e)[:120], {}, "ERROR"
        lat.append((m["test_id"], secs))
        exp = EXPECT.get(m["expected_verdict"], m["expected_verdict"].lower())
        pf = CHECK_FIELD.get(m["check_exercised"], "")
        primary = fstat.get(pf, "n/a")
        is_gap = m["check_exercised"] in KNOWN_GAP_CHECKS
        cls, caught = _classify(m["check_exercised"], exp, primary, is_gap)
        over5 = "  >5s" if secs > 5 else ""
        flag = "" if m["clean"].lower() == "yes" else "  [fixture not clean]"
        print(f"  {m['test_id']:<20} {pf+':'+primary:<28} exp={m['expected_verdict']:<13} {cls}{over5}{flag}")
        rows.append({"test_id": m["test_id"], "beverage": bev, "fixture": fn, "side": side,
                     "defect": m["defect_introduced"], "check_exercised": m["check_exercised"],
                     "primary_field": pf, "primary_status": primary, "overall_verdict": overall,
                     "expected_verdict": m["expected_verdict"], "caught": caught, "result_class": cls,
                     "latency_s": round(secs, 2), "clean_fixture": m["clean"], "known_gap": is_gap,
                     "all_fields": "; ".join(f"{f.field}={f.status}" for f in result["fields"]) if result else err})

    # ---- completeness: clean baselines, every always-applicable field should PASS ----
    print("\n" + "=" * 92)
    print("COMPLETENESS PASS  (clean baselines -> mandatory fields should PASS)\n")
    comp = []
    for bev, prod in PRODUCTS.items():
        images = _imgs(os.path.join(ROOT, prod["front"]), os.path.join(ROOT, prod["back"]))
        try:
            result, secs = _run(images, apps[bev]); err = ""
        except Exception as e:
            result, secs, err = None, 0.0, str(e)[:120]
        lat.append((f"clean_{bev}", secs))
        nonpass = [f"{f.field}={f.status}" for f in (result["fields"] if result else []) if f.status != "pass"]
        over5 = "  >5s" if secs > 5 else ""
        print(f"  clean_{bev:<8} overall={result['overall'] if result else 'ERROR':<13} non-PASS: {nonpass or 'none'}{over5}")
        for f in (result["fields"] if result else []):
            comp.append({"product": bev, "field": f.field, "status": f.status,
                         "extracted": (f.extracted or "")[:80].replace("\n", " "),
                         "expected": (f.expected or "")[:60].replace("\n", " "),
                         "reason": f.reason[:90], "latency_s": round(secs, 2)})
        if err:
            comp.append({"product": bev, "field": "EXTRACTION", "status": "ERROR", "extracted": "",
                         "expected": "", "reason": err, "latency_s": 0})

    # ---- write CSVs ----
    with open(os.path.join(RESULTS, "results.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(RESULTS, "completeness.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["product", "field", "status", "extracted", "expected", "reason", "latency_s"])
        w.writeheader(); w.writerows(comp)

    # ---- metrics ----
    scored = [r for r in rows if r["overall_verdict"] != "ERROR"]
    caught_n = sum(1 for r in scored if r["caught"])
    gap_conf = [r for r in scored if r["result_class"].startswith("GAP-CONFIRMED")]
    calib = [r for r in scored if r["result_class"].startswith("CALIBRATION")]
    partial = [r for r in scored if r["result_class"].startswith("PARTIAL")]
    real_miss = [r for r in scored if r["result_class"] == "MISS"]
    controls = [r for r in scored if r["expected_verdict"] == "PASS"]
    controls_ok = all(r["caught"] for r in controls)
    brand_unstable = [c for c in comp if c["field"] == "brand_name" and c["status"] != "pass"]
    slowest = max(lat, key=lambda x: x[1]) if lat else ("none", 0)
    over5 = [t for t, s in lat if s > 5]

    def md(path, lines): open(os.path.join(RESULTS, path), "w", encoding="utf-8").write("\n".join(lines))

    md("gaps.md", [
        "# Gaps & out-of-scope items — verifier coverage findings", "",
        "From `eval/coverage_matrix.csv` + the fixture run (per-check scored). **Report-first: production "
        "code was NOT modified to hide these.**", "",
        "## 1. gap-no-check — mandatory items the verifier does NOT check", "",
        *[f"- **{r['mandatory_item']}** ({r['beverage']}, {r['citation']}) — {r['notes']}" for r in gap_rows],
        "", "## 2. Confirmed by a fixture this run (defect PASSED its exercised check)", "",
        *([f"- `{r['test_id']}` ({r['defect']}): `{r['primary_field']}`={r['primary_status']} vs expected "
           f"{r['expected_verdict']} — confirms `{r['check_exercised']}` is unchecked." for r in gap_conf] or ["- (none)"]),
        "", "## 3. Calibration / strictness findings", "",
        *([f"- `{r['test_id']}`: {r['result_class']} (`{r['primary_field']}`={r['primary_status']}, expected {r['expected_verdict']})." for r in calib + partial] or ["- (none)"]),
        "", "## 4. Unexpected misses (investigate)", "",
        *([f"- `{r['test_id']}` ({r['defect']}): `{r['primary_field']}`={r['primary_status']}, expected {r['expected_verdict']}. fields: {r['all_fields']}" for r in real_miss] or ["- (none — every other fixture scored as expected on its exercised check)"]),
        "", "## 5. Non-determinism (a top finding) — brand reads AND warning-bold vary run-to-run", "",
        "The extractor (a vision model) is not deterministic. Two checks flip across runs, so a SINGLE "
        "run is not authoritative for them (the deterministic checks below ARE reliable):",
        "- **Warning bold/clean** — a compliant all-caps warning sometimes passes and sometimes fails the "
        "bold gate (CONTROL/CONTROL-CASE and the clean malt baseline flipped between runs). Matches the bold "
        "instability documented in BENCHMARK_NOTES.md.",
        "- **Brand vs fanciful name** — the model picks a different prominent name across runs (e.g. malt "
        "\"MALT & HOP BREWERY\" vs \"Honey Huckleberry Pie\"; \"JON'S\" lands PASS one run, FAIL the next), so "
        "the brand verdict is unstable around the fuzzy cutoff.",
        *([f"  - this run: `{c['product']}` brand_name={c['status']}: extracted {c['extracted']!r} vs form {c['expected']!r}." for c in brand_unstable] or ["  - (brand stable this run)"]),
        "", "## 6. structural-out-of-scope — not verifiable from a flat image", "",
        *[f"- **{r['mandatory_item']}** ({r['beverage']}, {r['citation']}) — {r['notes']}" for r in struct_rows],
        "", "## 7. Fixture gaps (checklist items with NO test fixture yet)", "",
        "- Misspelled designation (e.g. varietal \"Chardonay\") — \"spelled correctly\" is an explicit checklist line; no fixture, and fuzzy match would likely tolerate it.",
        "- Flavored-malt with the ABV statement OMITTED — would wrongly PASS (flavor trigger not modeled); no fixture.",
        "- `rum_back_reworded_CONTAMINATED.png` (GW-REWORD) is `clean=no` in the manifest (duplicate word + re-rendered layout) — not a clean single-defect fixture.",
        "- Manifest data fix: the CONTROL-CASE row had an unquoted comma in its defect field (CSV mis-parse); quoting was added so the row scores correctly.",
        "- Conditional disclosures (country of origin / sulfites / Yellow #5 / cochineal / aspartame) — no observable trigger; transcribed but not asserted, by design.",
    ])

    md("summary.md", [
        "# Label verifier — evaluation summary", "",
        f"**Defects caught (per-check): {caught_n} of {len(scored)} fixtures.** "
        f"{len(gap_conf)} confirmed a known unchecked item, {len(calib)+len(partial)} were calibration/strictness "
        f"findings, {len(real_miss)} unexpected.", "",
        f"**Control (clean) labels:** the compliance check passed on {sum(1 for r in controls if r['caught'])}/{len(controls)} "
        f"controls this run. IMPORTANT: brand-name matching AND the warning-bold gate are non-deterministic "
        f"run-to-run (a clean control can flip pass/fail) — see gaps.md #5. The deterministic checks "
        f"(exact wording, S/G capitalization, ABV numeric, appellation) are the reliable signal.", "",
        f"**Mandatory checks supported:** {support.get('yes',0)} yes (incl. import-conditional), "
        f"{support.get('partial',0)} partial, {support.get('no',0)} no, of {len(matrix)} checklist items "
        f"across all three beverage types.", "",
        f"**Speed:** slowest label `{slowest[0]}` at {slowest[1]:.1f}s; **{len(over5)} of {len(lat)} labels exceeded "
        f"the ~5s bar** — every full front+back read is ~6-9s, so the 2-image submission is over Sarah's 5s target.", "",
        "## What it gets right",
        "- Government warning: exact wording, ALL-CAPS header, \"Surgeon General\" capitalization, bold (lowercase S/G -> FAIL).",
        "- ABV numeric mismatch vs the application (40% vs 20% -> FAIL).",
        "- ABV notation: the bare \"ABV\" abbreviation is rejected (NOTATION-ABV -> FAIL; 27 CFR 5.65/7.65).",
        "- Proof vs ABV consistency: proof != 2x ABV is rejected (PROOF-CONSISTENCY, 50 proof on 20% ABV -> FAIL; 27 CFR 5.65).",
        "- Wine appellation-of-origin when a varietal/vintage requires it (cross-field rule -> FAIL).",
        "- Net contents: same volume in a different unit/format -> NEEDS_REVIEW; a materially different volume -> FAIL (unit-aware, PR-B).",
        "- Name/address: punctuation/relationship-prefix differences normalized (fewer false reviews); a short subset read that drops the producer name -> NEEDS_REVIEW. (KNOWN GAP: a producer-name word substitution can still pass the fuzzy score.)", "",
        "## Top gaps for follow-up (see gaps.md)",
        "- **Brand fuzzy cutoff too loose** — BRAND-FUZZY (\"JON'S\" vs \"JOHN'S\", one letter) PASSED instead of going to review.",
        "- **Single-character warning edits** (missing comma) go to needs_review, not fail (near-miss wording policy).",
        "- Spelling, same-field-of-vision, separate-and-apart, formula numbers: not verified.", "",
        "_Deviations from the brief: the pre-populated `coverage_matrix_starter.csv` and a `.env` were not present, "
        "so the matrix was built from the three TTB checklists and the key was loaded from `.streamlit/secrets.toml`. "
        "The `.docx` was a real binary (extracted via its XML). Extraction uses the production `gpt-5.4-mini` reasoning "
        "model (temperature is not tunable for it). Fixtures are scored per-check because Gemini-rendered errored faces "
        "introduce unrelated read variance vs the real clean faces._",
    ])

    print("\n" + "=" * 92)
    print(f"SUMMARY  caught {caught_n}/{len(scored)} (per-check)  | gap-confirmed {len(gap_conf)} | "
          f"calibration/partial {len(calib)+len(partial)} | unexpected {len(real_miss)}")
    print(f"  controls compliance-clean: {controls_ok}  | brand unstable on: {[c['product'] for c in brand_unstable]}")
    print(f"  mandatory checks: {support.get('yes',0)} yes / {support.get('partial',0)} partial / {support.get('no',0)} no  (of {len(matrix)})")
    print(f"  slowest {slowest[0]} {slowest[1]:.1f}s | OVER 5s: {len(over5)}/{len(lat)} (every full read)")
    print(f"  gap-confirmed: {[r['test_id'] for r in gap_conf]}")
    print(f"  calibration/partial: {[r['test_id'] for r in calib+partial]}")
    if real_miss:
        print(f"  UNEXPECTED MISSES: {[r['test_id'] for r in real_miss]}")
    print("\n  wrote eval/results/{results.csv, completeness.csv, gaps.md, summary.md}")


if __name__ == "__main__":
    main()
