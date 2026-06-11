"""error_label_application_eval.py -- benchmark-only.

The real target scenario: each INTENTIONALLY-ALTERED printed label (test_labels/error_labels)
verified against its product's APPLICATION JSON (test_labels/applications), to confirm the
model + deterministic verifier CATCHES the difference between the application and the label.

Each error fixture is one altered FACE. It is paired with the matching product's CLEAN other
face (from test_labels/baseline_labels) to form a full front+back submission, then run through
the production pipeline unchanged:  extract_fields(images) -> verify(extracted, application).
Scoring is on the field the defect EXERCISES (face-relevant), with the overall verdict alongside;
pairing an altered face with a real clean face adds unrelated read variance to the *overall*, so
the per-field verdict is the signal.

It also runs the 3 CLEAN baselines vs their applications to surface false-fails, and contrasts:
  - clean labels should be PASS (or NEEDS_REVIEW only for the known bold-format uncertainty);
  - error labels should FAIL/REVIEW on the intentionally changed field.

Does NOT modify production code. Real OpenAI vision calls (key from env / .env / secrets.toml).

Run (from repo root):  python scripts/benchmarks/error_label_application_eval.py
Writes: artifacts/error_label_application_eval_results.{md,json}
"""
import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT

import csv
import json
import os
import re
import sys
import time
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from extraction import extract_fields
from verification import verify
from config import EXTRACTION_MODEL

ERR = os.path.join(ROOT, "test_labels", "error_labels")
APPS = os.path.join(ROOT, "test_labels", "applications")
BASE = os.path.join(ROOT, "test_labels", "baseline_labels")
ARTIFACTS = os.path.join(ROOT, "artifacts")

# A product = the application JSON + the CLEAN front/back baseline faces. An error fixture
# replaces ONE of those faces with its altered version (by `side`); the other stays clean.
PRODUCTS = {
    "spirits": {"app": "rum.json",  "front": os.path.join(BASE, "baseline_1_Front.png"),
                "back": os.path.join(BASE, "baseline_1_Other.png")},
    "malt":    {"app": "malt.json", "front": os.path.join(BASE, "baseline_2_Front.png"),
                "back": os.path.join(BASE, "baseline_2_Other.png")},
    "wine":    {"app": "wine.json", "front": os.path.join(BASE, "baseline_3_Front.png"),
                "back": os.path.join(BASE, "baseline_3_Other.png")},
}

# The verifier field each fixture's defect actually exercises.
CHECK_FIELD = {
    "government_warning_exact_match": "government_warning",
    "case_normalization": "government_warning",
    "brand_name_fuzzy_match": "brand_name",
    "proof_equals_2x_abv": "alcohol_content",
    "abv_numeric_match": "alcohol_content",
    "abv_notation_format": "alcohol_content",
    "net_contents_normalization": "net_contents",
    "vintage_requires_appellation": "appellation",
    "none": "government_warning",   # control: the warning is the key always-applicable field
}
# Application-JSON key holding the expected value for that verifier field.
APP_KEY = {
    "government_warning": "health_warning",
    "brand_name": "brand_name",
    "alcohol_content": "alcohol_content",
    "net_contents": "net_contents",
    "appellation": "appellation",
    "name_and_address": "name_and_address",
}
# Items the verifier is known NOT to check (no dedicated logic) -- a miss here is a documented
# gap, not a regression of this benchmark.
KNOWN_GAP_CHECKS = {"proof_equals_2x_abv", "abv_notation_format"}
EXPECT = {"FAIL": "fail", "NEEDS_REVIEW": "needs_review", "PASS": "pass"}
# Application-matched fields (a clean-baseline FAIL on one of these is a false-fail). The
# government warning is rule-based, not matched to a form value, so it's tracked separately.
MATCHED_FIELDS = {"brand_name", "class_type", "alcohol_content",
                  "net_contents", "name_and_address", "country_of_origin"}


def load_key():
    """OPENAI_API_KEY from env, then .env, then .streamlit/secrets.toml (mirrors eval/run_eval.py)."""
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


def _media(p):
    return "image/jpeg" if p.lower().endswith((".jpg", ".jpeg")) else "image/png"


def _imgs(*paths):
    return [(open(p, "rb").read(), _media(p)) for p in paths]


def _app(name):
    d = json.load(open(os.path.join(APPS, name), encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _run(images, application):
    """extract + verify, retrying transient API errors so a flake isn't scored as a miss."""
    last = None
    for k in range(3):
        try:
            t = time.perf_counter()
            extracted = extract_fields(images)
            secs = time.perf_counter() - t
            return verify(extracted, application), extracted, secs
        except Exception as e:
            last = e
            time.sleep(4 + 5 * k)
    raise last


def _cell(s, n=58):
    """Sanitize a value for a markdown table cell."""
    s = "" if s is None else str(s)
    s = s.replace("\n", " ").replace("|", "/").strip()
    return (s[: n - 1] + "…") if len(s) > n else s


def run_error_fixtures(manifest, apps):
    rows = []
    print("=" * 100)
    print("ERROR LABELS vs APPLICATION  (altered face + clean other face; scored on the EXERCISED field)\n")
    for m in manifest:
        bev, side, fn = m["beverage"], m["side"], m["rename_to"]
        prod = PRODUCTS[bev]
        errored = os.path.join(ERR, fn)
        other = prod["back"] if side == "front" else prod["front"]
        images = _imgs(errored, other) if side == "front" else _imgs(other, errored)
        try:
            result, extracted, secs = _run(images, apps[bev])
            err = ""
            fstat = {f.field: f for f in result["fields"]}
            overall = result["overall"]
        except Exception as e:
            result, extracted, secs, err = None, None, 0.0, str(e)[:140]
            fstat, overall = {}, "ERROR"

        field = CHECK_FIELD.get(m["check_exercised"], "government_warning")
        fr = fstat.get(field)
        field_verdict = fr.status if fr else "n/a"
        extracted_val = fr.extracted if fr else ""
        reason = fr.reason if fr else err
        app_val = apps[bev].get(APP_KEY.get(field, ""), "")
        exp = EXPECT.get(m["expected_verdict"], m["expected_verdict"].lower())
        is_defect = m["expected_verdict"] in ("FAIL", "NEEDS_REVIEW")
        caught = (field_verdict == exp)
        false_pass = bool(is_defect and field_verdict == "pass")
        is_gap = m["check_exercised"] in KNOWN_GAP_CHECKS

        rows.append({
            "test_id": m["test_id"], "beverage": bev,
            "intentional_edit": m["defect_introduced"], "expected_field": field,
            "application_value": app_val, "extracted_value": extracted_val,
            "field_verdict": field_verdict, "reason": reason, "overall_verdict": overall,
            "expected_verdict": m["expected_verdict"], "caught": caught,
            "false_pass": false_pass, "needs_review": field_verdict == "needs_review",
            "known_gap": is_gap, "clean_fixture": m["clean"], "latency_s": round(secs, 2),
        })
        tag = "CAUGHT" if caught else ("FALSE-PASS" if false_pass else "miss/partial")
        gap = "  [known gap]" if is_gap and false_pass else ""
        print(f"  {m['test_id']:<20} {field+':'+field_verdict:<28} exp={m['expected_verdict']:<13} {tag}{gap}")
    return rows


def run_clean_baselines(apps):
    rows = []
    print("\n" + "=" * 100)
    print("CLEAN BASELINES vs APPLICATION  (should PASS on matched fields; warning may REVIEW)\n")
    for bev, prod in PRODUCTS.items():
        images = _imgs(prod["front"], prod["back"])
        try:
            result, extracted, secs = _run(images, apps[bev])
            err = ""
        except Exception as e:
            result, extracted, secs, err = None, None, 0.0, str(e)[:140]
        fields = result["fields"] if result else []
        false_fails = [f.field for f in fields if f.field in MATCHED_FIELDS and f.status == "fail"]
        warn = next((f.status for f in fields if f.field == "government_warning"), "n/a")
        for f in fields:
            rows.append({
                "product": bev, "field": f.field, "status": f.status,
                "extracted": f.extracted, "expected": f.expected, "reason": f.reason,
                "overall_verdict": result["overall"] if result else "ERROR",
                "false_fail": f.field in MATCHED_FIELDS and f.status == "fail",
                "latency_s": round(secs, 2),
            })
        if err:
            rows.append({"product": bev, "field": "EXTRACTION", "status": "ERROR",
                         "extracted": "", "expected": "", "reason": err,
                         "overall_verdict": "ERROR", "false_fail": False, "latency_s": 0.0})
        print(f"  clean_{bev:<8} overall={(result['overall'] if result else 'ERROR'):<13} "
              f"warning={warn:<13} matched-field false-fails: {false_fails or 'none'}")
    return rows


def _md(err_rows, clean_rows, when):
    caught = [r for r in err_rows if r["caught"]]
    false_pass = [r for r in err_rows if r["false_pass"]]
    fp_real = [r for r in false_pass if not r["known_gap"]]
    fp_gap = [r for r in false_pass if r["known_gap"]]
    review = [r for r in err_rows if r["needs_review"]]
    false_fail = [r for r in clean_rows if r.get("false_fail")]
    warn_review = [r for r in clean_rows if r["field"] == "government_warning" and r["status"] != "pass"]
    defects = [r for r in err_rows if r["expected_verdict"] in ("FAIL", "NEEDS_REVIEW")]

    L = []
    L.append("# Error-label vs application evaluation\n")
    L.append(f"_Model `{EXTRACTION_MODEL}` · generated {when} · benchmark-only, no production code changed._\n")
    L.append(f"**Caught {len(caught)}/{len(err_rows)} fixtures** on their exercised field "
             f"(of which {len(defects)} are actual defects). "
             f"**{len(fp_real)} real false-pass**, {len(fp_gap)} false-pass on a known-unchecked item. "
             f"**{len(false_fail)} matched-field false-fail** on clean baselines. "
             f"{len(review)} fixtures landed in needs-review.\n")

    L.append("## Error fixtures — application value vs altered printed label\n")
    hdr = ("| product | intentional edit | exp. field | application value | extracted value | "
           "field verdict | reason | overall | caught? | false-pass? | latency |")
    L.append(hdr)
    L.append("|" + "---|" * 11)
    for r in err_rows:
        L.append("| " + " | ".join([
            _cell(f"{r['test_id']} ({r['beverage']})", 26),
            _cell(r["intentional_edit"], 40),
            _cell(r["expected_field"], 18),
            _cell(r["application_value"], 40),
            _cell(r["extracted_value"], 40),
            _cell(r["field_verdict"], 14),
            _cell(r["reason"], 52),
            _cell(r["overall_verdict"], 13),
            "yes" if r["caught"] else "no",
            ("yes" + (" (known gap)" if r["known_gap"] else "")) if r["false_pass"] else "no",
            f"{r['latency_s']:.1f}s",
        ]) + " |")

    L.append("\n## Caught (intentional edit detected on its field)\n")
    L += [f"- `{r['test_id']}` — {r['intentional_edit']} → `{r['expected_field']}`={r['field_verdict']} "
          f"(expected {r['expected_verdict']})" for r in caught] or ["- (none)"]

    L.append("\n## False-passes (altered label, but the exercised field PASSED)\n")
    if fp_real:
        L += [f"- **`{r['test_id']}`** — {r['intentional_edit']} → `{r['expected_field']}`=pass "
              f"(expected {r['expected_verdict']}). Overall: {r['overall_verdict']}." for r in fp_real]
    else:
        L.append("- (none — every defect the verifier is designed to check was flagged)")
    if fp_gap:
        L.append("\n_Known unchecked items (documented gaps, not regressions):_")
        L += [f"- `{r['test_id']}` — {r['intentional_edit']} → `{r['expected_field']}`=pass "
              f"(no dedicated check: proof-vs-ABV / ABV-notation)." for r in fp_gap]

    L.append("\n## Needs-review (flagged for a human, not auto-passed)\n")
    L += [f"- `{r['test_id']}` — `{r['expected_field']}`=needs_review (expected {r['expected_verdict']}): "
          f"{_cell(r['reason'], 90)}" for r in review] or ["- (none)"]

    L.append("\n## Clean baselines vs application (false-fail check)\n")
    L.append("| product | field | status | extracted | expected | overall | false-fail? |")
    L.append("|" + "---|" * 7)
    for r in clean_rows:
        L.append("| " + " | ".join([
            _cell(r["product"], 10), _cell(r["field"], 18), _cell(r["status"], 13),
            _cell(r["extracted"], 38), _cell(r["expected"], 34), _cell(r["overall_verdict"], 13),
            "YES" if r.get("false_fail") else "no",
        ]) + " |")
    L.append("")
    if false_fail:
        L.append(f"**Matched-field false-fails: {len(false_fail)}** — "
                 + ", ".join(f"{r['product']}.{r['field']}" for r in false_fail))
    else:
        L.append("**Matched-field false-fails: 0** — no clean baseline failed application matching.")
    if warn_review:
        L.append(f"\n_Warning gate non-pass on clean baselines (known bold-format uncertainty, not a "
                 f"matching false-fail): {', '.join(r['product']+':'+r['status'] for r in warn_review)}._")

    L.append("\n## Reading this\n")
    L.append("- **Error labels should FAIL/REVIEW the changed field; clean labels should PASS** "
             "(the warning may REVIEW on bold-format uncertainty — that is by design).")
    L.append("- `PROOF-CONSISTENCY` and `NOTATION-ABV` are **known unchecked items** (proof-vs-ABV and "
             "ABV-notation have no dedicated rule); their passes are documented gaps, not regressions.")
    L.append("- A single altered face paired with a real clean face adds unrelated read variance to the "
             "*overall* verdict, so the **per-field verdict** is the signal, not the rollup.")
    return "\n".join(L)


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    if not load_key():
        sys.exit("ERROR: no OpenAI key (set OPENAI_API_KEY in env / .env / .streamlit/secrets.toml).")

    manifest = list(csv.DictReader(open(os.path.join(ERR, "test_fixtures_manifest.csv"), encoding="utf-8")))
    apps = {bev: _app(p["app"]) for bev, p in PRODUCTS.items()}

    err_rows = run_error_fixtures(manifest, apps)
    clean_rows = run_clean_baselines(apps)

    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = _md(err_rows, clean_rows, when)
    out = {
        "generated": when, "model": EXTRACTION_MODEL,
        "error_fixtures": err_rows, "clean_baselines": clean_rows,
        "summary": {
            "fixtures": len(err_rows),
            "caught": sum(1 for r in err_rows if r["caught"]),
            "false_pass_real": sum(1 for r in err_rows if r["false_pass"] and not r["known_gap"]),
            "false_pass_known_gap": sum(1 for r in err_rows if r["false_pass"] and r["known_gap"]),
            "needs_review": sum(1 for r in err_rows if r["needs_review"]),
            "matched_field_false_fails": sum(1 for r in clean_rows if r.get("false_fail")),
        },
    }
    md_path = os.path.join(ARTIFACTS, "error_label_application_eval_results.md")
    json_path = os.path.join(ARTIFACTS, "error_label_application_eval_results.json")
    open(md_path, "w", encoding="utf-8").write(md)
    json.dump(out, open(json_path, "w", encoding="utf-8"), indent=2, default=str)

    s = out["summary"]
    print("\n" + "=" * 100)
    print(f"SUMMARY  caught {s['caught']}/{s['fixtures']} | real false-pass {s['false_pass_real']} | "
          f"known-gap false-pass {s['false_pass_known_gap']} | needs-review {s['needs_review']} | "
          f"clean false-fails {s['matched_field_false_fails']}")
    print(f"  wrote {os.path.relpath(md_path, ROOT)} and {os.path.relpath(json_path, ROOT)}")


if __name__ == "__main__":
    main()
