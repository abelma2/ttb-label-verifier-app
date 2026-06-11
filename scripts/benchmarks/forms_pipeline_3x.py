"""Full production pipeline vs the APPLICATION FORMS, 3x, on the clean baseline + clearer_baseline
products -- with the triple-gate government-warning verdict shown alongside the production one.

For each product (front+back read together) x3 reps:
  1. PRODUCTION pipeline (read-only): extraction.extract_fields([front, back]) -> verification.verify
     (extracted, application)  -> per-field PASS/REVIEW/FAIL vs the matched form + overall verdict.
  2. TRIPLE GATE on the back (identical TG._run_reads / TG.decide): main gpt-5.4-mini:A bold read +
     2x gpt-4.1+S -> an alternative government-warning verdict, shown beside the production warning
     verdict (so you can see what the triple gate would do inside the real verification).

The forms were transcribed FROM these labels and the labels are COMPLIANT, so EVERY applicable field
should PASS: a FAIL = false-fail, a needs_review = over-caution. Tracks per-field stability across the
3 reps, a per-folder (baseline vs clearer) split, and latency (the full front+back extract is the slow
part, ~6-9s per eval notes; the triple-gate reads run in parallel and are much faster).

Mapping (from each form's _meta): rum->_1, malt->_2, wine->_3 (baseline_* and clear_baseline_*).
No production code is modified; extraction/verification are imported and called read-only.

Usage: python scripts/benchmarks/forms_pipeline_3x.py
Writes output/forms_pipeline_3x_<ts>.{txt,json}.
"""
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT
OUT_DIR = os.path.join(ROOT, "output")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import bold_prompt_safety as B          # loads OPENAI_API_KEY from secrets at import
import triple_gate_compliant as TG      # identical triple gate: _run_reads / decide
from extraction import extract_fields    # read-only
from verification import verify          # read-only

APPS = os.path.join(ROOT, "test_labels", "applications")
REPS = 3
# folder, app, front, back, stem
PRODUCTS = [
    ("baseline_labels", "rum.json", "baseline_1_Front.png", "baseline_1_Other.png", "baseline_1"),
    ("baseline_labels", "malt.json", "baseline_2_Front.png", "baseline_2_Other.png", "baseline_2"),
    ("baseline_labels", "wine.json", "baseline_3_Front.png", "baseline_3_Other.png", "baseline_3"),
    ("clearer_baseline_labels", "rum.json", "clear_baseline_1_Front.png", "clear_baseline_1_Other.png", "clear_baseline_1"),
    ("clearer_baseline_labels", "malt.json", "clear_baseline_2_Front.png", "clear_baseline_2_Other.png", "clear_baseline_2"),
    ("clearer_baseline_labels", "wine.json", "clear_baseline_3_Front.png", "clear_baseline_3_Other.png", "clear_baseline_3"),
]


def _media(p):
    return "image/jpeg" if p.lower().endswith((".jpg", ".jpeg")) else "image/png"


def _imgs(*paths):
    return [(open(p, "rb").read(), _media(p)) for p in paths]


def _app(name):
    d = json.load(open(os.path.join(APPS, name), encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _extract_verify(images, application):
    """Production extract -> verify, with transient-error retry (like eval/run_eval)."""
    last = None
    for k in range(3):
        try:
            t = time.perf_counter()
            extracted = extract_fields(images)
            secs = round(time.perf_counter() - t, 2)
            return extracted, verify(extracted, application), secs, None
        except Exception as e:
            last = e
            time.sleep(3 + 4 * k)
    return None, None, 0.0, str(last)[:140]


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def main():
    print(f"PRODUCTION pipeline vs forms + triple-gate warning  |  {len(PRODUCTS)} products x {REPS} reps  "
          f"(model={B.__dict__.get('EXTRACTION_MODEL', 'gpt-5.4-mini default')})\n")
    apps = {p[1]: _app(p[1]) for p in PRODUCTS}

    records = []
    extract_lat, walls, bold_walls = [], [], []
    field_status = defaultdict(lambda: defaultdict(int))     # field -> status -> n
    by_folder = defaultdict(lambda: {"product_runs": 0, "all_pass": 0})
    per_field_reps = defaultdict(list)                        # (stem, field) -> [status...]
    warn_agree = warn_disagree = 0

    for folder, app, front, back, stem in PRODUCTS:
        front_p = os.path.join(ROOT, "test_labels", folder, front)
        back_p = os.path.join(ROOT, "test_labels", folder, back)
        for rep in range(1, REPS + 1):
            images = _imgs(front_p, back_p)
            extracted, result, esecs, eerr = _extract_verify(images, apps[app])
            # triple gate on the back (warning lives on the back)
            reads = TG._run_reads(back_p)
            m_f, m_dt, _me = reads["main"]
            s1_f, s1_dt, _s1 = reads["specialist_1"]
            s2_f, s2_dt, _s2 = reads["specialist_2"]
            bdts = [d for d in (m_dt, s1_dt, s2_dt) if d is not None]
            bold_wall = max(bdts) if bdts else None
            triple_verdict, triple_reasons = TG.decide(m_f, s1_f, s2_f)
            wall = max([x for x in (esecs, bold_wall) if x is not None], default=None)
            if esecs:
                extract_lat.append(esecs)
            if bold_wall is not None:
                bold_walls.append(bold_wall)
            if wall is not None:
                walls.append(wall)

            if not result:
                print(f"  {stem:16s} r{rep}  EXTRACTION ERROR: {eerr}")
                records.append({"folder": folder, "stem": stem, "app": app, "rep": rep,
                                "overall": "ERROR", "error": eerr, "extract_secs": esecs,
                                "triple_warning": triple_verdict})
                by_folder[folder]["product_runs"] += 1
                continue

            fields = {f.field: f.status for f in result["fields"]}
            reasons = {f.field: f.reason for f in result["fields"] if f.status != "pass"}
            for fn, st in fields.items():
                field_status[fn][st] += 1
                per_field_reps[(stem, fn)].append(st)
            overall = result["overall"]
            prod_warn = fields.get("government_warning", "n/a")
            # production extract's own warning bold read (context)
            gw = (extracted or {}).get("government_warning", {}) if extracted else {}
            prod_gw_bold = (gw.get("header_bold"), gw.get("header_bold_confidence"),
                            gw.get("body_bold"), gw.get("body_bold_confidence"))
            # warning agreement: do prod and triple land the same bucket? (pass vs PASS, etc.)
            norm = {"pass": "PASS", "needs_review": "REVIEW", "fail": "FAIL"}
            if norm.get(prod_warn, prod_warn.upper()) == triple_verdict:
                warn_agree += 1
            else:
                warn_disagree += 1

            by_folder[folder]["product_runs"] += 1
            nonpass = [f"{k}={v}" for k, v in fields.items() if v != "pass"]
            if not nonpass:
                by_folder[folder]["all_pass"] += 1

            bb1 = None if not s1_f else B._eff_body_bold(s1_f)
            bb2 = None if not s2_f else B._eff_body_bold(s2_f)
            wstr = f"{wall:.2f}" if wall is not None else "ERR"
            print(f"  {stem:16s} r{rep}  overall={overall:12s} extract={esecs}s wall={wstr}s  "
                  f"warn[prod={prod_warn} | triple={triple_verdict}]  nonPASS={nonpass or 'none'}")
            records.append({
                "folder": folder, "stem": stem, "app": app, "beverage": result.get("beverage_type"),
                "rep": rep, "overall": overall, "fields": fields, "nonpass_reasons": reasons,
                "prod_warning": prod_warn, "prod_gw_bold": prod_gw_bold,
                "triple_warning": triple_verdict, "triple_reasons": triple_reasons,
                "triple_reads": {"main_bb": (None if not m_f else B._eff_body_bold(m_f)),
                                 "main_hb": (None if not m_f else B._eff_header_bold(m_f)),
                                 "s1_bb": bb1, "s2_bb": bb2},
                "extract_secs": esecs, "bold_wall": bold_wall, "wall": wall,
            })

    # ---- stability: per (product, field), identical across the 3 reps? ----
    unstable = {f"{stem}.{fn}": v for (stem, fn), v in per_field_reps.items()
                if len(v) >= 2 and len(set(v)) > 1}
    n_field_series = len(per_field_reps)
    n_stable = sum(1 for v in per_field_reps.values() if len(v) >= 2 and len(set(v)) == 1)

    report = {
        "products": len(PRODUCTS), "reps": REPS,
        "field_status_counts": {k: dict(v) for k, v in field_status.items()},
        "by_folder": {k: dict(v) for k, v in by_folder.items()},
        "warning_prod_vs_triple": {"agree": warn_agree, "disagree": warn_disagree},
        "stable_field_series": f"{n_stable}/{n_field_series}", "unstable_field_series": unstable,
        "extract_lat": {"avg": round(sum(extract_lat) / len(extract_lat), 2) if extract_lat else None,
                        "p50": _pct(extract_lat, 50), "max": max(extract_lat) if extract_lat else None,
                        "over_5s": sum(1 for x in extract_lat if x > 5)},
        "bold_wall": {"avg": round(sum(bold_walls) / len(bold_walls), 2) if bold_walls else None,
                      "p50": _pct(bold_walls, 50), "max": max(bold_walls) if bold_walls else None},
        "combined_wall": {"avg": round(sum(walls) / len(walls), 2) if walls else None,
                          "p50": _pct(walls, 50), "max": max(walls) if walls else None,
                          "over_5s": sum(1 for x in walls if x > 5)},
        "records": records,
    }
    _summary(report)
    _write(report)


def _summary(r):
    print("\n  === FIELD STATUS across all product-runs (COMPLIANT -> every cell should be PASS) ===")
    for fn, sc in sorted(r["field_status_counts"].items()):
        total = sum(sc.values())
        nonpass = {k: v for k, v in sc.items() if k != "pass"}
        print(f"  {fn:22s} pass {sc.get('pass',0)}/{total}" + (f"   NON-PASS {nonpass}" if nonpass else ""))
    bf = {k: f"{v['all_pass']}/{v['product_runs']}" for k, v in r["by_folder"].items()}
    print(f"\n  per-folder all-fields-PASS product-runs: {bf}")
    print(f"  warning prod-vs-triple: agree {r['warning_prod_vs_triple']['agree']}, "
          f"disagree {r['warning_prod_vs_triple']['disagree']}")
    print(f"  field stability across reps: {r['stable_field_series']}  "
          f"(unstable: {list(r['unstable_field_series'].keys()) or 'none'})")
    el, cw = r["extract_lat"], r["combined_wall"]
    print(f"  latency: production extract avg {el['avg']}s p50 {el['p50']}s max {el['max']}s "
          f"(>5s {el['over_5s']})  |  combined wall p50 {cw['p50']}s max {cw['max']}s (>5s {cw['over_5s']})\n")


def _write(r):
    L = ["", "=" * 104, "PRODUCTION PIPELINE vs APPLICATION FORMS (+ triple-gate warning), 3x", "=" * 104,
         f"{r['products']} products (rum/malt/wine x baseline+clearer) x {r['reps']} reps. Forms transcribed "
         "FROM these compliant labels -> every applicable field should PASS (FAIL = false-fail, "
         "needs_review = over-caution).", ""]
    L.append("FIELD STATUS across all product-runs:")
    for fn, sc in sorted(r["field_status_counts"].items()):
        total = sum(sc.values())
        L.append(f"   {fn:22s} pass {sc.get('pass',0)}/{total}   review {sc.get('needs_review',0)}   "
                 f"fail {sc.get('fail',0)}")
    L.append("")
    bf = {k: f"{v['all_pass']}/{v['product_runs']}" for k, v in r["by_folder"].items()}
    L.append(f"per-folder all-fields-PASS product-runs: {bf}")
    wv = r["warning_prod_vs_triple"]
    L.append(f"government-warning prod-verdict vs triple-gate verdict: agree {wv['agree']}, disagree {wv['disagree']}")
    L.append(f"field stability across the {r['reps']} reps: {r['stable_field_series']} series identical")
    if r["unstable_field_series"]:
        L.append("  UNSTABLE field series (flip across reps):")
        for k, v in r["unstable_field_series"].items():
            L.append(f"     {k:28s} {v}")
    el, bw, cw = r["extract_lat"], r["bold_wall"], r["combined_wall"]
    L.append("")
    L.append(f"latency  production front+back extract: avg {el['avg']}s p50 {el['p50']}s max {el['max']}s "
             f"(>5s {el['over_5s']})")
    L.append(f"         triple-gate bold reads (parallel wall): avg {bw['avg']}s p50 {bw['p50']}s max {bw['max']}s")
    L.append(f"         combined assumed-parallel wall: avg {cw['avg']}s p50 {cw['p50']}s max {cw['max']}s "
             f"(>5s {cw['over_5s']})")
    L.append("")
    L.append("per product-run (warn[prod|triple]; prod_gw_bold = production extract's own header/body bold read):")
    for rec in r["records"]:
        if rec.get("overall") == "ERROR":
            L.append(f"   {rec['stem']:16s} r{rec['rep']}  ERROR: {rec.get('error')}")
            continue
        pg = rec["prod_gw_bold"]
        pgs = f"hb={pg[0]}/{(pg[1] or '-')[:1]} bb={pg[2]}/{(pg[3] or '-')[:1]}"
        tr = rec["triple_reads"]
        nonpass = [f"{k}={v}" for k, v in rec["fields"].items() if v != "pass"]
        L.append(f"   {rec['stem']:16s} r{rec['rep']} overall={rec['overall']:12s} "
                 f"extract={rec['extract_secs']}s wall={rec['wall']}s")
        L.append(f"        warn prod={rec['prod_warning']:12s} (prod_gw_bold {pgs})  |  "
                 f"triple={rec['triple_warning']:7s} (main hb={tr['main_hb']} bb={tr['main_bb']} "
                 f"S1.bb={tr['s1_bb']} S2.bb={tr['s2_bb']})")
        if nonpass:
            L.append(f"        NON-PASS fields: {nonpass}")
            for fn, why in rec["nonpass_reasons"].items():
                L.append(f"           - {fn}: {why[:110]}")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"forms_pipeline_3x_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(report_safe(r), fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written {os.path.relpath(base, ROOT)}.txt / .json")


def report_safe(r):
    return r


if __name__ == "__main__":
    main()
