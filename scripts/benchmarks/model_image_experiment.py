"""Cross-model TIME + ACCURACY experiment on the clearer_baseline_labels set.

Answers: for each candidate vision model, how FAST is a label read (single-call latency and
batch throughput) and how ACCURATE is the resulting verdict on these clean, compliant labels?

It runs the FULL production pipeline per model -- the same extract_fields() call (via
extraction._build_content / _model_params / _create_with_fallbacks / _parse_response) followed
by the deterministic verify() against each product's matched application -- so the numbers
reflect what the app would actually produce if EXTRACTION_MODEL were swapped.

Structure (mirrors the user's request):
  1. SINGLE phase: each product read 3x, SEQUENTIALLY, per model -> clean per-call latency and
     run-to-run STABILITY (do the 3 repeats agree field-for-field?).
  2. BATCH phase: all 3 products read CONCURRENTLY once (ThreadPoolExecutor, like app.py) ->
     wall-clock throughput.

Ground truth: these are clean, compliant labels paired with their hand-written applications
(test_labels/applications/{rum,malt,wine}.json). So a FAIL on any field is a FALSE FAIL (model
error); needs_review is acceptable (esp. the bold gate, known machine-unstable). The
product->application mapping is FIXED by file (verified by eye), independent of the model's own
beverage_type read -- so a type misread shows up as its own metric, never silently swaps the
application.

Usage (from the project root):
    python scripts/benchmarks/model_image_experiment.py                 # the Broad-6 set
    python scripts/benchmarks/model_image_experiment.py gpt-5.4-mini gpt-5.5   # only these
Writes output/model_image_experiment_<ts>.{txt,json}. Real billed API calls.
"""
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT
OUT_DIR = os.path.join(ROOT, "output")

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console mangles em-dashes otherwise
except Exception:
    pass

from config import BATCH_MAX_WORKERS
from extraction import (_build_content, _model_params, _create_with_fallbacks,
                        _parse_response, _get_client)
from verification import verify, PASS, REVIEW, FAIL


def _ensure_openai_key():
    """The production client reads OPENAI_API_KEY from the env; app.py sets it from
    st.secrets, but a bare script must load it from .streamlit/secrets.toml itself
    (same as the other benchmarks)."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    import re
    secrets = os.path.join(ROOT, ".streamlit", "secrets.toml")
    if os.path.exists(secrets):
        with open(secrets, encoding="utf-8") as fh:
            m = re.search(r'OPENAI_API_KEY\s*=\s*"([^"]+)"', fh.read())
        if m and m.group(1) and not m.group(1).startswith(("sk-...", "...")):
            os.environ["OPENAI_API_KEY"] = m.group(1)


_ensure_openai_key()

# --- models (Broad 6 by default; override on argv) ---------------------------
DEFAULT_MODELS = ["gpt-5.4-mini", "gpt-5.4", "gpt-5.5", "gpt-5-mini", "gpt-5.2", "gpt-4.1"]

# --- products: FIXED file -> application mapping (verified by eye) ------------
IMG_DIR = os.path.join(ROOT, "test_labels", "clearer_baseline_labels")
APP_DIR = os.path.join(ROOT, "test_labels", "applications")
APP_FIELDS = ("brand_name", "class_type", "alcohol_content", "net_contents",
              "name_and_address", "country_of_origin")
REPEATS = 3


def _load_app(name):
    with open(os.path.join(APP_DIR, name), encoding="utf-8") as fh:
        d = json.load(fh)
    return {f: (d.get(f) or "") for f in APP_FIELDS}


PRODUCTS = [
    {"label": "clear_baseline_1", "app": _load_app("rum.json"), "expected_bev": "spirits",
     "images": ["clear_baseline_1_Front.png", "clear_baseline_1_Other.png"]},
    {"label": "clear_baseline_2", "app": _load_app("malt.json"), "expected_bev": "beer",
     "images": ["clear_baseline_2_Front.png", "clear_baseline_2_Other.png"]},
    {"label": "clear_baseline_3", "app": _load_app("wine.json"), "expected_bev": "wine",
     "images": ["clear_baseline_3_Front.png", "clear_baseline_3_Other.png"]},
]
for p in PRODUCTS:
    p["loaded"] = [(open(os.path.join(IMG_DIR, n), "rb").read(), "image/png") for n in p["images"]]


# --- one full-pipeline read for a given model --------------------------------

def _extract_with_model(model, images):
    """Exactly the production extract path, but with an explicit model (the app uses the
    global EXTRACTION_MODEL). Shares the production client (timeout, max_retries=0)."""
    content = _build_content(images, "image/png")
    params = _model_params(model)            # reasoning_effort / temperature per family
    resp = _create_with_fallbacks(_get_client(), content, params)   # SO->json_object fallback
    return _parse_response(resp)


def _read_and_verify(model, product):
    """Time the extraction call, then verify against the product's application. Returns a
    per-run record. Raises on a hard model/parse error (caller records it)."""
    t0 = time.perf_counter()
    extracted = _extract_with_model(model, product["loaded"])
    extract_s = time.perf_counter() - t0
    result = verify(extracted, product["app"])
    statuses = {f.field: f.status for f in result["fields"]}
    fails = sorted(f for f, s in statuses.items() if s == FAIL)
    reviews = sorted(f for f, s in statuses.items() if s == REVIEW)
    gw_field = next((f for f in result["fields"] if f.field == "government_warning"), None)
    gw = extracted.get("government_warning") or {}
    return {
        "extract_seconds": round(extract_s, 2),
        "overall": result["overall"],
        "beverage_type": extracted.get("beverage_type"),
        "beverage_ok": extracted.get("beverage_type") == product["expected_bev"],
        "warning": getattr(gw_field, "status", None),
        # the warning verdict is bold-gate-driven (machine-unreliable by design), so it is
        # tracked SEPARATELY from the clean fields; capture WHY plus the raw bold observations.
        "warning_cause": getattr(gw_field, "cause", None),
        "warning_reason": (getattr(gw_field, "reason", None) or "")[:160],
        "warning_obs": {"header_all_caps": gw.get("header_all_caps"),
                        "header_bold": gw.get("header_bold"),
                        "header_bold_confidence": gw.get("header_bold_confidence"),
                        "body_bold": gw.get("body_bold"),
                        "body_bold_confidence": gw.get("body_bold_confidence")},
        "statuses": statuses,
        "fail_fields": fails,
        # the CLEAN accuracy signal: a FAIL on any NON-warning field is unambiguously wrong
        # (these fields are compliant and matched to the application).
        "nonwarning_fail_fields": [f for f in fails if f != "government_warning"],
        "review_fields": reviews,
    }


# --- accuracy aggregation ----------------------------------------------------

def _stability(runs_by_product):
    """Across the REPEATS single runs of each product, the fraction of (product, field) cells
    whose status was identical on every repeat. 1.0 == fully deterministic."""
    agree = total = 0
    for runs in runs_by_product.values():
        ok_runs = [r for r in runs if "error" not in r]
        if len(ok_runs) < 2:
            continue
        fields = set().union(*(r["statuses"].keys() for r in ok_runs))
        for fld in fields:
            vals = {r["statuses"].get(fld) for r in ok_runs}
            total += 1
            agree += (len(vals) == 1)
    return (agree / total) if total else None


def main():
    models = sys.argv[1:] or DEFAULT_MODELS
    print(f"Models: {', '.join(models)}")
    print(f"Products: {', '.join(p['label'] for p in PRODUCTS)}  (3 repeats single + 1 batch each)\n")

    report = {}   # model -> {...}

    for model in models:
        print(f"=== {model} ===")
        single = {p["label"]: [] for p in PRODUCTS}   # label -> [run,...]
        unavailable_reason = None

        # 1) SINGLE phase: sequential, clean per-call latency + stability
        for p in PRODUCTS:
            for rep in range(1, REPEATS + 1):
                try:
                    rec = _read_and_verify(model, p)
                    single[p["label"]].append(rec)
                    tag = (f"overall={rec['overall']} warn={rec['warning']} "
                           f"bev={'ok' if rec['beverage_ok'] else rec['beverage_type']} "
                           f"{rec['extract_seconds']}s"
                           + (f" FAILS={rec['fail_fields']}" if rec['fail_fields'] else ""))
                    print(f"  single {p['label']} rep{rep}: {tag}")
                except Exception as exc:
                    single[p["label"]].append({"error": str(exc)[:200]})
                    unavailable_reason = unavailable_reason or str(exc)[:200]
                    print(f"  single {p['label']} rep{rep}: ERROR {str(exc)[:120]}")

        ok_runs = [r for runs in single.values() for r in runs if "error" not in r]
        if not ok_runs:
            report[model] = {"available": False, "reason": unavailable_reason}
            print(f"  -> UNAVAILABLE: {unavailable_reason}\n")
            continue

        # 2) BATCH phase: all products concurrently, once (throughput)
        batch_runs, batch_total = {}, None
        t0 = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=BATCH_MAX_WORKERS) as pool:
                futs = {pool.submit(_read_and_verify, model, p): p["label"] for p in PRODUCTS}
                for fut in as_completed(futs):
                    label = futs[fut]
                    try:
                        batch_runs[label] = fut.result()
                    except Exception as exc:
                        batch_runs[label] = {"error": str(exc)[:200]}
            batch_total = round(time.perf_counter() - t0, 2)
            n_err = sum("error" in r for r in batch_runs.values())
            print(f"  batch (3 products, {BATCH_MAX_WORKERS} workers): {batch_total}s total"
                  + (f"  ({n_err} errored)" if n_err else ""))
        except Exception as exc:
            print(f"  batch: ERROR {str(exc)[:120]}")

        # --- accuracy aggregation over the SINGLE runs ---
        all_single = [r for runs in single.values() for r in runs if "error" not in r]
        n_cells = sum(len(r["statuses"]) for r in all_single)
        # CLEAN signal: false-fails on NON-warning fields (those are compliant + app-matched).
        nonwarn_fail_counter = Counter(f for r in all_single for f in r["nonwarning_fail_fields"])
        n_nonwarn_fail = sum(len(r["nonwarning_fail_fields"]) for r in all_single)
        n_review = sum(len(r["review_fields"]) for r in all_single)
        review_counter = Counter(f for r in all_single for f in r["review_fields"])
        # warning: tracked on its own (bold-gate-driven), with cause + bold observations.
        warn_dist = Counter(r["warning"] for r in all_single)
        warn_cause = Counter(r["warning_cause"] for r in all_single if r["warning_cause"])
        overall_dist = Counter(r["overall"] for r in all_single)
        bev_ok = sum(r["beverage_ok"] for r in all_single)
        times = sorted(r["extract_seconds"] for r in all_single)
        stab = _stability(single)

        report[model] = {
            "available": True,
            "single_runs": single,
            "batch_runs": batch_runs,
            "batch_total_seconds": batch_total,
            "metrics": {
                "single_n": len(all_single),
                "single_mean_s": round(sum(times) / len(times), 2),
                "single_median_s": times[len(times) // 2],
                "single_min_s": times[0],
                "single_max_s": times[-1],
                "nonwarning_false_fail_cells": n_nonwarn_fail,
                "nonwarning_fail_fields": dict(nonwarn_fail_counter),
                "review_cells": n_review,
                "review_fields": dict(review_counter),
                "total_field_cells": n_cells,
                "warning_verdicts": dict(warn_dist),
                "warning_causes": dict(warn_cause),
                "overall_verdicts": dict(overall_dist),
                "beverage_type_ok": f"{bev_ok}/{len(all_single)}",
                "stability": round(stab, 3) if stab is not None else None,
            },
        }
        m = report[model]["metrics"]
        print(f"  -> non-warning false-fails {m['nonwarning_false_fail_cells']}, "
              f"warning {dict(warn_dist)}, reviews {m['review_cells']}, "
              f"stability {m['stability']}, single mean {m['single_mean_s']}s, batch {batch_total}s\n")

    _write_report(models, report)


# --- reporting ---------------------------------------------------------------

def _write_report(models, report):
    L = ["", "=" * 92, "MODEL TIME + ACCURACY EXPERIMENT  (clearer_baseline_labels, clean compliant)",
         "=" * 92,
         "products: clear_baseline_1=rum/spirits, _2=malt/beer, _3=wine  (3 single repeats + 1 batch)",
         "accuracy: these labels are COMPLIANT, so any FAIL is a FALSE FAIL; needs_review is "
         "acceptable (bold gate).", ""]

    # summary table  (false-fail = NON-warning fields only; warning verdict is its own column)
    L.append(f"{'model':14s} {'avail':6s} {'single':8s} {'median':7s} {'batch':7s} "
             f"{'NW-fail':8s} {'warn(P/R/F)':13s} {'reviews':8s} {'stable':7s} {'bev':5s}")
    L.append("-" * 92)
    for model in models:
        r = report.get(model, {})
        if not r.get("available"):
            L.append(f"{model:14s} {'NO':6s}  {str(r.get('reason',''))[:60]}")
            continue
        m = r["metrics"]
        wv = m["warning_verdicts"]
        warn_prf = f"{wv.get(PASS,0)}/{wv.get(REVIEW,0)}/{wv.get(FAIL,0)}"
        L.append(f"{model:14s} {'yes':6s} {str(m['single_mean_s'])+'s':8s} "
                 f"{str(m['single_median_s'])+'s':7s} {str(r['batch_total_seconds'])+'s':7s} "
                 f"{str(m['nonwarning_false_fail_cells']):8s} {warn_prf:13s} "
                 f"{str(m['review_cells']):8s} {str(m['stability']):7s} {m['beverage_type_ok']:5s}")
    L.append("  NW-fail = false FAILs on non-warning fields (these are compliant -> should be 0).")
    L.append("  warn(P/R/F) = government-warning verdict counts over 9 runs (bold-gate-driven).")
    L.append("")

    # per-model detail
    for model in models:
        r = report.get(model, {})
        if not r.get("available"):
            L.append(f"--- {model}: UNAVAILABLE ({str(r.get('reason',''))[:120]}) ---\n")
            continue
        m = r["metrics"]
        L.append(f"--- {model} ---")
        L.append(f"   timing:   single mean {m['single_mean_s']}s  median {m['single_median_s']}s  "
                 f"(min {m['single_min_s']}s, max {m['single_max_s']}s)   batch(3) {r['batch_total_seconds']}s")
        L.append(f"   accuracy: non-warning false-fail {m['nonwarning_false_fail_cells']}   "
                 f"reviews {m['review_cells']}   stability {m['stability']}   "
                 f"beverage_type {m['beverage_type_ok']}")
        L.append(f"   overall verdicts: {m['overall_verdicts']}")
        L.append(f"   warning verdicts: {m['warning_verdicts']}   causes: {m['warning_causes']}")
        if m["nonwarning_fail_fields"]:
            L.append(f"   *** NON-WARNING FALSE-FAILS: {m['nonwarning_fail_fields']} ***")
        if m["review_fields"]:
            L.append(f"   review fields:     {m['review_fields']}")
        # per-product per-repeat: overall / warning(cause) + bold observation
        for label, runs in r["single_runs"].items():
            cells = []
            for rr in runs:
                if "error" in rr:
                    cells.append("ERR")
                else:
                    o = rr["warning_obs"]
                    cells.append(f"{rr['overall']}/{rr['warning']}"
                                 f"[hb={o['header_bold']}/{o['header_bold_confidence'][:1]},"
                                 f"bb={o['body_bold']}/{o['body_bold_confidence'][:1]}]")
            L.append(f"     {label}: " + "  ".join(cells))
        L.append("       (overall/warning[header_bold/conf, body_bold/conf] per repeat)")
        L.append("")

    # rankings
    avail = [m for m in models if report.get(m, {}).get("available")]
    if avail:
        by_speed = sorted(avail, key=lambda m: report[m]["metrics"]["single_mean_s"])
        L.append("--- ranked by single-call speed (fastest first) ---")
        for m in by_speed:
            L.append(f"   {m:14s} {report[m]['metrics']['single_mean_s']}s mean  "
                     f"batch(3) {report[m]['batch_total_seconds']}s")
        by_acc = sorted(avail, key=lambda m: (report[m]["metrics"]["nonwarning_false_fail_cells"],
                                              report[m]["metrics"]["review_cells"],
                                              -(report[m]["metrics"]["stability"] or 0)))
        L.append("\n--- ranked by accuracy (fewest non-warning false-fails, then reviews, most stable) ---")
        for m in by_acc:
            mm = report[m]["metrics"]
            L.append(f"   {m:14s} non-warning false-fail {mm['nonwarning_false_fail_cells']}  "
                     f"reviews {mm['review_cells']}  stability {mm['stability']}  "
                     f"warning {mm['warning_verdicts']}")
        L.append("")

    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt = os.path.join(OUT_DIR, f"model_image_experiment_{stamp}.txt")
    js = os.path.join(OUT_DIR, f"model_image_experiment_{stamp}.json")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written to:\n  {os.path.relpath(txt, ROOT)}\n  {os.path.relpath(js, ROOT)}")


if __name__ == "__main__":
    main()
