"""10x stability run of the production pipeline on the clean baseline labels at the two
candidate downscale widths (1024 / 896), on gpt-5.4-mini -- the model + sizes targeting ~5-6s.

For each (width, product) it runs extract_fields (image downscaled to `width`) -> verify()
against the product's application 10 times, recording the per-FIELD verdict and the wall-clock,
so we can see exactly WHAT PASSES/FAILS and how STABLE it is run-to-run, plus the time at each
width. These are clean, compliant labels: every non-warning field SHOULD pass every run; the
government warning is bold-gate-driven and is the one expected to vary.

Usage:  python scripts/benchmarks/stability_downscale.py [model] [reps]
Writes output/stability_downscale_<ts>.{txt,json}.
"""
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT
OUT_DIR = os.path.join(ROOT, "output")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from PIL import Image
import model_image_experiment as M
from verification import verify, PASS, REVIEW, FAIL

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-5.4-mini"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
WIDTHS = [None, 1024, 896]   # None = original (1632px), treated as the baseline
_ABBR = {PASS: "P", REVIEW: "R", FAIL: "F"}


def _wlabel(w):
    return "1632px(orig)" if w is None else f"{w}px"


def _resize(loaded, width):
    out = []
    for b, _mt in loaded:
        im = Image.open(io.BytesIO(b)).convert("RGB")
        if width and im.width > width:
            im = im.resize((width, int(im.height * width / im.width)))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        out.append((buf.getvalue(), "image/png"))
    return out


def main():
    print(f"model: {MODEL}   widths: {WIDTHS}   products: {len(M.PRODUCTS)}   reps: {REPS}\n")
    resized = {(p["label"], w): _resize(p["loaded"], w) for p in M.PRODUCTS for w in WIDTHS}

    report = {}
    for w in WIDTHS:
        print(f"================ width {_wlabel(w)} ================")
        width_times = []
        per_product = {}
        for p in M.PRODUCTS:
            times = []
            overall_dist = Counter()
            field_dist = defaultdict(Counter)     # field -> Counter(status)
            warn_cause = Counter()
            runs = []
            for rep in range(1, REPS + 1):
                try:
                    t0 = time.perf_counter()
                    ex = M._extract_with_model(MODEL, resized[(p["label"], w)])
                    dt = round(time.perf_counter() - t0, 2)
                    r = verify(ex, p["app"])
                    statuses = {f.field: f.status for f in r["fields"]}
                    gw = next((f for f in r["fields"] if f.field == "government_warning"), None)
                    times.append(dt)
                    width_times.append(dt)
                    overall_dist[r["overall"]] += 1
                    for fld, s in statuses.items():
                        field_dist[fld][s] += 1
                    if gw is not None and getattr(gw, "cause", None):
                        warn_cause[gw.cause] += 1
                    runs.append({"t": dt, "overall": r["overall"],
                                 "warning": getattr(gw, "status", None),
                                 "statuses": statuses})
                    nonwarn_fail = [f for f, s in statuses.items()
                                    if s == FAIL and f != "government_warning"]
                    print(f"  {p['label']} rep{rep:2d}: {dt:5.2f}s  overall={r['overall']:12s} "
                          f"warn={getattr(gw,'status',None)}"
                          + (f"  NW-FAIL={nonwarn_fail}" if nonwarn_fail else ""))
                except Exception as exc:
                    print(f"  {p['label']} rep{rep:2d}: ERROR {str(exc)[:100]}")
            ts = sorted(times)
            per_product[p["label"]] = {
                "time_mean": round(sum(times) / len(times), 2) if times else None,
                "time_median": ts[len(ts) // 2] if ts else None,
                "time_min": ts[0] if ts else None, "time_max": ts[-1] if ts else None,
                "overall": dict(overall_dist),
                "fields": {f: dict(c) for f, c in field_dist.items()},
                "warning_causes": dict(warn_cause),
                "nonwarning_false_fails": sum(
                    c.get(FAIL, 0) for f, c in field_dist.items() if f != "government_warning"),
            }
            pp = per_product[p["label"]]
            print(f"   -> {p['label']}: time mean {pp['time_mean']}s  overall {dict(overall_dist)}  "
                  f"warning {dict(field_dist['government_warning'])}  "
                  f"NW-false-fails {pp['nonwarning_false_fails']}\n")
        wt = sorted(width_times)
        wl = _wlabel(w)
        report[wl] = {
            "time_mean": round(sum(width_times) / len(width_times), 2) if width_times else None,
            "time_median": wt[len(wt) // 2] if wt else None,
            "time_min": wt[0] if wt else None, "time_max": wt[-1] if wt else None,
            "products": per_product,
        }
        print(f"width {wl} overall time: mean {report[wl]['time_mean']}s "
              f"median {report[wl]['time_median']}s "
              f"(min {report[wl]['time_min']} max {report[wl]['time_max']})\n")
    _write(report)


def _write(report):
    L = ["", "=" * 90,
         f"DOWNSCALE STABILITY  ({MODEL}, {REPS}x, clean baseline labels vs application)",
         "=" * 90,
         "clean compliant labels: every NON-warning field should pass every run; the government "
         "warning is bold-gate-driven and may vary (P/R/F).", ""]
    # headline time + accuracy comparison
    L.append(f"{'width':13s} {'time mean':10s} {'median':8s} {'min..max':14s} {'NW-fails':9s} "
             f"{'warning P/R/F (all products)':28s}")
    L.append("-" * 90)
    for wkey, r in report.items():
        nwf = sum(p["nonwarning_false_fails"] for p in r["products"].values())
        warn = Counter()
        for p in r["products"].values():
            for s, n in p["fields"].get("government_warning", {}).items():
                warn[s] += n
        prf = f"{warn.get(PASS,0)}/{warn.get(REVIEW,0)}/{warn.get(FAIL,0)}"
        L.append(f"{wkey:13s} {str(r['time_mean'])+'s':10s} {str(r['time_median'])+'s':8s} "
                 f"{str(r['time_min'])+'..'+str(r['time_max']):14s} {str(nwf):9s} {prf:28s}")
    L.append("")
    # per width / per product detail
    for wkey, r in report.items():
        L.append(f"================ width {wkey}  (time mean {r['time_mean']}s, "
                 f"median {r['time_median']}s) ================")
        for label, p in r["products"].items():
            L.append(f"--- {label} ---  time {p['time_mean']}s (med {p['time_median']}s, "
                     f"{p['time_min']}..{p['time_max']})   overall {p['overall']}")
            for fld, c in p["fields"].items():
                breakdown = " ".join(f"{_ABBR.get(s,s)}={n}" for s, n in c.items())
                tag = "  <-- has FAIL" if c.get(FAIL) and fld != "government_warning" else ""
                L.append(f"     {fld:22s} {breakdown}{tag}")
            if p["warning_causes"]:
                L.append(f"     (warning causes: {p['warning_causes']})")
            L.append("")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt = os.path.join(OUT_DIR, f"stability_downscale_{stamp}.txt")
    js = os.path.join(OUT_DIR, f"stability_downscale_{stamp}.json")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written to:\n  {os.path.relpath(txt, ROOT)}\n  {os.path.relpath(js, ROOT)}")


if __name__ == "__main__":
    main()
