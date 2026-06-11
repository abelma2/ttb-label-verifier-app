"""Latency-reduction experiment: can we get a full product read to ~5s WITHOUT losing the
bold-reading accuracy that only the gpt-5.x models have?

The measured single-call latency is dominated by SEQUENTIAL output decode (+ reasoning on the
gpt-5.x models), not the prompt -- so the lever is STRUCTURAL: read the front and back in two
calls that run IN PARALLEL (wall-clock = max(front, back), and each call decodes less) instead
of one ~10s serial front+back read.

Key idea: the government warning -- the only bold-critical read -- lives on the BACK label. So a
fast, bold-BLIND model (gpt-4o) can read the FRONT (brand/class/ABV: it scored 0 false-fails on
those), while a real bold-reader (gpt-5.4-mini / gpt-5.5) reads the BACK (warning + net +
address). The expensive accuracy is isolated to one smaller parallel call.

Variants (all measured for wall-clock + warning verdict + NON-warning false-fails, merged):
  - combined            : today's path -- one call, both images (baseline)
  - split_full          : front-call + back-call, SAME model, full schema, in parallel
  - split_het           : gpt-4o on the FRONT  ||  bold model on the BACK, in parallel

Both split calls use the FULL schema so the back call still reads the back-label non-warning
fields (net contents, name/address) -- a warning-ONLY back call would miss those.

Ground truth + product->application mapping are reused from model_image_experiment.

Usage:  python scripts/benchmarks/latency_variants.py
Writes output/latency_variants_<ts>.{txt,json}.  Real billed calls; cost ignored by design.
"""
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT
OUT_DIR = os.path.join(ROOT, "output")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import model_image_experiment as M   # PRODUCTS (loaded + app), _extract_with_model, key load
from verification import verify, PASS, REVIEW, FAIL

REPEATS = 2
_CONF = {"high": 3, "medium": 2, "low": 1}
# scalar fields merged from whichever call read them (front vs back image)
_MERGE_SCALARS = ("brand_name", "fanciful_name", "class_type", "statement_of_composition",
                  "net_contents", "name_and_address", "country_of_origin", "appellation",
                  "vintage", "sulfite_declaration", "alcohol_content")


def _pick(a, b):
    """Pick the better of two field objects for the SAME field read from the two images:
    prefer the one that's present; if both, the higher-confidence one; ties -> back (b)."""
    a, b = a or {}, b or {}
    ap, bp = bool(a.get("present")), bool(b.get("present"))
    if ap and not bp:
        return a
    if bp and not ap:
        return b
    if ap and bp:
        return a if _CONF.get(a.get("confidence"), 0) > _CONF.get(b.get("confidence"), 0) else b
    return b or a


def _merge(front, back):
    """Merge a front-image extraction and a back-image extraction into one schema dict.
    The government_warning comes from whichever call actually read it (the BACK), preserving
    that call's bold observations."""
    out = dict(back)
    for f in _MERGE_SCALARS:
        out[f] = _pick(front.get(f), back.get(f))
    fw, bw = front.get("government_warning") or {}, back.get("government_warning") or {}
    out["government_warning"] = bw if (bw.get("present") and bw.get("text")) else (
        fw if fw.get("present") else bw)
    fb, bbev = front.get("beverage_type"), back.get("beverage_type")
    out["beverage_type"] = fb if fb and fb != "unknown" else bbev
    out["additional_statements"] = ((front.get("additional_statements") or [])
                                    + (back.get("additional_statements") or []))
    out["image_quality_notes"] = front.get("image_quality_notes") or back.get("image_quality_notes")
    return out


def _timed(fn, *a):
    t0 = time.perf_counter()
    return fn(*a), round(time.perf_counter() - t0, 2)


def run_combined(model, product):
    (ex, secs) = _timed(M._extract_with_model, model, product["loaded"])
    return ex, {"wall": secs, "front": None, "back": None}


def run_split(model_front, model_back, product):
    """Front and back read in PARALLEL; wall-clock measured around the whole block."""
    front_img = [product["loaded"][0]]
    back_img = [product["loaded"][1]]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_fut = pool.submit(_timed, M._extract_with_model, model_front, front_img)
        b_fut = pool.submit(_timed, M._extract_with_model, model_back, back_img)
        (front_ex, f_s) = f_fut.result()
        (back_ex, b_s) = b_fut.result()
    wall = round(time.perf_counter() - t0, 2)
    return _merge(front_ex, back_ex), {"wall": wall, "front": f_s, "back": b_s}


VARIANTS = [
    {"name": "combined gpt-5.4-mini", "fn": lambda p: run_combined("gpt-5.4-mini", p)},
    {"name": "combined gpt-5.5",      "fn": lambda p: run_combined("gpt-5.5", p)},
    {"name": "split  5.4-mini|5.4-mini", "fn": lambda p: run_split("gpt-5.4-mini", "gpt-5.4-mini", p)},
    {"name": "split  5.5|5.5",           "fn": lambda p: run_split("gpt-5.5", "gpt-5.5", p)},
    {"name": "split  4o|5.4-mini",       "fn": lambda p: run_split("gpt-4o", "gpt-5.4-mini", p)},
    {"name": "split  4o|5.5",            "fn": lambda p: run_split("gpt-4o", "gpt-5.5", p)},
]


def _verify_record(merged, product):
    result = verify(merged, product["app"])
    statuses = {f.field: f.status for f in result["fields"]}
    gw = next((f for f in result["fields"] if f.field == "government_warning"), None)
    nonwarn_fail = [f for f, s in statuses.items() if s == FAIL and f != "government_warning"]
    return {"overall": result["overall"], "warning": getattr(gw, "status", None),
            "warning_cause": getattr(gw, "cause", None), "nonwarning_fail": nonwarn_fail}


def main():
    print(f"Variants: {len(VARIANTS)}   products: {len(M.PRODUCTS)}   repeats: {REPEATS}\n")
    report = {}
    for v in VARIANTS:
        walls, fronts, backs = [], [], []
        warn_dist, overall_dist = Counter(), Counter()
        nonwarn_fail_total = 0
        per_product = {}
        print(f"=== {v['name']} ===")
        for p in M.PRODUCTS:
            cells = []
            for rep in range(1, REPEATS + 1):
                try:
                    merged, t = v["fn"](p)
                    rec = _verify_record(merged, p)
                    walls.append(t["wall"])
                    if t["front"] is not None:
                        fronts.append(t["front"]); backs.append(t["back"])
                    warn_dist[rec["warning"]] += 1
                    overall_dist[rec["overall"]] += 1
                    nonwarn_fail_total += len(rec["nonwarning_fail"])
                    comp = (f" [f={t['front']} b={t['back']}]" if t["front"] is not None else "")
                    cells.append(f"{t['wall']}s {rec['overall']}/{rec['warning']}{comp}"
                                 + (f" FAIL={rec['nonwarning_fail']}" if rec["nonwarning_fail"] else ""))
                    print(f"  {p['label']} rep{rep}: {cells[-1]}")
                except Exception as exc:
                    cells.append(f"ERR {str(exc)[:80]}")
                    print(f"  {p['label']} rep{rep}: ERROR {str(exc)[:100]}")
            per_product[p["label"]] = cells
        if walls:
            walls_s = sorted(walls)
            report[v["name"]] = {
                "wall_mean": round(sum(walls) / len(walls), 2),
                "wall_median": walls_s[len(walls_s) // 2],
                "wall_min": walls_s[0], "wall_max": walls_s[-1],
                "front_mean": round(sum(fronts) / len(fronts), 2) if fronts else None,
                "back_mean": round(sum(backs) / len(backs), 2) if backs else None,
                "warning_verdicts": dict(warn_dist),
                "overall_verdicts": dict(overall_dist),
                "nonwarning_false_fail": nonwarn_fail_total,
                "per_product": per_product,
            }
            r = report[v["name"]]
            print(f"  -> wall mean {r['wall_mean']}s (median {r['wall_median']}s), "
                  f"warning {dict(warn_dist)}, non-warning false-fails {nonwarn_fail_total}\n")
    _write(report)


def _write(report):
    L = ["", "=" * 92,
         "LATENCY-REDUCTION VARIANTS  (target ~5s, accuracy-first)  clearer_baseline_labels",
         "=" * 92,
         "wall = parallel wall-clock per product; f/b = front/back component seconds; "
         "warning P/R/F over 6 runs.", ""]
    L.append(f"{'variant':26s} {'wall mean':10s} {'median':7s} {'front':6s} {'back':6s} "
             f"{'warn(P/R/F)':13s} {'NW-fail':7s}")
    L.append("-" * 92)
    for name, r in report.items():
        wv = r["warning_verdicts"]
        prf = f"{wv.get(PASS,0)}/{wv.get(REVIEW,0)}/{wv.get(FAIL,0)}"
        L.append(f"{name:26s} {str(r['wall_mean'])+'s':10s} {str(r['wall_median'])+'s':7s} "
                 f"{str(r['front_mean'] or '-'):6s} {str(r['back_mean'] or '-'):6s} "
                 f"{prf:13s} {str(r['nonwarning_false_fail']):7s}")
    L.append("")
    # rank by speed
    by_speed = sorted(report.items(), key=lambda kv: kv[1]["wall_mean"])
    L.append("--- ranked by wall-clock (fastest first) ---")
    for name, r in by_speed:
        wv = r["warning_verdicts"]
        L.append(f"   {name:26s} {r['wall_mean']}s   warning P/R/F "
                 f"{wv.get(PASS,0)}/{wv.get(REVIEW,0)}/{wv.get(FAIL,0)}   "
                 f"NW-fail {r['nonwarning_false_fail']}")
    L.append("")
    for name, r in report.items():
        L.append(f"--- {name} ---")
        L.append(f"   wall mean {r['wall_mean']}s  median {r['wall_median']}s  "
                 f"(min {r['wall_min']}s max {r['wall_max']}s)   front {r['front_mean']}  back {r['back_mean']}")
        L.append(f"   overall {r['overall_verdicts']}   warning {r['warning_verdicts']}   "
                 f"non-warning false-fails {r['nonwarning_false_fail']}")
        for label, cells in r["per_product"].items():
            L.append(f"     {label}: " + "  |  ".join(cells))
        L.append("")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt = os.path.join(OUT_DIR, f"latency_variants_{stamp}.txt")
    js = os.path.join(OUT_DIR, f"latency_variants_{stamp}.json")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written to:\n  {os.path.relpath(txt, ROOT)}\n  {os.path.relpath(js, ROOT)}")


if __name__ == "__main__":
    main()
