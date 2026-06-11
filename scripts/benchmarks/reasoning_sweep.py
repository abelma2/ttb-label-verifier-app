"""Model x reasoning-effort sweep on the clean baseline labels (full-res), time + accuracy.

Sweeps reasoning_effort {low, medium, high} on the REASONING models (gpt-5.4-mini, gpt-5.4-nano)
and runs the NON-reasoning models (gpt-4.1 / -mini / -nano, gpt-4o / -mini) once -- they have no
reasoning knob (temperature=0). 3 products x 3 reps per config = a stability baseline.

For each config it runs the production pipeline (extract_fields with the config's params ->
verify() against each product's application) and records:
  - time (mean/median/min/max)
  - NON-warning false-fails (clean signal: these compliant fields should never FAIL)
  - government-warning verdict P/R/F (bold-gate-driven; the accuracy-critical signal)
  - overall verdict mix, beverage_type correctness

Reasoning capability is detected from _model_params (whether it emits reasoning_effort), so the
matrix is built correctly without hardcoding which model is which.

Usage:  python scripts/benchmarks/reasoning_sweep.py [reps]
Writes output/reasoning_sweep_<ts>.{txt,json}.  Real billed calls.
"""
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT
OUT_DIR = os.path.join(ROOT, "output")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import model_image_experiment as M
from extraction import (_build_content, _model_params, _create_with_fallbacks, _get_client,
                        _parse_response)
from verification import verify, PASS, REVIEW, FAIL

MODELS = ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-4.1-mini", "gpt-4.1",
          "gpt-4o-mini", "gpt-4o", "gpt-4.1-nano"]
REASONING_LEVELS = ["low", "medium", "high"]
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def _is_reasoning(model):
    return "reasoning_effort" in _model_params(model)


def _build_configs():
    configs = []
    for m in MODELS:
        if _is_reasoning(m):
            for lvl in REASONING_LEVELS:
                configs.append({"model": m, "reasoning": lvl, "label": f"{m} [{lvl}]"})
        else:
            configs.append({"model": m, "reasoning": None, "label": f"{m} [n/a]"})
    return configs


def _extract(model, reasoning, images):
    params = _model_params(model)
    if reasoning is not None:
        params["reasoning_effort"] = reasoning      # override the family default
        # medium/high reasoning emits far more reasoning tokens than the production 3000 cap,
        # which truncates the JSON (finish_reason=length). Give the reasoning configs headroom
        # so the read isn't cut off -- the model still stops when the JSON is done.
        params["max_completion_tokens"] = 16000
    content = _build_content(images, "image/png")
    return _parse_response(_create_with_fallbacks(_get_client(), content, params))


def _run_one(model, reasoning, product):
    t0 = time.perf_counter()
    ex = _extract(model, reasoning, product["loaded"])
    dt = round(time.perf_counter() - t0, 2)
    r = verify(ex, product["app"])
    statuses = {f.field: f.status for f in r["fields"]}
    gw = next((f for f in r["fields"] if f.field == "government_warning"), None)
    nonwarn_fail = [f for f, s in statuses.items() if s == FAIL and f != "government_warning"]
    return {"t": dt, "overall": r["overall"], "warning": getattr(gw, "status", None),
            "warning_cause": getattr(gw, "cause", None),
            "beverage_ok": ex.get("beverage_type") == product["expected_bev"],
            "nonwarning_fail": nonwarn_fail}


def main():
    configs = _build_configs()
    print(f"{len(configs)} configs x {len(M.PRODUCTS)} products x {REPS} reps "
          f"= {len(configs)*len(M.PRODUCTS)*REPS} calls (full-res baseline)\n")
    report = {}
    for cfg in configs:
        print(f"=== {cfg['label']} ===")
        times, warn, overall, warn_cause = [], Counter(), Counter(), Counter()
        nonwarn_fail_total, bev_ok, n_ok, n_err = 0, 0, 0, 0
        nonwarn_fields = Counter()
        per_product = {}
        for p in M.PRODUCTS:
            cells = []
            for rep in range(1, REPS + 1):
                try:
                    r = _run_one(cfg["model"], cfg["reasoning"], p)
                    times.append(r["t"]); n_ok += 1
                    warn[r["warning"]] += 1
                    overall[r["overall"]] += 1
                    if r["warning_cause"]:
                        warn_cause[r["warning_cause"]] += 1
                    bev_ok += int(r["beverage_ok"])
                    nonwarn_fail_total += len(r["nonwarning_fail"])
                    for f in r["nonwarning_fail"]:
                        nonwarn_fields[f] += 1
                    cells.append(f"{r['t']}s {r['overall']}/{r['warning']}"
                                 + (f" FAIL={r['nonwarning_fail']}" if r["nonwarning_fail"] else ""))
                    print(f"  {p['label']} rep{rep}: {cells[-1]}")
                except Exception as exc:
                    n_err += 1
                    cells.append(f"ERR {str(exc)[:70]}")
                    print(f"  {p['label']} rep{rep}: ERROR {str(exc)[:90]}")
            per_product[p["label"]] = cells
        if times:
            ts = sorted(times)
            report[cfg["label"]] = {
                "model": cfg["model"], "reasoning": cfg["reasoning"],
                "n_ok": n_ok, "n_err": n_err,
                "time_mean": round(sum(times) / len(times), 2), "time_median": ts[len(ts) // 2],
                "time_min": ts[0], "time_max": ts[-1],
                "nonwarning_false_fail": nonwarn_fail_total, "nonwarning_fields": dict(nonwarn_fields),
                "warning_verdicts": dict(warn), "warning_causes": dict(warn_cause),
                "overall_verdicts": dict(overall), "beverage_ok": f"{bev_ok}/{n_ok}",
                "per_product": per_product,
            }
            r = report[cfg["label"]]
            print(f"  -> time mean {r['time_mean']}s  NW-fail {nonwarn_fail_total}  "
                  f"warn {dict(warn)}  bev {r['beverage_ok']}"
                  + (f"  ERRORS={n_err}" if n_err else "") + "\n")
        else:
            report[cfg["label"]] = {"model": cfg["model"], "reasoning": cfg["reasoning"],
                                    "n_ok": 0, "n_err": n_err, "unavailable": True,
                                    "per_product": per_product}
            print(f"  -> UNAVAILABLE ({n_err} errors)\n")
    _write(report)


def _write(report):
    L = ["", "=" * 100,
         f"MODEL x REASONING SWEEP  ({REPS}x, full-res clean baseline labels vs application)",
         "=" * 100,
         "NW-fail = false FAILs on non-warning (compliant) fields -> should be 0.",
         "warn(P/R/F) = government-warning verdict over all runs (the bold-gate / accuracy-critical signal).",
         "reasoning levels only vary for the gpt-5.4 reasoning models; [n/a] = no reasoning knob.", ""]
    L.append(f"{'config':24s} {'time mean':10s} {'median':8s} {'NW-fail':8s} "
             f"{'warn(P/R/F)':13s} {'overall(P/R/F)':15s} {'bev':6s}")
    L.append("-" * 100)
    for label, r in report.items():
        if r.get("unavailable"):
            L.append(f"{label:24s} UNAVAILABLE ({r['n_err']} errors)")
            continue
        wv, ov = r["warning_verdicts"], r["overall_verdicts"]
        wprf = f"{wv.get(PASS,0)}/{wv.get(REVIEW,0)}/{wv.get(FAIL,0)}"
        oprf = f"{ov.get(PASS,0)}/{ov.get(REVIEW,0)}/{ov.get(FAIL,0)}"
        err = f"  ({r['n_err']}err)" if r.get("n_err") else ""
        L.append(f"{label:24s} {str(r['time_mean'])+'s':10s} {str(r['time_median'])+'s':8s} "
                 f"{str(r['nonwarning_false_fail']):8s} {wprf:13s} {oprf:15s} {r['beverage_ok']:6s}{err}")
    L.append("")
    # rankings (available configs)
    av = {k: v for k, v in report.items() if not v.get("unavailable")}
    if av:
        L.append("--- fastest (by mean time) ---")
        for k, v in sorted(av.items(), key=lambda kv: kv[1]["time_mean"]):
            L.append(f"   {k:24s} {v['time_mean']}s   NW-fail {v['nonwarning_false_fail']}   "
                     f"warn P/R/F {v['warning_verdicts'].get(PASS,0)}/"
                     f"{v['warning_verdicts'].get(REVIEW,0)}/{v['warning_verdicts'].get(FAIL,0)}")
        L.append("\n--- most accurate (fewest NW-fails, then most warning passes, then faster) ---")
        for k, v in sorted(av.items(), key=lambda kv: (kv[1]["nonwarning_false_fail"],
                           -kv[1]["warning_verdicts"].get(PASS, 0), kv[1]["time_mean"])):
            L.append(f"   {k:24s} NW-fail {v['nonwarning_false_fail']}   "
                     f"warn P/R/F {v['warning_verdicts'].get(PASS,0)}/"
                     f"{v['warning_verdicts'].get(REVIEW,0)}/{v['warning_verdicts'].get(FAIL,0)}   "
                     f"{v['time_mean']}s")
        L.append("")
    # per-config warning detail
    for label, r in report.items():
        if r.get("unavailable"):
            continue
        L.append(f"--- {label} ---  time {r['time_mean']}s (med {r['time_median']}s, "
                 f"{r['time_min']}..{r['time_max']})")
        L.append(f"   NW-fail {r['nonwarning_false_fail']} {r['nonwarning_fields'] or ''}   "
                 f"warning {r['warning_verdicts']}  causes {r['warning_causes']}   "
                 f"overall {r['overall_verdicts']}   bev {r['beverage_ok']}")
        for plabel, cells in r["per_product"].items():
            L.append(f"     {plabel}: " + "  |  ".join(cells))
        L.append("")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt = os.path.join(OUT_DIR, f"reasoning_sweep_{stamp}.txt")
    js = os.path.join(OUT_DIR, f"reasoning_sweep_{stamp}.json")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written to:\n  {os.path.relpath(txt, ROOT)}\n  {os.path.relpath(js, ROOT)}")


if __name__ == "__main__":
    main()
