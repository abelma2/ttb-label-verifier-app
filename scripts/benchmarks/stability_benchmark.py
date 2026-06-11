"""N-run stability pass for the serious extraction-model candidates.

Runs each model N times (default 5) on the 4 font-controlled adversarial labels and the 3
baseline bottles, then reports stability: adversarial accuracy across all runs, how often
03_notbold correctly FAILs (on bold) and 01_compliant correctly PASSes, baseline pass counts
per run, average latency, and run-to-run variance in header_bold / header_bold_confidence.

The verdict reported is the GOVERNMENT-WARNING verdict (_check_warning) -- the only thing that
varies on these labels -- under the live WARNING_BOLD_POLICY (default medium_pass_gate since
2026-06-11; historical runs used confidence_gate).

Usage:
  python scripts/benchmarks/stability_benchmark.py                          # default 4 models, 5 runs
  python scripts/benchmarks/stability_benchmark.py --runs 3 gpt-4.1 gpt-5.4-mini
"""
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from model_benchmark import _load_keys, _build_model_list, _client_for, _extract
from verification import _check_warning, PASS, REVIEW, FAIL

ADV = os.path.join(ROOT, "adversarial")
BASE = os.path.join(ROOT, "test_labels", "baseline_labels")
CASES = [
    ("01_compliant", [os.path.join(ADV, "01_compliant.png")], PASS),
    ("02_titlecase", [os.path.join(ADV, "02_titlecase.png")], FAIL),
    ("03_notbold",   [os.path.join(ADV, "03_notbold.png")],   FAIL),
    ("04_reworded",  [os.path.join(ADV, "04_reworded.png")],  FAIL),
    ("baseline_1", [os.path.join(BASE, "baseline_1_Front.png"), os.path.join(BASE, "baseline_1_Other.png")], None),
    ("baseline_2", [os.path.join(BASE, "baseline_2_Front.png"), os.path.join(BASE, "baseline_2_Other.png")], None),
    ("baseline_3", [os.path.join(BASE, "baseline_3_Front.png"), os.path.join(BASE, "baseline_3_Other.png")], None),
]
ADV_IDS = ("01_compliant", "02_titlecase", "03_notbold", "04_reworded")
BASE_IDS = ("baseline_1", "baseline_2", "baseline_3")
DEFAULT_MODELS = ["gpt-5.4-mini", "gpt-5.5", "gpt-5.4", "gpt-4.1"]
_EXPECTED = {cid: exp for cid, _, exp in CASES}


def main():
    args = sys.argv[1:]
    runs = 5
    if "--runs" in args:
        i = args.index("--runs")
        runs = int(args[i + 1])
        del args[i:i + 2]
    models_req = args or DEFAULT_MODELS

    openai_key = _load_keys()
    if not openai_key:
        sys.exit("ERROR: no OpenAI key (env OPENAI_API_KEY or .streamlit/secrets.toml).")
    models = _build_model_list(set(models_req), openai_key)
    for m in models:
        m["client"] = _client_for(m)
    loaded = {cid: [(open(p, "rb").read(), "image/png") for p in paths] for cid, paths, _ in CASES}

    results = {m["name"]: {cid: [None] * runs for cid, _, _ in CASES} for m in models}
    jobs = [(m, cid, run) for m in models for cid, _, _ in CASES for run in range(runs)]
    total, done = len(jobs), 0
    print(f"{len(models)} models x {len(CASES)} cases x {runs} runs = {total} calls\n")

    def _run(m, cid, run):
        try:
            t = time.perf_counter()
            extracted = _extract(m["client"], m["name"], loaded[cid])
            sec = time.perf_counter() - t
            gw = extracted.get("government_warning", {})
            return m["name"], cid, run, {"verdict": _check_warning(gw).status,
                                         "bold": gw.get("header_bold"),
                                         "conf": gw.get("header_bold_confidence"),
                                         "seconds": round(sec, 2)}
        except Exception as exc:
            return m["name"], cid, run, {"error": str(exc)[:120]}

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_run, m, cid, run) for m, cid, run in jobs]
        for fut in as_completed(futs):
            name, cid, run, cell = fut.result()
            results[name][cid][run] = cell
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  {done}/{total} done")

    lines = ["", "=" * 80, f"STABILITY PASS  ({runs} runs per case)  policy=confidence_gate", "=" * 80]
    summary = {}
    for m in models:
        name = m["name"]
        cells = results[name]
        ok = {cid: [c for c in cells[cid] if c and "error" not in c] for cid in cells}
        if not any(ok.values()):
            err = next((c["error"] for cid in cells for c in cells[cid] if c and "error" in c), "?")
            lines.append(f"\n--- {name} : UNAVAILABLE ({err}) ---")
            continue

        adv_correct = sum(1 for cid in ADV_IDS for c in ok[cid] if c["verdict"] == _EXPECTED[cid])
        adv_total = sum(len(ok[cid]) for cid in ADV_IDS)
        n03 = sum(1 for c in ok["03_notbold"] if c["verdict"] == FAIL)
        n01 = sum(1 for c in ok["01_compliant"] if c["verdict"] == PASS)
        n02 = sum(1 for c in ok["02_titlecase"] if c["verdict"] == FAIL)
        n04 = sum(1 for c in ok["04_reworded"] if c["verdict"] == FAIL)
        base_per_run = []
        for run in range(runs):
            cnt = sum(1 for bid in BASE_IDS
                      if cells[bid][run] and cells[bid][run].get("verdict") == PASS)
            base_per_run.append(cnt)
        secs = [c["seconds"] for cid in cells for c in ok[cid]]
        avg_lat = sum(secs) / len(secs) if secs else 0.0

        lines.append(f"\n--- {name} ---")
        lines.append(f"  adversarial accuracy:        {adv_correct}/{adv_total}")
        lines.append(f"  03_notbold FAILs (bold):     {n03}/{len(ok['03_notbold'])}")
        lines.append(f"  01_compliant PASSes:         {n01}/{len(ok['01_compliant'])}")
        lines.append(f"  02_titlecase FAILs (caps):   {n02}/{len(ok['02_titlecase'])}")
        lines.append(f"  04_reworded FAILs (wording): {n04}/{len(ok['04_reworded'])}")
        lines.append(f"  baselines passing per run:   {base_per_run}  (of 3)")
        lines.append(f"  avg latency:                 {avg_lat:.2f}s/label")
        lines.append("  run-to-run variance (header_bold / header_bold_confidence):")
        for cid, _, _ in CASES:
            bolds = Counter(str(c.get("bold")) for c in ok[cid])
            confs = Counter(str(c.get("conf")) for c in ok[cid])
            bstr = ", ".join(f"{k}x{v}" for k, v in bolds.items()) or "—"
            cstr = ", ".join(f"{k}x{v}" for k, v in confs.items()) or "—"
            lines.append(f"     {cid:15s} bold[{bstr}]   conf[{cstr}]")
        summary[name] = dict(adv_correct=adv_correct, adv_total=adv_total, n03=n03, n01=n01,
                             n02=n02, n04=n04, base_sum=sum(base_per_run), base_per_run=base_per_run,
                             avg_lat=avg_lat, runs=runs)

    # ---- decision rule ----
    lines.append("\n" + "=" * 80)
    lines.append("DECISION (applying the rule)")
    lines.append("=" * 80)
    mini, g41 = summary.get("gpt-5.4-mini"), summary.get("gpt-4.1")
    if mini:
        gate = (mini["n03"] == mini["runs"] and mini["n01"] == mini["runs"]
                and mini["n02"] == mini["runs"] and mini["n04"] == mini["runs"])
        lines.append(f"gpt-5.4-mini gate  03={mini['n03']}/{mini['runs']} 01={mini['n01']}/{mini['runs']} "
                     f"02={mini['n02']}/{mini['runs']} 04={mini['n04']}/{mini['runs']}  -> "
                     f"{'PASS' if gate else 'FAIL'}")
        if g41:
            better = mini["base_sum"] > g41["base_sum"]
            lines.append(f"baselines passed (sum over runs):  gpt-5.4-mini {mini['base_sum']} vs "
                         f"gpt-4.1 {g41['base_sum']}  -> mini {'BETTER' if better else 'not better'}")
            if gate and better:
                lines.append("==> RECOMMENDATION: switch EXTRACTION_MODEL to gpt-5.4-mini.")
            elif gate and not better:
                lines.append("==> RECOMMENDATION: gpt-5.4-mini meets the gate but is NOT better on baselines; "
                             "keep gpt-4.1 (or use gpt-5.5 if you want the generous-but-accurate reads).")
            else:
                lines.append("==> RECOMMENDATION: gpt-5.4-mini FAILS the 5/5 gate -> do NOT switch; keep gpt-4.1. "
                             "Consider gpt-5.5 only if it passes the gate.")
        g55 = summary.get("gpt-5.5")
        if g55:
            g55_gate = (g55["n03"] == g55["runs"] and g55["n01"] == g55["runs"]
                        and g55["n02"] == g55["runs"] and g55["n04"] == g55["runs"])
            lines.append(f"gpt-5.5 (ceiling) gate -> {'PASS' if g55_gate else 'FAIL'}, "
                         f"baselines {g55['base_sum']}, latency {g55['avg_lat']:.1f}s")

    report = "\n".join(lines)
    print(report)
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    out = os.path.join(ROOT, "output", f"stability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\nWritten to: {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
