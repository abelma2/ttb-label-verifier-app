"""Summarize per-model latency from saved model_benchmark JSON runs.

Every benchmark cell records the extraction read time; this aggregates it per model so you
can compare model speed. Times are split by single-image (adv_*) vs two-image (baseline_*)
calls, since reading a front+back pair is slower than one image.

With no args it scans output/model_benchmark_*.json and uses the MOST RECENT run that
contains each model.

Run:  python scripts/benchmarks/timing_summary.py
      python scripts/benchmarks/timing_summary.py output/model_benchmark_A.json output/...B.json
"""
import glob
import json
import os
import statistics
import sys

import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "output", "model_benchmark_*.json")))
    if not files:
        sys.exit("No benchmark JSON files found. Run scripts/benchmarks/model_benchmark.py first.")

    # model -> {case_id: elapsed}; iterate oldest->newest so the newest run wins per model
    per = {}
    used = {}
    for f in sorted(files):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        for model, cases in data.get("results", {}).items():
            for cid, cell in cases.items():
                if isinstance(cell, dict) and "elapsed" in cell:
                    per.setdefault(model, {})[cid] = cell["elapsed"]
                    used[model] = os.path.basename(f)

    rows = []
    for model, cases in per.items():
        times = list(cases.values())
        one = [t for c, t in cases.items() if c.startswith("adv_")]
        two = [t for c, t in cases.items() if c.startswith("baseline")]
        rows.append({
            "model": model, "n": len(times),
            "mean": statistics.mean(times), "median": statistics.median(times),
            "min": min(times), "max": max(times),
            "one": statistics.mean(one) if one else None,
            "two": statistics.mean(two) if two else None,
            "src": used[model],
        })
    rows.sort(key=lambda r: r["mean"])

    print(f"\nLatency per model (seconds), fastest first — from {len(files)} run file(s)\n")
    print(f"{'model':16s} {'n':>2s}  {'mean':>5s} {'med':>5s} {'min':>5s} {'max':>5s}  "
          f"{'1img':>5s} {'2img':>5s}")
    print("-" * 64)

    def f(v):
        return f"{v:5.2f}" if v is not None else "   — "

    for r in rows:
        print(f"{r['model']:16s} {r['n']:>2d}  {f(r['mean'])} {f(r['median'])} "
              f"{f(r['min'])} {f(r['max'])}  {f(r['one'])} {f(r['two'])}")

    print("\nnote: 1img = single-image adversarial cases; 2img = front+back baseline pairs. "
          "Times are wall-clock per extraction call and include network; rerun for fresh numbers.")


if __name__ == "__main__":
    main()
