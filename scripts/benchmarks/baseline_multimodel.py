"""Full-pipeline baseline check across many models.

For each model it screens the three baseline bottles (extract front+back, then the COMPLETE
verify_label_only -- all mandatory fields AND the government warning) and prints a compact
model x bottle grid of the OVERALL verdict, so you can see which models clear the compliant
baselines and exactly where they trip.

Reuses the per-model client / extraction helpers from model_benchmark.py. Runs under the
CURRENT config (so WARNING_BOLD_POLICY affects the warning verdict). Unknown model names are
reported as unavailable rather than crashing the run.

Usage:
  python scripts/benchmarks/baseline_multimodel.py                       # default set
  python scripts/benchmarks/baseline_multimodel.py gpt-4.1 gpt-4o o4-mini ...
"""
import json
import os
import sys
import time
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
from verification import verify_label_only, PASS, REVIEW, FAIL

BASE = os.path.join(ROOT, "test_labels", "baseline_labels")
BOTTLES = [
    ("baseline_1", [os.path.join(BASE, "baseline_1_Front.png"), os.path.join(BASE, "baseline_1_Other.png")]),
    ("baseline_2", [os.path.join(BASE, "baseline_2_Front.png"), os.path.join(BASE, "baseline_2_Other.png")]),
    ("baseline_3", [os.path.join(BASE, "baseline_3_Front.png"), os.path.join(BASE, "baseline_3_Other.png")]),
]
DEFAULT_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
                  "gpt-5", "gpt-5-mini", "gpt-5-nano", "o4-mini"]
_ABBR = {PASS: "PASS", REVIEW: "REVIEW", FAIL: "FAIL"}


def _warn_cause(fields):
    fr = next((f for f in fields if f.field == "government_warning"), None)
    r = (fr.reason if fr else "").lower()
    if "bold" in r:
        return "bold"
    if "capital" in r:
        return "caps"
    if "wording" in r or "match" in r:
        return "wording"
    if "no government warning" in r:
        return "missing"
    return ""


def _cell(res):
    """Compact token for one bottle: PASS, or <verdict>·<cause>."""
    bad = [f for f in res["fields"] if f.status != PASS]
    if not bad:
        return "PASS"
    f = bad[0]   # baselines fail on at most one thing; show the first
    if f.field == "government_warning":
        cause = _warn_cause(res["fields"])
        tag = f"warn-{cause}" if cause else "warn"
    else:
        tag = f.field.replace("_", "")
    return f"{_ABBR[res['overall']]}·{tag}"


def main():
    requested = sys.argv[1:] or DEFAULT_MODELS
    openai_key = _load_keys()
    if not openai_key:
        sys.exit("ERROR: no OpenAI key (env OPENAI_API_KEY or .streamlit/secrets.toml).")
    models = _build_model_list(set(requested), openai_key)
    built = {m["name"]: m for m in models}

    loaded = {bid: [(open(p, "rb").read(), "image/png") for p in paths] for bid, paths in BOTTLES}
    for m in models:
        m["client"] = _client_for(m)

    grid = {}   # model -> bottle -> {cell, time, overall, err}
    jobs = [(m, bid) for m in models for bid, _ in BOTTLES]
    total, done = len(jobs), 0
    print(f"Screening {len(models)} model(s) x {len(BOTTLES)} bottles = {total} runs "
          f"(policy drives the warning verdict)...\n")

    def _run(m, bid):
        try:
            t = time.perf_counter()
            extracted = _extract(m["client"], m["name"], loaded[bid])
            el = time.perf_counter() - t
            res = verify_label_only(extracted)
            return m["name"], bid, {"cell": _cell(res), "time": round(el, 2), "overall": res["overall"]}
        except Exception as exc:
            return m["name"], bid, {"cell": "ERR", "err": str(exc)[:120]}

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_run, m, bid) for m, bid in jobs]
        for fut in as_completed(futs):
            name, bid, cell = fut.result()
            done += 1
            grid.setdefault(name, {})[bid] = cell
            print(f"  [{done}/{total}] {name:16s} {bid:12s} -> {cell['cell']}")

    # --- grid report (requested order) ---
    lines = ["", "=" * 78, "BASELINE FULL-PIPELINE CHECK  (compliant labels; expect all PASS)", "=" * 78,
             "cell = overall verdict; '·warn-bold' etc = the field/reason that broke it", ""]
    hdr = f"{'model':16s} {'baseline_1':14s} {'baseline_2':14s} {'baseline_3':14s}  pass"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for name in requested:
        if name not in built:
            lines.append(f"{name:16s} (not run — no key)")
            continue
        cells = grid.get(name, {})
        if all(c.get("cell") == "ERR" for c in cells.values()):
            err = next((c.get("err", "") for c in cells.values() if "err" in c), "")
            lines.append(f"{name:16s} UNAVAILABLE — {err}")
            continue
        row = [name]
        npass = 0
        for bid, _ in BOTTLES:
            c = cells.get(bid, {})
            row.append(c.get("cell", "?"))
            if c.get("overall") == PASS:
                npass += 1
        lines.append(f"{row[0]:16s} {row[1]:14s} {row[2]:14s} {row[3]:14s}  {npass}/3")

    # timing
    lines.append("")
    lines.append("read time (mean s/bottle):")
    times = []
    for name in requested:
        ts = [c["time"] for c in grid.get(name, {}).values() if "time" in c]
        if ts:
            times.append((name, sum(ts) / len(ts)))
    for name, mt in sorted(times, key=lambda x: x[1]):
        lines.append(f"   {name:16s} {mt:.2f}s")

    report = "\n".join(lines)
    print(report)

    out = os.path.join(ROOT, "output", f"baseline_multimodel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\nWritten to: {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
