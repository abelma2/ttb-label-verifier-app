"""Full-pipeline multi-model test across all three label sets.

For each model and each label group -- adversarial (single, controlled ground truth), baseline
(synthetic pairs), and real-photo (pairs) -- it runs extract + verify_label_only and reports the
overall verdict, the non-pass fields, the warning bold read, and the time. Clients use
max_retries=0 so a flaky/slow model (e.g. a preview endpoint) fast-fails instead of stalling the
whole run.

Usage:
  python scripts/benchmarks/full_multimodel.py gpt-5.4-mini gpt-4.1 gpt-5
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

from openai import OpenAI
from model_benchmark import _load_keys, _build_model_list, _extract
from verification import verify_label_only, PASS, REVIEW, FAIL
from config import REQUEST_TIMEOUT_SECONDS

ADV = os.path.join(ROOT, "adversarial")
BASE = os.path.join(ROOT, "test_labels", "baseline_labels")
REAL = os.path.join(ROOT, "test_labels", "real_labels")
DEFAULT = ["gpt-5.4-mini"]
_AB = {PASS: "PASS", REVIEW: "REVIEW", FAIL: "FAIL"}

# (id, [paths], set, expected_overall_or_None)
CASES = (
    [("01_compliant", [os.path.join(ADV, "01_compliant.png")], "adv", PASS),
     ("02_titlecase", [os.path.join(ADV, "02_titlecase.png")], "adv", FAIL),
     ("03_notbold",   [os.path.join(ADV, "03_notbold.png")],   "adv", FAIL),
     ("04_reworded",  [os.path.join(ADV, "04_reworded.png")],  "adv", FAIL)]
    + [(f"baseline_{n}", [os.path.join(BASE, f"baseline_{n}_Front.png"),
                          os.path.join(BASE, f"baseline_{n}_Other.png")], "baseline", None)
       for n in range(1, 4)]
    + [(f"test_{n}", [os.path.join(REAL, f"test_{n}_Front.jpeg"),
                      os.path.join(REAL, f"test_{n}_Other.jpeg")], "real", None)
       for n in range(1, 14)]
)


def _client(cfg):
    return OpenAI(api_key=cfg["key"], timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)


def _warn_cause(res):
    fr = next((f for f in res["fields"] if f.field == "government_warning"), None)
    r = (fr.reason if fr else "").lower()
    if "bold" in r:
        return "bold"
    if "could not verify required" in r or "does not match" in r or "not an exact match" in r:
        return "wording/unverifiable"
    if "capital" in r:
        return "caps"
    return ""


def main():
    models_req = sys.argv[1:] or DEFAULT
    openai_key = _load_keys()
    if not openai_key:
        sys.exit("ERROR: no OpenAI key (env or .streamlit/secrets.toml).")
    models = _build_model_list(set(models_req), openai_key)
    for m in models:
        m["client"] = _client(m)
    loaded = {cid: [(open(p, "rb").read(), "image/png") for p in paths] for cid, paths, _, _ in CASES}

    def _run(m, cid):
        try:
            t = time.perf_counter()
            ext = _extract(m["client"], m["name"], loaded[cid])
            el = time.perf_counter() - t
            res = verify_label_only(ext)
            gw = ext.get("government_warning", {})
            nonpass = [f.field for f in res["fields"] if f.status != PASS]
            return m["name"], cid, {"overall": res["overall"], "nonpass": nonpass,
                                    "cause": _warn_cause(res), "bold": gw.get("header_bold"),
                                    "bconf": gw.get("header_bold_confidence"), "time": round(el, 1)}
        except Exception as exc:
            return m["name"], cid, {"error": str(exc)[:90]}

    grid = {}
    jobs = [(m, cid) for m in models for cid, _, _, _ in CASES]
    total, done = len(jobs), 0
    print(f"{len(models)} models x {len(CASES)} cases = {total} runs\n")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_run, m, cid) for m, cid in jobs]
        for fut in as_completed(futs):
            name, cid, cell = fut.result()
            grid.setdefault(name, {})[cid] = cell
            done += 1
            if done % 12 == 0 or done == total:
                print(f"  {done}/{total} done")

    exp = {cid: e for cid, _, _, e in CASES}
    sset = {cid: s for cid, _, s, _ in CASES}
    lines = ["", "=" * 92, "FULL-PIPELINE MULTI-MODEL TEST", "=" * 92]
    for name in [m["name"] for m in models]:
        cells = grid.get(name, {})
        lines.append(f"\n{'='*4} {name} {'='*4}")
        for setname, title in (("adv", "ADVERSARIAL (controlled: 01 pass; 02 caps, 03 bold, 04 wording fail)"),
                               ("baseline", "BASELINES (synthetic compliant)"),
                               ("real", "REAL PHOTOS")):
            ids = [cid for cid, _, s, _ in CASES if s == setname]
            lines.append(f"  -- {title} --")
            for cid in ids:
                c = cells.get(cid, {})
                if "error" in c:
                    lines.append(f"     {cid:14s} ERROR: {c['error']}")
                    continue
                v = _AB.get(c["overall"], c["overall"])
                npf = ",".join(c["nonpass"]) if c["nonpass"] else "-"
                bold = f"bold={c['bold']}/{(c['bconf'] or '?')[:3]}"
                ok = ""
                if exp[cid] is not None:
                    ok = " OK" if c["overall"] == exp[cid] else " XX"
                lines.append(f"     {cid:14s} {v:7s} [{bold:16s}] fail:{npf:24s} {c['time']:>4}s{ok}")
            # set summary
            ok_cells = [cells[cid] for cid in ids if cid in cells and "error" not in cells[cid]]
            errs = sum(1 for cid in ids if cid in cells and "error" in cells[cid])
            if setname == "adv":
                acc = sum(1 for cid in ids if cid in cells and "error" not in cells[cid]
                          and cells[cid]["overall"] == exp[cid])
                lines.append(f"     => adversarial accuracy {acc}/{len(ids)}"
                             + (f"   ({errs} errors)" if errs else ""))
            else:
                dist = Counter(c["overall"] for c in ok_cells)
                bold_fails = sum(1 for c in ok_cells if c["overall"] == FAIL and c["cause"] == "bold")
                times = [c["time"] for c in ok_cells]
                avg = sum(times) / len(times) if times else 0
                lines.append(f"     => PASS {dist.get(PASS,0)} / REVIEW {dist.get(REVIEW,0)} / "
                             f"FAIL {dist.get(FAIL,0)}  bold-fails {bold_fails}  "
                             f"avg {avg:.1f}s" + (f"  ({errs} errors)" if errs else ""))

    report = "\n".join(lines)
    print(report)
    out = os.path.join(ROOT, "output", f"full_multimodel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\nWritten to: {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
