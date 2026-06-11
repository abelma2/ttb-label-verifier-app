"""Run-to-run stability on ONE real photo: gpt-4.1+N and gpt-5.4-mini+N, 10x each, paired into
the fail-closed panel per rep. Tests how stable each model's bold read is on a real warning, and
whether the panel verdict is consistent across reps. No production code touched.

Usage: python scripts/benchmarks/stability_single.py [path]
Writes output/stability_single_<ts>.{txt,json}.
"""
import json
import os
import sys
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

import bold_prompt_safety as B

IMAGE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "test_labels", "real_labels", "test_1_Other.jpeg")
MODELS = ["gpt-4.1", "gpt-5.4-mini"]
PROMPT = "N"
REPS = 10


def _verdict(f):
    if not f or not f.get("warning_present"):
        return "no-warning"
    hb, bb = B._eff_header_bold(f), B._eff_body_bold(f)
    hbc, bbc = f.get("header_bold_confidence"), f.get("body_bold_confidence")
    if f.get("legibility") == "poor":
        return "review"
    if bb is True and bbc == "high":
        return "FAIL-body-bold"
    if hb is False and hbc == "high":
        return "FAIL-header"
    if hb is True and bb is False and hbc == "high" and bbc == "high":
        return "PASS"
    return "review"


def _panel(v1, v2):
    return v1 if v1 == v2 else "review (disagreement)"


def _read_str(f):
    hb, bb = B._eff_header_bold(f), B._eff_body_bold(f)
    return f"hb={hb}/{(f.get('header_bold_confidence') or '-')[:1]} bb={bb}/{(f.get('body_bold_confidence') or '-')[:1]}"


def main():
    print(f"image={os.path.basename(IMAGE)}  models={MODELS}  prompt={PROMPT}  reps={REPS}\n")
    per_model = {m: [] for m in MODELS}
    panel_verdicts = []
    for rep in range(1, REPS + 1):
        reps_now = {}
        for m in MODELS:
            fields, dt, retries, err = B._call(m, B._prompt(PROMPT), [IMAGE])
            v = _verdict(fields) if fields else "ERROR"
            per_model[m].append({"verdict": v, "fields": fields, "dt": dt, "err": err})
            reps_now[m] = (v, fields)
        v41 = reps_now["gpt-4.1"][0]
        v54 = reps_now["gpt-5.4-mini"][0]
        panel = _panel(v41, v54)
        panel_verdicts.append(panel)
        f41, f54 = reps_now["gpt-4.1"][1] or {}, reps_now["gpt-5.4-mini"][1] or {}
        print(f"  rep{rep:2d}: 4.1={v41:16s} [{_read_str(f41)}]   5.4={v54:16s} [{_read_str(f54)}]   -> {panel}")
    _write(per_model, panel_verdicts)


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def _write(per_model, panel_verdicts):
    L = ["", "=" * 96, f"STABILITY (10x) on one real photo: {os.path.basename(IMAGE)}", "=" * 96,
         "Each model read 10x; reads paired per-rep into the fail-closed panel "
         "(agree -> that; disagree -> review).", ""]
    for m in MODELS:
        recs = per_model[m]
        ok = [r for r in recs if r["fields"]]
        vdist = Counter(r["verdict"] for r in recs)
        reads = Counter(_read_str(r["fields"]) for r in ok)
        lat = [r["dt"] for r in ok if r["dt"] is not None]
        stable = len(set(r["verdict"] for r in recs)) == 1
        L.append(f"--- {m} + {PROMPT} ---")
        L.append(f"   verdicts (10): {dict(vdist)}   {'STABLE (all 10 identical)' if stable else 'UNSTABLE (varies)'}")
        L.append(f"   raw reads:     {dict(reads)}")
        L.append(f"   latency: avg {round(sum(lat)/len(lat),2) if lat else None}s p50 {_pct(lat,50)}s "
                 f"p90 {_pct(lat,90)}s max {max(lat) if lat else None}s  >5s {sum(1 for x in lat if x>5)}")
        L.append("")
    pdist = Counter(panel_verdicts)
    L.append(f"--- FAIL-CLOSED PANEL (10 paired reps) ---")
    L.append(f"   panel verdicts: {dict(pdist)}")
    L.append(f"   -> the panel sends this label to REVIEW on "
             f"{sum(v for k, v in pdist.items() if 'review' in k)}/{len(panel_verdicts)} reps; "
             f"PASS on {pdist.get('PASS', 0)}/{len(panel_verdicts)}")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"stability_single_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump({"image": IMAGE,
                   "per_model": {m: [{"verdict": r["verdict"], "dt": r["dt"]} for r in per_model[m]] for m in MODELS},
                   "panel": panel_verdicts}, fh, indent=2, ensure_ascii=False)
    print(f"Written {os.path.relpath(base, ROOT)}.txt / .json")


if __name__ == "__main__":
    main()
