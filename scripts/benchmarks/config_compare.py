"""Compare a single main read vs two fail-closed panels on CLEAN clearer_baseline_labels, 3x.

Configs (each a list of witnesses; panel = fail-closed agree->verdict, disagree->review):
  1. main                    = [gpt-5.4-mini:A]               (single read)
  2. main + gpt-5.4-mini:N    = [gpt-5.4-mini:A, gpt-5.4-mini:N]  (same-model panel)
  3. gpt-5.4-mini:A + gpt-4.1:N = [gpt-5.4-mini:A, gpt-4.1:N]      (cross-model panel)

These labels are COMPLIANT (bold header, non-bold body on the back; no warning on the front), so
ACCURACY = correct compliant-PASS on the 3 backs (a FAIL = false-fail; a review = over-caution);
fronts: correct = no-warning. LATENCY for a panel = the PARALLEL wall (max of its witness calls).

Reuses bold_prompt_safety prompts A/N + _call. No production code touched.
Usage: python scripts/benchmarks/config_compare.py
Writes output/config_compare_<ts>.{txt,json}.
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

DIR = os.path.join(ROOT, "test_labels", "clearer_baseline_labels")
IMAGES = [f"clear_baseline_{n}_{side}.png" for n in (1, 2, 3) for side in ("Front", "Other")]
REPS = 3
CONFIGS = [
    {"name": "main (5.4-mini:A)", "witnesses": [("gpt-5.4-mini", "A")]},
    {"name": "main + 5.4-mini:N", "witnesses": [("gpt-5.4-mini", "A"), ("gpt-5.4-mini", "N")]},
    {"name": "5.4-mini:A + 4.1:N", "witnesses": [("gpt-5.4-mini", "A"), ("gpt-4.1", "N")]},
]


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


def _combine(verdicts):
    if len(verdicts) == 1:
        return verdicts[0]
    return verdicts[0] if verdicts[0] == verdicts[1] else "review (disagreement)"


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def main():
    print(f"configs={[c['name'] for c in CONFIGS]}  images={len(IMAGES)}  reps={REPS}\n")
    report = {}
    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"=== {name} ===")
        walls, back_verdicts, front_ok = [], [], 0
        n_back = n_front = 0
        per_image = {}
        for fname in IMAGES:
            path = os.path.join(DIR, fname)
            is_back = fname.endswith("Other.png")
            cells = []
            for rep in range(1, REPS + 1):
                vs, times = [], []
                for (m, pv) in cfg["witnesses"]:
                    f, dt, _r, _e = B._call(m, B._prompt(pv), [path])
                    vs.append(_verdict(f) if f else "ERR")
                    if dt is not None:
                        times.append(dt)
                verdict = _combine(vs)
                wall = max(times) if times else None   # parallel wall = slowest witness
                if wall is not None:
                    walls.append(wall)
                if is_back:
                    n_back += 1
                    back_verdicts.append(verdict)
                else:
                    n_front += 1
                    front_ok += int(verdict == "no-warning")
                cells.append(f"{verdict}{'' if len(vs)==1 else ' '+str(vs)}")
            per_image[fname] = cells
            print(f"  {fname:26s} " + "  |  ".join(cells))
        bp = Counter(back_verdicts)
        clean_pass = bp.get("PASS", 0)
        false_fail = sum(v for k, v in bp.items() if "FAIL" in k)
        review = sum(v for k, v in bp.items() if "review" in k)
        report[name] = {
            "witnesses": cfg["witnesses"], "n_back": n_back, "n_front": n_front,
            "clean_pass": clean_pass, "false_fail": false_fail, "review": review,
            "back_verdicts": dict(bp), "front_no_warning_ok": front_ok,
            "wall_mean": round(sum(walls) / len(walls), 2) if walls else None,
            "wall_p50": _pct(walls, 50), "wall_max": max(walls) if walls else None,
            "over_5s": sum(1 for x in walls if x > 5), "per_image": per_image,
        }
        r = report[name]
        print(f"  -> backs: clean-PASS {clean_pass}/{n_back}, false-fail {false_fail}, review {review}  |  "
              f"fronts no-warning {front_ok}/{n_front}  |  wall mean {r['wall_mean']}s p50 {r['wall_p50']}s "
              f"max {r['wall_max']}s (>5s {r['over_5s']})\n")
    _write(report)


def _write(report):
    L = ["", "=" * 100, "CONFIG COMPARE on clearer_baseline_labels (CLEAN/compliant), 3x -- ACCURACY + TIME",
         "=" * 100,
         "backs are compliant -> clean-PASS is correct; FAIL = false-fail; review = over-caution. "
         "panel latency = parallel wall (max of witnesses).", ""]
    L.append(f"{'config':22s} {'clean-PASS':12s} {'false-fail':11s} {'review':8s} {'frontOK':9s} "
             f"{'wall avg':9s} {'p50':6s} {'max':6s} {'>5s':4s}")
    L.append("-" * 96)
    for name, r in report.items():
        cp = f"{r['clean_pass']}/{r['n_back']}"
        fo = f"{r['front_no_warning_ok']}/{r['n_front']}"
        L.append(f"{name:22s} {cp:12s} {str(r['false_fail']):11s} {str(r['review']):8s} {fo:9s} "
                 f"{str(r['wall_mean'])+'s':9s} {str(r['wall_p50']):6s} {str(r['wall_max']):6s} {str(r['over_5s']):4s}")
    L.append("")
    for name, r in report.items():
        L.append(f"--- {name}  (back verdicts: {r['back_verdicts']}) ---")
        for f, cells in r["per_image"].items():
            L.append(f"     {f:26s} " + "  |  ".join(cells))
        L.append("")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"config_compare_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written {os.path.relpath(base, ROOT)}.txt / .json")


if __name__ == "__main__":
    main()
