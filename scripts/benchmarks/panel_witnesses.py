"""General two-witness fail-closed panel on bold_safety, with an ERROR-CORRELATION metric.

Answers: does pairing two witnesses actually decorrelate their errors? The decisive number is,
of the bold VIOLATIONS, on how many do BOTH witnesses make the SAME miss (both read it as
compliant) -- those are the correlated dangerous errors the fail-closed panel CANNOT catch.

  panel catches a violation  <=>  at least one witness does NOT PASS it.
  panel MISSES a violation    <=>  BOTH witnesses PASS it (correlated miss).

Use it to compare a SAME-model/two-prompt panel (gpt-5.4-mini:A + gpt-5.4-mini:N) against the
CROSS-model panel (gpt-4.1:N + gpt-5.4-mini:N).

Usage:
  python scripts/benchmarks/panel_witnesses.py --w1 gpt-5.4-mini:A --w2 gpt-5.4-mini:N
  python scripts/benchmarks/panel_witnesses.py --w1 gpt-4.1:N --w2 gpt-5.4-mini:N
Writes output/panel_witnesses_<ts>.{txt,json}.
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

BS = os.path.join(ROOT, "bold_safety")
VIOLATION = {"boldbody", "notbold"}
COMPLIANT = {"bold_compliant", "titlecase"}


def _arg(flag, default):
    a = sys.argv[1:]
    return a[a.index(flag) + 1] if flag in a and a.index(flag) + 1 < len(a) else default


W1 = tuple(_arg("--w1", "gpt-5.4-mini:A").split(":"))
W2 = tuple(_arg("--w2", "gpt-5.4-mini:N").split(":"))


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


def main():
    with open(os.path.join(BS, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    images = sorted(man.items(), key=lambda kv: kv[0])
    print(f"W1={W1[0]}+{W1[1]}   W2={W2[0]}+{W2[1]}   images={len(images)}\n")
    rows, lat = [], {0: [], 1: []}
    for fname, gt in images:
        path = os.path.join(BS, fname)
        f1, d1, _r, _e = B._call(W1[0], B._prompt(W1[1]), [path])
        f2, d2, _r2, _e2 = B._call(W2[0], B._prompt(W2[1]), [path])
        if d1:
            lat[0].append(d1)
        if d2:
            lat[1].append(d2)
        v1, v2 = _verdict(f1) if f1 else "ERR", _verdict(f2) if f2 else "ERR"
        rows.append({"image": fname, "variant": gt["variant"], "v1": v1, "v2": v2,
                     "panel": _panel(v1, v2)})
        print(f"  {fname:26s} [{gt['variant']:14s}] W1={v1:16s} W2={v2:16s} -> {_panel(v1, v2)}")
    _write(rows, lat)


def _write(rows, lat):
    viol = [r for r in rows if r["variant"] in VIOLATION]
    comp = [r for r in rows if r["variant"] in COMPLIANT]
    w1_caught = sum(1 for r in viol if r["v1"] != "PASS")
    w2_caught = sum(1 for r in viol if r["v2"] != "PASS")
    panel_caught = sum(1 for r in viol if not (r["v1"] == "PASS" and r["v2"] == "PASS"))
    both_missed = [r["image"] for r in viol if r["v1"] == "PASS" and r["v2"] == "PASS"]
    only_one = sum(1 for r in viol if (r["v1"] == "PASS") != (r["v2"] == "PASS"))
    panel_clean = sum(1 for r in comp if r["panel"] == "PASS")

    L = ["", "=" * 100,
         f"TWO-WITNESS PANEL on bold_safety -- W1={W1[0]}+{W1[1]}  W2={W2[0]}+{W2[1]}", "=" * 100,
         "DECISIVE: 'both missed' = violations where BOTH witnesses PASS (correlated error the "
         "fail-closed panel CANNOT catch). Fewer = better decorrelation.", "",
         f"--- VIOLATIONS (of {len(viol)}) ---",
         f"   W1 ({W1[0]}+{W1[1]}) caught:  {w1_caught}/{len(viol)}",
         f"   W2 ({W2[0]}+{W2[1]}) caught:  {w2_caught}/{len(viol)}",
         f"   PANEL caught:               {panel_caught}/{len(viol)}",
         f"   decorrelated (only 1 caught it): {only_one}",
         f"   *** BOTH MISSED (correlated, panel FALSE-PASSES): {len(both_missed)}  {both_missed} ***", "",
         f"--- COMPLIANT clean-PASS (of {len(comp)}): {panel_clean}/{len(comp)} ---",
         f"latency: W1 avg {round(sum(lat[0])/len(lat[0]),2) if lat[0] else None}s  |  "
         f"W2 avg {round(sum(lat[1])/len(lat[1]),2) if lat[1] else None}s", "",
         f"   {'image':26s} {'variant':14s} {'W1':16s} {'W2':16s} {'PANEL':22s}"]
    for r in rows:
        L.append(f"   {r['image']:26s} {r['variant']:14s} {r['v1']:16s} {r['v2']:16s} {r['panel']:22s}")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"panel_witnesses_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump({"w1": W1, "w2": W2, "rows": rows, "both_missed": both_missed}, fh, indent=2, ensure_ascii=False)
    print(f"Written {os.path.relpath(base, ROOT)}.txt / .json")


if __name__ == "__main__":
    main()
