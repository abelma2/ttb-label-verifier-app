"""Once-through on bold_safety with the two winners AND a fail-closed witness panel.

bold_safety HAS font-controlled ground truth (manifest), so unlike the real-label run we can
score correctness AND test the panel hypothesis directly:
  - does a FAIL-CLOSED panel of gpt-4.1+N and gpt-5.4-mini+N catch the bold VIOLATIONS that each
    model misses alone (boldbody = body bold; notbold = header not bold)?
  - and does it over-review the bold-COMPLIANT typography (bold_compliant + titlecase)?

Panel rule: the two reads AGREE -> that verdict; DISAGREE -> needs_review (never auto-resolved,
never voted to PASS). A violation is "caught" when its verdict is anything but PASS.

Reuses Stage-1 prompt N + normalized schema + _call. No production code touched. 1 rep.
Usage: python scripts/benchmarks/panel_bold_safety.py
Writes output/panel_bold_safety_<ts>.{txt,json}.
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
MODELS = ["gpt-4.1", "gpt-5.4-mini"]
PROMPT = "N"
VIOLATION = {"boldbody", "notbold"}        # bold violations: must NOT be PASSed
COMPLIANT = {"bold_compliant", "titlecase"}  # bold-typography compliant: header bold, body not


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
    print(f"models={MODELS} prompt={PROMPT}  images={len(images)}  reps=1\n")
    rows, lat = [], {m: [] for m in MODELS}
    for fname, gt in images:
        path = os.path.join(BS, fname)
        reads = {}
        for m in MODELS:
            fields, dt, retries, err = B._call(m, B._prompt(PROMPT), [path])
            if dt is not None:
                lat[m].append(dt)
            reads[m] = {"fields": fields, "verdict": _verdict(fields) if fields else "ERROR"}
        v41, v54 = reads["gpt-4.1"]["verdict"], reads["gpt-5.4-mini"]["verdict"]
        panel = _panel(v41, v54)
        rows.append({"image": fname, "variant": gt["variant"], "v41": v41, "v54": v54,
                     "panel": panel, "agree": v41 == v54, "reads": reads})
        print(f"  {fname:26s} [{gt['variant']:14s}]  4.1={v41:16s} 5.4={v54:16s} -> {panel}")
    _write(rows, lat)


def _caught(verdict):
    return verdict not in ("PASS", "no-warning")   # violation routed away from PASS = caught


def _hi_false_pass(reads, variant):
    """A model PASSED a violation at high confidence (the dangerous case)."""
    f = reads["fields"] or {}
    if variant == "boldbody":
        return B._eff_body_bold(f) is False and f.get("body_bold_confidence") == "high" and f.get("warning_present")
    if variant == "notbold":
        return B._eff_header_bold(f) is True and f.get("header_bold_confidence") == "high" and f.get("warning_present")
    return False


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def _write(rows, lat):
    viol = [r for r in rows if r["variant"] in VIOLATION]
    comp = [r for r in rows if r["variant"] in COMPLIANT]

    def model_catch(rs, mk):
        return sum(1 for r in rs if _caught(r["reads"][mk]["verdict"]))

    def model_hifp(rs, mk):
        return sum(1 for r in rs if _hi_false_pass(r["reads"][mk], r["variant"]))

    panel_catch = sum(1 for r in viol if _caught(r["panel"]))
    panel_clean = sum(1 for r in comp if r["panel"] == "PASS")
    agree = sum(1 for r in rows if r["agree"])

    L = ["", "=" * 100, "PANEL ON bold_safety -- gpt-4.1+N & gpt-5.4-mini+N + fail-closed panel (1x, GROUND TRUTH)",
         "=" * 100,
         "VIOLATIONS (boldbody=body bold, notbold=header not bold): must NOT be PASSed -> 'caught' = "
         "verdict != PASS.",
         "COMPLIANT typography (bold_compliant, titlecase: bold header + non-bold body): want PASS, "
         "not over-review.",
         "Panel: reads AGREE -> that verdict; DISAGREE -> needs_review (fail-closed, no vote-to-PASS).", "",
         f"--- VIOLATIONS CAUGHT (of {len(viol)}) ---",
         f"   gpt-4.1+N alone:      {model_catch(viol,'gpt-4.1')}/{len(viol)}   "
         f"(high-conf FALSE-PASS: {model_hifp(viol,'gpt-4.1')})",
         f"   gpt-5.4-mini+N alone: {model_catch(viol,'gpt-5.4-mini')}/{len(viol)}   "
         f"(high-conf FALSE-PASS: {model_hifp(viol,'gpt-5.4-mini')})",
         f"   FAIL-CLOSED PANEL:    {panel_catch}/{len(viol)}   <-- does combining beat either alone?", "",
         f"--- COMPLIANT TYPOGRAPHY clean-PASS (of {len(comp)}) ---",
         f"   gpt-4.1+N alone:      {sum(1 for r in comp if r['reads']['gpt-4.1']['verdict']=='PASS')}/{len(comp)}",
         f"   gpt-5.4-mini+N alone: {sum(1 for r in comp if r['reads']['gpt-5.4-mini']['verdict']=='PASS')}/{len(comp)}",
         f"   FAIL-CLOSED PANEL:    {panel_clean}/{len(comp)}   (lower = more false-reviews on clean labels)", "",
         f"agreement: {agree}/{len(rows)}   panel verdicts: {dict(Counter(r['panel'] for r in rows))}",
         f"latency: gpt-4.1 avg {round(sum(lat['gpt-4.1'])/len(lat['gpt-4.1']),2) if lat['gpt-4.1'] else None}s "
         f"max {max(lat['gpt-4.1']) if lat['gpt-4.1'] else None}s >5s {sum(1 for x in lat['gpt-4.1'] if x>5)}  |  "
         f"gpt-5.4-mini avg {round(sum(lat['gpt-5.4-mini'])/len(lat['gpt-5.4-mini']),2) if lat['gpt-5.4-mini'] else None}s "
         f"max {max(lat['gpt-5.4-mini']) if lat['gpt-5.4-mini'] else None}s >5s {sum(1 for x in lat['gpt-5.4-mini'] if x>5)}", ""]
    L.append(f"   {'image':26s} {'variant':14s} {'gpt-4.1+N':16s} {'gpt-5.4-mini+N':16s} {'PANEL':22s}")
    for r in rows:
        L.append(f"   {r['image']:26s} {r['variant']:14s} {r['v41']:16s} {r['v54']:16s} {r['panel']:22s}")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"panel_bold_safety_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump({"rows": [{k: v for k, v in r.items() if k != "reads"} for r in rows]},
                  fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written {os.path.relpath(base, ROOT)}.txt / .json")


if __name__ == "__main__":
    main()
