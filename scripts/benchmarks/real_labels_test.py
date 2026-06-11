"""Real-world test: gpt-4.1 + N and gpt-5.4-mini + N on the photographed real_labels, ONCE each.

These are real bottle photos with NO font-controlled ground truth, so we cannot score
correct/incorrect. What we CAN observe -- and what actually matters for the recommended
architecture -- is:
  - each model's per-warning verdict (PASS / FAIL-body-bold / FAIL-header-not-bold / review),
  - AGREEMENT vs DISAGREEMENT between the two models  (a fail-closed witness panel routes any
    disagreement to needs_review, never auto-resolves it -- so the disagreement rate IS the
    panel's real-world human-review rate),
  - body-bold flags (a potential real violation OR a false-flag -- unknown without ground truth),
  - latency on real photographed labels.

Reuses the Stage-1 prompt N + normalized schema + _call. No production code touched. 1 rep.
Usage: python scripts/benchmarks/real_labels_test.py
Writes output/real_labels_test_<ts>.{txt,json}.
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

import bold_prompt_safety as B   # _prompt, _call, _eff_header_bold, _eff_body_bold, key load

REAL = os.path.join(ROOT, "test_labels", "real_labels")
MODELS = ["gpt-4.1", "gpt-5.4-mini"]
PROMPT = "N"
IMAGES = [f"test_{n}_{side}.jpeg" for n in range(1, 14) for side in ("Front", "Other")]


def _verdict(f):
    """Per-read warning-bold verdict, fail-closed (mirrors the production gate's spirit)."""
    if not f or not f.get("warning_present"):
        return "no-warning"
    hb, bb = B._eff_header_bold(f), B._eff_body_bold(f)
    hbc, bbc = f.get("header_bold_confidence"), f.get("body_bold_confidence")
    if f.get("legibility") == "poor":
        return "review"
    if bb is True and bbc == "high":
        return "FAIL-body-bold"
    if hb is False and hbc == "high":
        return "FAIL-header-not-bold"
    if hb is True and bb is False and hbc == "high" and bbc == "high":
        return "PASS"
    return "review"


def _panel(v1, v2):
    """Fail-closed witness panel: agree -> that verdict; disagree -> needs_review (never
    auto-resolved, never majority-voted to PASS)."""
    if v1 == v2:
        return v1
    if "no-warning" in (v1, v2) and ("PASS" in (v1, v2) or "FAIL" in v1 + v2):
        return "review (presence disagreement)"
    return "review (witness disagreement)"


def main():
    print(f"models={MODELS} prompt={PROMPT}  images={len(IMAGES)}  reps=1\n")
    rows, lat = [], {m: [] for m in MODELS}
    for fname in IMAGES:
        path = os.path.join(REAL, fname)
        reads = {}
        for m in MODELS:
            fields, dt, retries, err = B._call(m, B._prompt(PROMPT), [path])
            if dt is not None:
                lat[m].append(dt)
            reads[m] = {"fields": fields, "dt": dt, "err": err,
                        "verdict": _verdict(fields) if fields else "ERROR"}
        v41, v54 = reads["gpt-4.1"]["verdict"], reads["gpt-5.4-mini"]["verdict"]
        panel = _panel(v41, v54)
        rows.append({"image": fname, "v_gpt41": v41, "v_gpt54mini": v54, "panel": panel,
                     "agree": v41 == v54, "reads": reads})
        f41, f54 = reads["gpt-4.1"]["fields"] or {}, reads["gpt-5.4-mini"]["fields"] or {}
        print(f"  {fname:22s}  4.1={v41:22s} 5.4mini={v54:22s} -> panel: {panel}"
              f"   [4.1 hb={B._eff_header_bold(f41)}/bb={B._eff_body_bold(f41)} | "
              f"5.4 hb={B._eff_header_bold(f54)}/bb={B._eff_body_bold(f54)}]")
    _write(rows, lat)


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def _write(rows, lat):
    warn_rows = [r for r in rows if not (r["v_gpt41"] == "no-warning" and r["v_gpt54mini"] == "no-warning")]
    agree = sum(1 for r in warn_rows if r["agree"])
    disagree = len(warn_rows) - agree
    panel_dist = Counter(r["panel"] for r in warn_rows)
    v41 = Counter(r["v_gpt41"] for r in warn_rows)
    v54 = Counter(r["v_gpt54mini"] for r in warn_rows)
    bb41 = sum(1 for r in warn_rows if (r["reads"]["gpt-4.1"]["fields"] or {}) and
               B._eff_body_bold(r["reads"]["gpt-4.1"]["fields"]) is True)
    bb54 = sum(1 for r in warn_rows if (r["reads"]["gpt-5.4-mini"]["fields"] or {}) and
               B._eff_body_bold(r["reads"]["gpt-5.4-mini"]["fields"]) is True)

    L = ["", "=" * 100, "REAL-LABELS TEST -- gpt-4.1+N vs gpt-5.4-mini+N on photographed bottles (1x, NO ground truth)",
         "=" * 100,
         "No font-controlled truth here, so reads are NOT scored correct/incorrect. The signal is "
         "AGREEMENT vs DISAGREEMENT:",
         "a fail-closed witness panel routes every DISAGREEMENT to needs_review (never auto-resolved). "
         "That disagreement rate = the panel's real human-review load.", "",
         f"warning-bearing images (>=1 model saw a warning): {len(warn_rows)} of {len(rows)}",
         f"AGREE: {agree}/{len(warn_rows)}   DISAGREE (-> panel review): {disagree}/{len(warn_rows)}",
         f"gpt-4.1 verdicts:      {dict(v41)}",
         f"gpt-5.4-mini verdicts: {dict(v54)}",
         f"body-bold flags (potential violation OR false-flag): gpt-4.1={bb41}, gpt-5.4-mini={bb54}",
         f"FAIL-CLOSED PANEL verdicts: {dict(panel_dist)}", "",
         f"latency: gpt-4.1 avg {round(sum(lat['gpt-4.1'])/len(lat['gpt-4.1']),2) if lat['gpt-4.1'] else None}s "
         f"p50 {_pct(lat['gpt-4.1'],50)}s p90 {_pct(lat['gpt-4.1'],90)}s max {max(lat['gpt-4.1']) if lat['gpt-4.1'] else None}s "
         f">5s {sum(1 for x in lat['gpt-4.1'] if x>5)}",
         f"         gpt-5.4-mini avg {round(sum(lat['gpt-5.4-mini'])/len(lat['gpt-5.4-mini']),2) if lat['gpt-5.4-mini'] else None}s "
         f"p50 {_pct(lat['gpt-5.4-mini'],50)}s p90 {_pct(lat['gpt-5.4-mini'],90)}s max {max(lat['gpt-5.4-mini']) if lat['gpt-5.4-mini'] else None}s "
         f">5s {sum(1 for x in lat['gpt-5.4-mini'] if x>5)}", ""]
    L.append("--- per image ---")
    L.append(f"   {'image':22s} {'gpt-4.1+N':24s} {'gpt-5.4-mini+N':24s} {'panel':28s} {'agree':5s}")
    for r in rows:
        L.append(f"   {r['image']:22s} {r['v_gpt41']:24s} {r['v_gpt54mini']:24s} {r['panel']:28s} "
                 f"{'yes' if r['agree'] else 'NO':5s}")
    L.append("")
    L.append("--- DISAGREEMENTS (these are the panel's human-review cases) ---")
    for r in warn_rows:
        if not r["agree"]:
            f41 = r["reads"]["gpt-4.1"]["fields"] or {}
            f54 = r["reads"]["gpt-5.4-mini"]["fields"] or {}
            L.append(f"   {r['image']:22s} 4.1: hb={B._eff_header_bold(f41)}/{f41.get('header_bold_confidence')} "
                     f"bb={B._eff_body_bold(f41)}/{f41.get('body_bold_confidence')} basis={f41.get('short_basis')!r}")
            L.append(f"   {'':22s} 5.4: hb={B._eff_header_bold(f54)}/{f54.get('header_bold_confidence')} "
                     f"bb={B._eff_body_bold(f54)}/{f54.get('body_bold_confidence')} basis={f54.get('short_basis')!r}")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"real_labels_test_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump({"rows": [{k: v for k, v in r.items() if k != "reads"} for r in rows]},
                  fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written {os.path.relpath(base, ROOT)}.txt / .json")


if __name__ == "__main__":
    main()
