"""Combined neutral+specialist BOLD gate on COMPLIANT labels (baseline + clearer_baseline + real),
1x, with TIME + ACCURACY. No production code touched; reuses bold_prompt_safety prompt A + _call and
prompt_S_test's PROMPT_S verbatim.

Two reads per image, run as a parallel pair (wall = max of the two call latencies):
  NEUTRAL / main extractor : gpt-5.4-mini + main prompt A
  S specialist             : gpt-4.1     + PROMPT_S ("traps")

Decision rule under test (the user's):
  PASS    -> main says header bold + body NOT bold at high conf  AND  S raises no credible
             body-bold concern (S body_bold not True) and S does not contradict the header.
  FAIL    -> any HIGH-confidence body-bold from a NEUTRAL (or crop) read. (No crop read exists,
             so only the neutral/main read can FAIL here. S can never FAIL a label.)
  REVIEW  -> model disagreement, low-res/poor-legibility image, or S-only body-bold.
A missing/errored witness routes to REVIEW (fail-closed).

These labels are COMPLIANT, so on the warning-bearing backs: PASS = correct, FAIL = FALSE-fail
(the cost of trusting an over-flag), REVIEW = over-caution. Fronts: correct = no warning.
Per-image also shows what each read's solo production gate (header_body_gate) would say.

Usage: python scripts/benchmarks/combined_gate_compliant.py
Writes output/combined_gate_compliant_<ts>.{txt,json}.
"""
import json
import os
import sys
from collections import defaultdict
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
import prompt_S_test as PS

NEUTRAL = ("gpt-5.4-mini", "A")          # main extractor (main prompt)
SPEC_MODEL = "gpt-4.1"                    # S specialist
PROMPT_S = PS.PROMPT_S
FOLDERS = ["baseline_labels", "clearer_baseline_labels", "real_labels"]


def _gather():
    out = []
    for folder in FOLDERS:
        d = os.path.join(ROOT, "test_labels", folder)
        for fname in sorted(os.listdir(d)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                out.append((folder, fname, os.path.join(d, fname), "_other" in fname.lower()))
    return out


def _vals(f):
    return (B._eff_header_bold(f), B._eff_body_bold(f),
            f.get("header_bold_confidence"), f.get("body_bold_confidence"), f.get("legibility"))


def _solo_gate(f):
    """What a single read would say under the production header_body_gate."""
    if not f:
        return "ERR"
    if f.get("warning_present") is False:
        return "no-warning"
    hb, bb, hbc, bbc, leg = _vals(f)
    if leg == "poor":
        return "REVIEW"
    if bb is True and bbc == "high":
        return "FAIL"
    if hb is False and hbc == "high":
        return "FAIL"
    if hb is True and bb is False and hbc == "high" and bbc == "high":
        return "PASS"
    return "REVIEW"


def decide(main, spec):
    """The combined neutral+specialist rule. Returns (verdict, reason)."""
    if not main or not spec:
        return "REVIEW", "witness error"
    hb_m, bb_m, hbc_m, bbc_m, leg_m = _vals(main)
    hb_s, bb_s, hbc_s, bbc_s, leg_s = _vals(spec)
    # FAIL: high-confidence body-bold from the NEUTRAL read (crop read: none in this codebase)
    if bb_m is True and bbc_m == "high":
        return "FAIL", "neutral high-conf body-bold"
    # REVIEW: low-res / poor legibility on either read
    if leg_m == "poor" or leg_s == "poor":
        return "REVIEW", "low-res/poor-legibility"
    # REVIEW: S-only body-bold (specialist flags body-bold; neutral did not fail on it)
    if bb_s is True:
        return "REVIEW", "S-only body-bold"
    # PASS vs disagreement vs uncertain
    main_clean = hb_m is True and bb_m is False and hbc_m == "high" and bbc_m == "high"
    spec_contradicts_header = hb_s is False and hbc_s == "high"
    if main_clean and not spec_contradicts_header:
        return "PASS", "main clean + S no concern"
    if main_clean and spec_contradicts_header:
        return "REVIEW", "disagreement (header)"
    return "REVIEW", "uncertain/disagreement"


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def main():
    images = _gather()
    n_back = sum(1 for _, _, _, b in images if b)
    print(f"NEUTRAL={NEUTRAL[0]}:A   SPEC={SPEC_MODEL}+S   images={len(images)} "
          f"({n_back} backs) over {len(FOLDERS)} compliant folders\n")
    backs = {"n": 0, "PASS": 0, "FAIL": 0, "REVIEW": 0}
    fronts = {"n": 0, "no_warning_ok": 0, "hallucinated": 0}
    by_folder = defaultdict(lambda: {"n": 0, "PASS": 0, "FAIL": 0, "REVIEW": 0})
    walls, per_image, reasons = [], [], defaultdict(int)
    np = B._prompt(NEUTRAL[1])
    for folder, fname, path, is_back in images:
        m_f, m_dt, _r, m_e = B._call(NEUTRAL[0], np, [path])
        s_f, s_dt, _r2, s_e = B._call(SPEC_MODEL, PROMPT_S, [path])
        dts = [d for d in (m_dt, s_dt) if d is not None]
        wall = max(dts) if dts else None
        if wall is not None:
            walls.append(wall)
        m_solo, s_solo = _solo_gate(m_f), _solo_gate(s_f)
        if is_back:
            verdict, reason = decide(m_f, s_f)
            backs["n"] += 1
            backs[verdict] += 1
            by_folder[folder]["n"] += 1
            by_folder[folder][verdict] += 1
            reasons[reason] += 1
            tag = verdict
        else:
            wp_m = m_f.get("warning_present") if m_f else None
            wp_s = s_f.get("warning_present") if s_f else None
            fronts["n"] += 1
            if wp_m is True or wp_s is True:
                fronts["hallucinated"] += 1; tag = "HALLUCINATED"; reason = "front warning seen"
            else:
                fronts["no_warning_ok"] += 1; tag = "no-warning-OK"; reason = "-"
            verdict = tag
        wstr = f"{wall:.2f}" if wall is not None else "ERR"
        per_image.append({"img": f"{folder}/{fname}", "verdict": verdict, "reason": reason,
                          "main_solo": m_solo, "spec_solo": s_solo, "wall": wall,
                          "main_dt": m_dt, "spec_dt": s_dt,
                          "main": (None if not m_f else list(_vals(m_f))),
                          "spec": (None if not s_f else list(_vals(s_f)))})
        print(f"  {folder}/{fname:24s} wall={wstr}s  main[{m_solo}] S[{s_solo}] -> {tag}"
              + (f"  ({reason})" if is_back else ""))

    report = {
        "neutral": f"{NEUTRAL[0]}:A", "spec": f"{SPEC_MODEL}+S",
        "backs": backs, "fronts": fronts, "by_folder": {k: dict(v) for k, v in by_folder.items()},
        "review_reasons": dict(reasons),
        "wall_avg": round(sum(walls) / len(walls), 2) if walls else None,
        "wall_p50": _pct(walls, 50), "wall_max": max(walls) if walls else None,
        "over_5s": sum(1 for x in walls if x > 5), "per_image": per_image,
    }
    print(f"\n  -> backs: PASS {backs['PASS']}/{backs['n']}  FALSE-FAIL {backs['FAIL']}  "
          f"review {backs['REVIEW']}  |  fronts no-warning {fronts['no_warning_ok']}/{fronts['n']}  |  "
          f"wall avg {report['wall_avg']}s p50 {report['wall_p50']}s max {report['wall_max']}s "
          f"(>5s {report['over_5s']})\n")
    _write(report)


def _write(report):
    b, f = report["backs"], report["fronts"]
    L = ["", "=" * 100,
         "COMBINED NEUTRAL+SPECIALIST gate on COMPLIANT labels (baseline+clearer+real), 1x",
         "=" * 100,
         f"NEUTRAL = {report['neutral']} (main extractor)   SPEC = {report['spec']} (S 'traps')",
         "rule: PASS = main clean(hdr bold+body not-bold/high) AND S no body-bold concern; "
         "FAIL = neutral high-conf body-bold; REVIEW = disagreement / low-res / S-only body-bold.",
         "labels are COMPLIANT -> PASS = correct, FAIL = FALSE-fail, REVIEW = over-caution.", ""]
    cp = f"{b['PASS']}/{b['n']}"
    fo = f"{f['no_warning_ok']}/{f['n']}"
    L.append(f"{'gate':28s} {'PASS':10s} {'FALSE-FAIL':12s} {'review':8s} {'frontOK':9s} "
             f"{'wall avg':9s} {'p50':6s} {'max':6s} {'>5s':4s}")
    L.append("-" * 96)
    L.append(f"{report['neutral']+' + '+report['spec']:28s} {cp:10s} {str(b['FAIL']):12s} "
             f"{str(b['REVIEW']):8s} {fo:9s} {str(report['wall_avg'])+'s':9s} "
             f"{str(report['wall_p50']):6s} {str(report['wall_max']):6s} {str(report['over_5s']):4s}")
    L.append("")
    L.append(f"per-folder backs: {report['by_folder']}")
    L.append(f"review/verdict reasons (backs): {report['review_reasons']}")
    L.append("")
    L.append("per-image (main solo / S solo = each read's production-gate verdict; wall = parallel max):")
    for pi in report["per_image"]:
        wstr = f"{pi['wall']:.2f}s" if pi["wall"] is not None else "ERR"
        L.append(f"   {pi['img']:42s} wall={wstr:7s} main[{pi['main_solo']:9s}] S[{pi['spec_solo']:9s}] "
                 f"-> {pi['verdict']:7s} ({pi['reason']})")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"combined_gate_compliant_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written {os.path.relpath(base, ROOT)}.txt / .json")


if __name__ == "__main__":
    main()
