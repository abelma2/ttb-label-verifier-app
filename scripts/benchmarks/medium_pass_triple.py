"""Triple gate, HIGH-only PASS  vs  MEDIUM-pass PASS  -- the same 3 reads scored by BOTH gates, on
COMPLIANT labels (baseline+clearer) AND bold_safety VIOLATIONS, 3x. No production code touched.

The only delta between the two gates is the PASS threshold on the MAIN (gpt-5.4-mini:A) read:
  decide_high  (validated triple gate, = TG.decide): PASS needs main hb=True/HIGH and bb=False/HIGH.
  decide_medium (this proposal): PASS also accepts main hb/bb at MEDIUM confidence. FAIL logic and the
                specialist veto are UNCHANGED -- so a medium-confidence clean main read moves
                REVIEW->PASS *only if neither specialist flags body-bold and no header disagreement*.

Question: medium-pass should raise the COMPLIANT pass rate (fewer needless reviews of medium reads)
WITHOUT leaking violations, because the 2 gpt-4.1+S specialists backstop it. The risk is asymmetric:
  - boldbody (body-bold) is backstopped by the S BODY-BOLD veto (strong: 15/15 prior).
  - notbold (header-bold) loses the medium safety margin and relies ONLY on the S HEADER cross-check.
This run measures both: throughput gain on compliant + bold_compliant, and any NEW false-pass on
boldbody/notbold under medium-pass.

Usage: python scripts/benchmarks/medium_pass_triple.py
Writes output/medium_pass_triple_<ts>.{txt,json}.
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
import triple_gate_compliant as TG       # _run_reads, _vals, decide (=decide_high)

REPS = 3
COMPLIANT_FOLDERS = ["baseline_labels", "clearer_baseline_labels"]
VIOLATIONS = ("boldbody", "notbold")
_OK_MED = ("high", "medium")


def decide_high(main, s1, s2):
    return TG.decide(main, s1, s2)


def decide_medium(main, s1, s2):
    """Identical to the triple gate, except PASS accepts MEDIUM main confidence too."""
    if not main or not s1 or not s2:
        return "REVIEW", ["timeout/error"]
    hb_m, bb_m, hbc_m, bbc_m, leg_m = TG._vals(main)
    bb_s1, bb_s2 = B._eff_body_bold(s1), B._eff_body_bold(s2)
    hb_s1, hb_s2 = B._eff_header_bold(s1), B._eff_header_bold(s2)
    # FAIL: inert (needs 2 independent non-S/crop body-bold high; only main is non-S -> never 2)
    if sum(1 for (bb, bbc) in [(bb_m, bbc_m)] if bb is True and bbc == "high") >= 2:
        return "FAIL", ["two-independent-nonS-body-bold"]
    reasons = []
    if bb_s1 is True or bb_s2 is True:
        reasons.append("S body-bold veto")
    if bb_m is True:                                  # any main body-bold blocks PASS
        reasons.append("main body-bold -> review")
    if hb_m is True and (hb_s1 is False or hb_s2 is False):
        reasons.append("header disagreement (S says header not bold)")
    if leg_m in ("poor", "limited"):
        reasons.append("limited/poor legibility")
    main_clean_med = (hb_m is True and hbc_m in _OK_MED and bb_m is False and bbc_m in _OK_MED
                      and leg_m == "good")
    if not main_clean_med and not reasons:
        reasons.append("main not clean (even at medium)")
    if reasons:
        return "REVIEW", reasons
    return "PASS", ["main clean@medium + both S no body-bold"]


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def _gather_compliant():
    out = []
    for folder in COMPLIANT_FOLDERS:
        d = os.path.join(ROOT, "test_labels", folder)
        for fname in sorted(os.listdir(d)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                out.append((folder, fname, os.path.join(d, fname), "_other" in fname.lower()))
    return out


def _run_compliant():
    images = _gather_compliant()
    tally = {"high": defaultdict(int), "medium": defaultdict(int)}   # verdict -> n (backs)
    by_folder = {g: {"high": defaultdict(int), "medium": defaultdict(int)} for g in COMPLIANT_FOLDERS}
    fronts = {"n": 0, "ok": 0, "hallucinated": 0}
    converted, walls, rows = [], [], []
    print("=== COMPLIANT (baseline + clearer), 3x ===")
    for folder, fname, path, is_back in images:
        for rep in range(1, REPS + 1):
            reads = TG._run_reads(path)
            m_f, m_dt, _ = reads["main"]
            s1_f, s1_dt, _ = reads["specialist_1"]
            s2_f, s2_dt, _ = reads["specialist_2"]
            dts = [d for d in (m_dt, s1_dt, s2_dt) if d is not None]
            if dts:
                walls.append(max(dts))
            if is_back:
                vh, _rh = decide_high(m_f, s1_f, s2_f)
                vm, _rm = decide_medium(m_f, s1_f, s2_f)
                tally["high"][vh] += 1
                tally["medium"][vm] += 1
                by_folder[folder]["high"][vh] += 1
                by_folder[folder]["medium"][vm] += 1
                conv = vh == "REVIEW" and vm == "PASS"
                if conv:
                    converted.append(f"{folder}/{fname} r{rep}")
                rows.append({"img": f"{folder}/{fname}", "rep": rep, "high": vh, "medium": vm,
                             "converted_review_to_pass": conv,
                             "main": (None if not m_f else list(TG._vals(m_f))),
                             "s1_bb": (None if not s1_f else B._eff_body_bold(s1_f)),
                             "s2_bb": (None if not s2_f else B._eff_body_bold(s2_f))})
                mv = "ERR" if not m_f else f"hb={TG._vals(m_f)[0]}/{(TG._vals(m_f)[2] or '-')[:1]} bb={TG._vals(m_f)[1]}/{(TG._vals(m_f)[3] or '-')[:1]}"
                tag = "  REVIEW->PASS" if conv else ""
                print(f"  {folder[:5]}/{fname:24s} r{rep} main({mv}) high={vh:7s} medium={vm:7s}{tag}")
            else:
                wp = any((r[0] or {}).get("warning_present") is True for r in
                         (reads["main"], reads["specialist_1"], reads["specialist_2"]))
                fronts["n"] += 1
                fronts["hallucinated" if wp else "ok"] += 1
    return {"tally": {g: dict(v) for g, v in tally.items()},
            "by_folder": {f: {g: dict(v) for g, v in d.items()} for f, d in by_folder.items()},
            "fronts": fronts, "converted_review_to_pass": converted,
            "wall_p50": _pct(walls, 50), "wall_max": max(walls) if walls else None,
            "over_5s": sum(1 for x in walls if x > 5), "rows": rows}


def _score_violation(variant, verdict):
    if variant in VIOLATIONS:
        return "FALSE-PASS" if verdict == "PASS" else "caught"
    if variant == "bold_compliant":
        return "correct-PASS" if verdict == "PASS" else ("over-review" if verdict == "REVIEW" else "FALSE-FAIL")
    return "obs"


def _run_boldsafety():
    images = B._bs_images()
    th = defaultdict(lambda: defaultdict(int))   # variant -> outcome -> n  (high gate)
    tm = defaultdict(lambda: defaultdict(int))   # variant -> outcome -> n  (medium gate)
    new_false_pass, walls, rows = [], [], []
    print("\n=== bold_safety VIOLATIONS, 3x ===")
    for im in images:
        var = im["variant"]
        for rep in range(1, REPS + 1):
            reads = TG._run_reads(im["path"])
            m_f, m_dt, _ = reads["main"]
            s1_f, s1_dt, _ = reads["specialist_1"]
            s2_f, s2_dt, _ = reads["specialist_2"]
            dts = [d for d in (m_dt, s1_dt, s2_dt) if d is not None]
            if dts:
                walls.append(max(dts))
            vh, _rh = decide_high(m_f, s1_f, s2_f)
            vm, rm = decide_medium(m_f, s1_f, s2_f)
            oh, om = _score_violation(var, vh), _score_violation(var, vm)
            th[var][oh] += 1
            tm[var][om] += 1
            leaked = var in VIOLATIONS and oh == "caught" and om == "FALSE-PASS"
            if leaked:
                new_false_pass.append(f"{im['name']} r{rep}")
            rows.append({"img": im["name"], "variant": var, "rep": rep, "high": vh, "medium": vm,
                         "high_outcome": oh, "medium_outcome": om, "medium_NEW_leak": leaked,
                         "medium_reasons": rm,
                         "main": (None if not m_f else list(TG._vals(m_f))),
                         "s1_bb": (None if not s1_f else B._eff_body_bold(s1_f)),
                         "s2_bb": (None if not s2_f else B._eff_body_bold(s2_f))})
            tag = "  <-- NEW LEAK (medium false-pass)" if leaked else ""
            mh = "ERR" if not m_f else f"hb={TG._vals(m_f)[0]}/{(TG._vals(m_f)[2] or '-')[:1]} bb={TG._vals(m_f)[1]}/{(TG._vals(m_f)[3] or '-')[:1]}"
            print(f"  {im['name']:26s} r{rep} [{var:14s}] main({mh}) high={vh:7s} medium={vm:7s}{tag}")
    return {"high": {k: dict(v) for k, v in th.items()}, "medium": {k: dict(v) for k, v in tm.items()},
            "new_false_pass_under_medium": new_false_pass,
            "wall_p50": _pct(walls, 50), "wall_max": max(walls) if walls else None,
            "over_5s": sum(1 for x in walls if x > 5), "rows": rows}


def main():
    print(f"HIGH-only triple gate  vs  MEDIUM-pass triple gate  (same reads, both scored)  reps={REPS}\n")
    comp = _run_compliant()
    viol = _run_boldsafety()
    report = {"compliant": comp, "bold_safety": viol}
    _summary(report)
    _write(report)


def _summary(r):
    c, v = r["compliant"], r["bold_safety"]
    print("\n  ===== SUMMARY =====")
    print("  COMPLIANT backs (PASS=correct, FAIL=false-fail, REVIEW=over-caution):")
    for g in ("high", "medium"):
        t = c["tally"][g]
        n = sum(t.values())
        print(f"    {g:7s}: PASS {t.get('PASS',0)}/{n}  FAIL {t.get('FAIL',0)}  REVIEW {t.get('REVIEW',0)}")
    print(f"    medium converted REVIEW->PASS: {len(c['converted_review_to_pass'])}  "
          f"{c['converted_review_to_pass']}")
    print("\n  bold_safety (CAUGHT good; FALSE-PASS = leaked violation):")
    for var in ("bold_compliant", "boldbody", "notbold"):
        h, m = v["high"].get(var, {}), v["medium"].get(var, {})
        if var == "bold_compliant":
            print(f"    {var:14s} high: PASS {h.get('correct-PASS',0)} review {h.get('over-review',0)}  |  "
                  f"medium: PASS {m.get('correct-PASS',0)} review {m.get('over-review',0)}")
        else:
            hn = sum(h.values()); mn = sum(m.values())
            print(f"    {var:14s} high: caught {h.get('caught',0)}/{hn} false-pass {h.get('FALSE-PASS',0)}  |  "
                  f"medium: caught {m.get('caught',0)}/{mn} FALSE-PASS {m.get('FALSE-PASS',0)}")
    print(f"    *** NEW violation leaks under MEDIUM: {len(v['new_false_pass_under_medium'])} "
          f"{v['new_false_pass_under_medium']} ***")
    print(f"\n  latency p50: compliant {c['wall_p50']}s / bold_safety {v['wall_p50']}s  "
          f"(max {c['wall_max']}/{v['wall_max']})\n")


def _write(r):
    c, v = r["compliant"], r["bold_safety"]
    L = ["", "=" * 104, "MEDIUM-PASS triple gate vs HIGH-only triple gate (same reads), 3x", "=" * 104,
         "delta = PASS gate accepts MEDIUM main confidence (FAIL + specialist veto unchanged). "
         "compliant: medium should add PASSES (never a false-fail, FAIL is inert). bold_safety: watch "
         "for NEW false-passes under medium -- esp. notbold (loses the medium safety margin, relies on "
         "the S header cross-check).", ""]
    L.append("--- COMPLIANT backs (PASS=correct, FAIL=false-fail, REVIEW=over-caution) ---")
    for g in ("high", "medium"):
        t = c["tally"][g]; n = sum(t.values())
        L.append(f"   {g:7s}: PASS {t.get('PASS',0)}/{n}   FAIL {t.get('FAIL',0)}   REVIEW {t.get('REVIEW',0)}")
    L.append(f"   medium REVIEW->PASS conversions: {len(c['converted_review_to_pass'])}")
    for x in c["converted_review_to_pass"]:
        L.append(f"      + {x}")
    L.append(f"   per-folder: {c['by_folder']}")
    L.append(f"   fronts no-warning OK: {c['fronts']['ok']}/{c['fronts']['n']}")
    L.append("")
    L.append("--- bold_safety (CAUGHT=routed to review; FALSE-PASS=leaked) ---")
    for var in ("bold_compliant", "boldbody", "notbold", "titlecase"):
        h, m = v["high"].get(var, {}), v["medium"].get(var, {})
        L.append(f"   {var:14s} HIGH {dict(h)}   MEDIUM {dict(m)}")
    L.append(f"   *** NEW violation leaks under MEDIUM: {len(v['new_false_pass_under_medium'])} "
             f"{v['new_false_pass_under_medium']} ***")
    L.append("")
    L.append(f"latency p50 compliant {c['wall_p50']}s (max {c['wall_max']}, >5s {c['over_5s']})  |  "
             f"bold_safety {v['wall_p50']}s (max {v['wall_max']}, >5s {v['over_5s']})")
    L.append("")
    L.append("compliant per-image-rep (REVIEW->PASS marked):")
    for row in c["rows"]:
        mv = row["main"]
        mvs = "ERR" if mv is None else f"hb={mv[0]}/{(mv[2] or '-')[:1]} bb={mv[1]}/{(mv[3] or '-')[:1]} leg={mv[4]}"
        conv = "  REVIEW->PASS" if row["converted_review_to_pass"] else ""
        L.append(f"   {row['img']:42s} r{row['rep']} main({mvs}) S1.bb={row['s1_bb']} S2.bb={row['s2_bb']} "
                 f"high={row['high']:7s} medium={row['medium']:7s}{conv}")
    L.append("")
    L.append("bold_safety per-image-rep (NEW LEAK marked):")
    for row in v["rows"]:
        mv = row["main"]
        mvs = "ERR" if mv is None else f"hb={mv[0]}/{(mv[2] or '-')[:1]} bb={mv[1]}/{(mv[3] or '-')[:1]}"
        leak = "  <-- NEW LEAK" if row["medium_NEW_leak"] else ""
        L.append(f"   {row['img']:26s} r{row['rep']} [{row['variant']:14s}] main({mvs}) "
                 f"S1.bb={row['s1_bb']} S2.bb={row['s2_bb']} high={row['high']:7s} "
                 f"medium={row['medium']:7s} ({row['high_outcome']}/{row['medium_outcome']}){leak}")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"medium_pass_triple_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(report := r, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written {os.path.relpath(base, ROOT)}.txt / .json")


if __name__ == "__main__":
    main()
