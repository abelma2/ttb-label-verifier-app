"""The SAME triple-read gate (main gpt-5.4-mini:A + 2x gpt-4.1+S repeat samples) on bold_safety
VIOLATIONS, 3x, scored against manifest.json ground truth. The recall complement to the compliant
runs. No production code touched; reuses triple_gate_compliant.decide/_run_reads/_solo_gate VERBATIM
(so the gate logic is identical) and bold_prompt_safety for the ground-truth enumeration.

bold_safety variants (5 distortions each = 20 imgs):
  bold_compliant : header bold, body NOT bold  -> correct verdict = PASS (over-review = caution)
  boldbody       : body IS bold (VIOLATION)     -> correct = CAUGHT (REVIEW); PASS = FALSE-PASS (miss)
  notbold        : header NOT bold (VIOLATION)   -> correct = CAUGHT (REVIEW); PASS = FALSE-PASS (miss)
  titlecase      : header title-case (CAPS issue, NOT bold) -> observational for a bold gate
                   (caps is caught by verification._check_warning's deterministic caps test, not here)

FAIL is INERT in this gate (no independent non-S/crop witness), so a violation can only be caught by
routing to REVIEW -- via the S body-bold veto, main's own body-bold read, or a header disagreement.
The crux: does the S veto catch the boldbody violations the MAIN read rubber-stamps as PASS?
(prod reference: gpt-5.4-mini main alone false-passes ~7/9 boldbody as body_bold=False/high.)

Usage: python scripts/benchmarks/triple_gate_boldsafety.py
Writes output/triple_gate_boldsafety_<ts>.{txt,json}.
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
import triple_gate_compliant as TG   # identical gate: decide / _run_reads / _solo_gate / READS

REPS = 3
VIOLATIONS = ("boldbody", "notbold")


def _score(variant, verdict):
    if variant == "bold_compliant":
        return "correct-PASS" if verdict == "PASS" else ("over-review" if verdict == "REVIEW" else "FALSE-FAIL")
    if variant in VIOLATIONS:
        return "FALSE-PASS" if verdict == "PASS" else ("caught" if verdict in ("REVIEW", "FAIL") else "?")
    return "obs-PASS" if verdict == "PASS" else ("obs-FAIL" if verdict == "FAIL" else "obs-REVIEW")


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def main():
    images = B._bs_images()
    print(f"main={TG.READS[0][1]}:A  specialist_1/2={TG.READS[1][1]}+S (repeat samples)  "
          f"images={len(images)}  reps={REPS}  (bold_safety violations + ground truth)\n")

    # per-variant tallies over all reps; FAIL inert so triple can't FAIL
    tally = defaultdict(lambda: defaultdict(int))          # variant -> outcome -> n   (triple gate)
    main_tally = defaultdict(lambda: defaultdict(int))     # variant -> outcome -> n   (main-alone)
    s_rescue = 0           # violation rep where main-alone=PASS (false-pass) but triple=REVIEW (S veto saved it)
    s_flagged_bb = 0       # boldbody rep where at least one S read body_bold=True (the catch mechanism)
    n_boldbody = 0
    walls, per_image_verdicts, records = [], defaultdict(list), []

    for im in images:
        var = im["variant"]
        for rep in range(1, REPS + 1):
            reads = TG._run_reads(im["path"])
            m_f, m_dt, m_e = reads["main"]
            s1_f, s1_dt, s1_e = reads["specialist_1"]
            s2_f, s2_dt, s2_e = reads["specialist_2"]
            dts = [d for d in (m_dt, s1_dt, s2_dt) if d is not None]
            wall = max(dts) if dts else None
            if wall is not None:
                walls.append(wall)
            verdict, reasons = TG.decide(m_f, s1_f, s2_f)
            main_solo = TG._solo_gate(m_f)
            out = _score(var, verdict)
            main_out = _score(var, main_solo)
            tally[var][out] += 1
            main_tally[var][main_out] += 1
            per_image_verdicts[im["name"]].append(verdict)

            bb1 = None if not s1_f else B._eff_body_bold(s1_f)
            bb2 = None if not s2_f else B._eff_body_bold(s2_f)
            mbb = None if not m_f else B._eff_body_bold(m_f)
            mhb = None if not m_f else B._eff_header_bold(m_f)
            if var == "boldbody":
                n_boldbody += 1
                if bb1 is True or bb2 is True:
                    s_flagged_bb += 1
            if var in VIOLATIONS and main_solo == "PASS" and verdict == "REVIEW":
                s_rescue += 1

            wstr = f"{wall:.2f}" if wall is not None else "ERR"
            records.append({
                "image": im["name"], "variant": var, "rep": rep, "wall": wall,
                "triple_verdict": verdict, "triple_outcome": out, "reasons": reasons,
                "main_alone": main_solo, "main_outcome": main_out,
                "main_hb": mhb, "main_bb": mbb,
                "main_bbc": (m_f or {}).get("body_bold_confidence"),
                "main_hbc": (m_f or {}).get("header_bold_confidence"),
                "s1_bb": bb1, "s1_bbc": (s1_f or {}).get("body_bold_confidence"),
                "s2_bb": bb2, "s2_bbc": (s2_f or {}).get("body_bold_confidence"),
            })
            print(f"  {im['name']:26s} r{rep} wall={wstr}s  main-alone[{main_solo:9s}] "
                  f"main(hb={mhb} bb={mbb}) S1.bb={bb1} S2.bb={bb2} -> {verdict:7s} [{out}]")

    stable = {n: (len(set(v)) == 1) for n, v in per_image_verdicts.items()}
    n_stable = sum(1 for s in stable.values() if s)
    report = {
        "main": f"{TG.READS[0][1]}:A", "specialists": f"{TG.READS[1][1]}+S x2 (repeat samples)",
        "reps": REPS, "fail_branch": "INERT (no independent non-S/crop witness) -> catch == REVIEW",
        "triple_tally": {k: dict(v) for k, v in tally.items()},
        "main_alone_tally": {k: dict(v) for k, v in main_tally.items()},
        "s_rescue_count": s_rescue,
        "s_flagged_boldbody": f"{s_flagged_bb}/{n_boldbody}",
        "stable_images": f"{n_stable}/{len(stable)}",
        "unstable_images": [n for n, s in stable.items() if not s],
        "wall_avg": round(sum(walls) / len(walls), 2) if walls else None,
        "wall_p50": _pct(walls, 50), "wall_max": max(walls) if walls else None,
        "over_5s": sum(1 for x in walls if x > 5), "records": records,
    }
    _summary_print(report)
    _write(report)


def _viol_caught(t):
    caught = t.get("caught", 0)
    fp = t.get("FALSE-PASS", 0)
    return caught, fp, caught + fp


def _summary_print(report):
    tt, mt = report["triple_tally"], report["main_alone_tally"]
    print("\n  === TRIPLE GATE (FAIL inert -> 'caught' = routed to REVIEW) ===")
    bc = tt.get("bold_compliant", {})
    print(f"  bold_compliant: correct-PASS {bc.get('correct-PASS',0)}, over-review {bc.get('over-review',0)}, "
          f"FALSE-FAIL {bc.get('FALSE-FAIL',0)}")
    for v in VIOLATIONS:
        c, fp, n = _viol_caught(tt.get(v, {}))
        print(f"  {v:14s}: CAUGHT {c}/{n}, FALSE-PASS {fp}/{n}")
    print(f"  S flagged body-bold on boldbody: {report['s_flagged_boldbody']}   "
          f"S rescued (main-alone PASS -> triple REVIEW): {report['s_rescue_count']}")
    print("\n  === MAIN-ALONE baseline (can FAIL) ===")
    bcm = mt.get("bold_compliant", {})
    print(f"  bold_compliant: correct-PASS {bcm.get('correct-PASS',0)}, over-review {bcm.get('over-review',0)}, "
          f"FALSE-FAIL {bcm.get('FALSE-FAIL',0)}")
    for v in VIOLATIONS:
        c, fp, n = _viol_caught(mt.get(v, {}))
        print(f"  {v:14s}: CAUGHT {c}/{n}, FALSE-PASS {fp}/{n}")
    print(f"\n  stability: {report['stable_images']} images identical across {report['reps']} reps  |  "
          f"wall avg {report['wall_avg']}s p50 {report['wall_p50']}s max {report['wall_max']}s "
          f"(>5s {report['over_5s']})\n")


def _write(report):
    tt, mt = report["triple_tally"], report["main_alone_tally"]
    L = ["", "=" * 104,
         "TRIPLE-READ gate (main + 2x S repeat samples) on bold_safety VIOLATIONS, 3x -- RECALL",
         "=" * 104,
         f"main = {report['main']}   specialists = {report['specialists']}   reps = {report['reps']}",
         f"FAIL branch: {report['fail_branch']}",
         "boldbody/notbold are VIOLATIONS -> CAUGHT = routed to REVIEW (good); FALSE-PASS = gate said "
         "PASS (a dangerous miss). titlecase = caps issue, OUT OF SCOPE for a bold gate (observational; "
         "caps is caught by verification._check_warning, not here). bold_compliant -> PASS is correct.", ""]
    L.append(f"{'variant':16s} {'TRIPLE (caught/false-pass or pass)':36s} {'MAIN-ALONE':30s}")
    L.append("-" * 96)
    bc, bcm = tt.get("bold_compliant", {}), mt.get("bold_compliant", {})
    L.append(f"{'bold_compliant':16s} "
             f"{'PASS '+str(bc.get('correct-PASS',0))+'  review '+str(bc.get('over-review',0))+'  fail '+str(bc.get('FALSE-FAIL',0)):36s} "
             f"{'PASS '+str(bcm.get('correct-PASS',0))+'  review '+str(bcm.get('over-review',0))+'  FAIL '+str(bcm.get('FALSE-FAIL',0)):30s}")
    for v in VIOLATIONS:
        c, fp, n = _viol_caught(tt.get(v, {}))
        cm, fpm, nm = _viol_caught(mt.get(v, {}))
        L.append(f"{v:16s} {'CAUGHT '+str(c)+'/'+str(n)+'   FALSE-PASS '+str(fp):36s} "
                 f"{'CAUGHT '+str(cm)+'/'+str(nm)+'   FALSE-PASS '+str(fpm):30s}")
    tc, mtc = tt.get("titlecase", {}), mt.get("titlecase", {})
    L.append(f"{'titlecase(obs)':16s} {str(dict(tc)):36s} {str(dict(mtc)):30s}")
    L.append("")
    L.append(f"S flagged body-bold on boldbody reps: {report['s_flagged_boldbody']}   "
             f"S RESCUE (main-alone PASS -> triple REVIEW): {report['s_rescue_count']}")
    L.append(f"stability: {report['stable_images']} images gave an identical verdict across all "
             f"{report['reps']} reps")
    if report["unstable_images"]:
        L.append(f"  unstable images: {report['unstable_images']}")
    L.append(f"latency (parallel wall = slowest of 3 reads): avg {report['wall_avg']}s  "
             f"p50 {report['wall_p50']}s  max {report['wall_max']}s  >5s {report['over_5s']}")
    L.append("")
    L.append("per-image-rep (main-alone = production gate on the gpt-5.4-mini:A read; can FAIL):")
    for r in report["records"]:
        wstr = f"{r['wall']:.2f}s" if r["wall"] is not None else "ERR"
        mbc = (r["main_bbc"] or "-")[:1]
        s1c = (r["s1_bbc"] or "-")[:1]
        s2c = (r["s2_bbc"] or "-")[:1]
        L.append(f"   {r['image']:26s} r{r['rep']} {wstr:7s} main-alone[{r['main_alone']:9s}] "
                 f"main(hb={r['main_hb']}/{(r['main_hbc'] or '-')[:1]} bb={r['main_bb']}/{mbc}) "
                 f"S1.bb={r['s1_bb']}/{s1c} S2.bb={r['s2_bb']}/{s2c} -> {r['triple_verdict']:7s} "
                 f"[{r['triple_outcome']}]")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"triple_gate_boldsafety_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written {os.path.relpath(base, ROOT)}.txt / .json")


if __name__ == "__main__":
    main()
