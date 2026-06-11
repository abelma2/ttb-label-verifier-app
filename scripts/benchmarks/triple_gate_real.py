"""The HIGH-only triple gate (main gpt-5.4-mini:A + 2x gpt-4.1+S; = TG.decide) on REAL photographed
labels, 3x, with TIME + ACCURACY + per-image stability + a main-alone baseline. No production code
touched; reuses triple_gate_compliant.decide/_run_reads/_solo_gate VERBATIM (identical gate logic).

real_labels are real commercial bottles (legally COMPLIANT: bold header, non-bold body on the back),
photographed -> the noisiest input. So on the warning-bearing backs: PASS = correct, FAIL = false-fail
(FAIL is inert in this gate -> expect 0), REVIEW = over-caution (expected to be HIGH on real photos --
that is the safe-but-low-throughput behavior we want to quantify). Fronts: correct = no warning.

3x so we can report how often a back's verdict is identical across reps (the instability that 1x runs
on noisy photos cannot show), and the MAIN-ALONE baseline (the prior-default header_body_gate on the
gpt-5.4-mini:A read, which CAN fail) to see whether the 2 specialists change real-label outcomes.

Usage: python scripts/benchmarks/triple_gate_real.py
Writes output/triple_gate_real_<ts>.{txt,json}.
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
import triple_gate_compliant as TG       # identical gate: _run_reads / decide / _solo_gate / _vals

FOLDER = "real_labels"
REPS = 3


def _gather():
    out = []
    d = os.path.join(ROOT, "test_labels", FOLDER)
    for fname in sorted(os.listdir(d)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            out.append((fname, os.path.join(d, fname), "_other" in fname.lower()))
    return out


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def main():
    images = _gather()
    n_back = sum(1 for *_, b in images if b)
    print(f"HIGH-only triple gate on {FOLDER}  |  {len(images)} images ({n_back} backs) x {REPS} reps\n")

    backs = {"n": 0, "PASS": 0, "FAIL": 0, "REVIEW": 0}
    base_backs = {"n": 0, "PASS": 0, "FAIL": 0, "REVIEW": 0}
    fronts = {"n": 0, "no_warning_ok": 0, "hallucinated": 0}
    walls, reason_counts, records = [], defaultdict(int), []
    back_verdicts = defaultdict(list)     # fname -> [verdict per rep]
    s_rescue = 0                          # back rep where main-alone PASS but triple REVIEW (S over-flag veto)

    for fname, path, is_back in images:
        for rep in range(1, REPS + 1):
            reads = TG._run_reads(path)
            m_f, m_dt, _ = reads["main"]
            s1_f, s1_dt, _ = reads["specialist_1"]
            s2_f, s2_dt, _ = reads["specialist_2"]
            dts = [d for d in (m_dt, s1_dt, s2_dt) if d is not None]
            wall = max(dts) if dts else None
            if wall is not None:
                walls.append(wall)
            main_solo = TG._solo_gate(m_f)

            if is_back:
                verdict, reasons = TG.decide(m_f, s1_f, s2_f)
                backs["n"] += 1
                backs[verdict] += 1
                reason_counts[reasons[0]] += 1
                back_verdicts[fname].append(verdict)
                base_backs["n"] += 1
                base_backs[main_solo if main_solo in base_backs else "REVIEW"] += 1
                if main_solo == "PASS" and verdict == "REVIEW":
                    s_rescue += 1
                tag, reason = verdict, "; ".join(reasons)
            else:
                wp = any((r[0] or {}).get("warning_present") is True for r in
                         (reads["main"], reads["specialist_1"], reads["specialist_2"]))
                fronts["n"] += 1
                fronts["hallucinated" if wp else "no_warning_ok"] += 1
                tag, reason, verdict = ("HALLUCINATED" if wp else "no-warning-OK"), "-", None

            bb1 = None if not s1_f else B._eff_body_bold(s1_f)
            bb2 = None if not s2_f else B._eff_body_bold(s2_f)
            mv = None if not m_f else list(TG._vals(m_f))
            wstr = f"{wall:.2f}" if wall is not None else "ERR"
            records.append({"img": fname, "rep": rep, "is_back": is_back, "verdict": tag,
                            "reason": reason, "main_alone": main_solo, "wall": wall,
                            "main": mv, "s1_bb": bb1, "s2_bb": bb2})
            mvs = "ERR" if mv is None else f"hb={mv[0]}/{(mv[2] or '-')[:1]} bb={mv[1]}/{(mv[3] or '-')[:1]}"
            print(f"  {fname:22s} r{rep} {wstr:6s}s main-alone[{main_solo:9s}] main({mvs}) "
                  f"S1.bb={bb1} S2.bb={bb2} -> {tag}" + (f"  ({reason})" if is_back else ""))

    stable = {fn: (len(set(v)) == 1) for fn, v in back_verdicts.items()}
    n_stable = sum(1 for s in stable.values() if s)
    report = {
        "folder": FOLDER, "reps": REPS, "backs": backs, "main_alone_backs": base_backs,
        "fronts": fronts, "review_reasons": dict(reason_counts), "s_rescue_count": s_rescue,
        "back_verdicts_per_image": {k: v for k, v in back_verdicts.items()},
        "stable_backs": f"{n_stable}/{len(stable)}",
        "unstable_backs": {fn: back_verdicts[fn] for fn, s in stable.items() if not s},
        "wall_avg": round(sum(walls) / len(walls), 2) if walls else None,
        "wall_p50": _pct(walls, 50), "wall_max": max(walls) if walls else None,
        "over_5s": sum(1 for x in walls if x > 5), "records": records,
    }
    _summary(report)
    _write(report)


def _summary(r):
    b, mb, f = r["backs"], r["main_alone_backs"], r["fronts"]
    print("\n  ===== SUMMARY (real_labels, COMPLIANT -> PASS correct, FAIL false-fail, REVIEW over-caution) =====")
    print(f"  TRIPLE GATE backs: PASS {b['PASS']}/{b['n']}  FALSE-FAIL {b['FAIL']}  REVIEW {b['REVIEW']}")
    print(f"  MAIN-ALONE  backs: PASS {mb['PASS']}/{mb['n']}  FALSE-FAIL {mb['FAIL']}  REVIEW {mb['REVIEW']}")
    print(f"  S rescued (main-alone PASS -> triple REVIEW): {r['s_rescue_count']}")
    print(f"  fronts no-warning OK: {f['no_warning_ok']}/{f['n']}  (hallucinated {f['hallucinated']})")
    print(f"  per-image back-verdict stability across {r['reps']} reps: {r['stable_backs']}  "
          f"(unstable: {list(r['unstable_backs'].keys()) or 'none'})")
    print(f"  review reasons: {r['review_reasons']}")
    print(f"  latency: avg {r['wall_avg']}s p50 {r['wall_p50']}s max {r['wall_max']}s (>5s {r['over_5s']})\n")


def _write(r):
    b, mb, f = r["backs"], r["main_alone_backs"], r["fronts"]
    L = ["", "=" * 104, f"HIGH-only TRIPLE GATE on {r['folder']} (real photographed bottles), {r['reps']}x",
         "=" * 104,
         "real commercial labels = COMPLIANT (bold header, non-bold body); photographed = noisiest input. "
         "PASS = correct, FAIL = false-fail (FAIL inert -> 0), REVIEW = over-caution (expected high on photos).", ""]
    L.append(f"{'gate':22s} {'PASS':10s} {'FALSE-FAIL':12s} {'REVIEW':8s} {'frontOK':9s} "
             f"{'wall avg':9s} {'p50':6s} {'max':6s} {'>5s':4s}")
    L.append("-" * 96)
    cp, mcp, fo = f"{b['PASS']}/{b['n']}", f"{mb['PASS']}/{mb['n']}", f"{f['no_warning_ok']}/{f['n']}"
    L.append(f"{'triple (main + 2xS)':22s} {cp:10s} {str(b['FAIL']):12s} {str(b['REVIEW']):8s} {fo:9s} "
             f"{str(r['wall_avg'])+'s':9s} {str(r['wall_p50']):6s} {str(r['wall_max']):6s} {str(r['over_5s']):4s}")
    L.append(f"{'main-alone baseline':22s} {mcp:10s} {str(mb['FAIL']):12s} {str(mb['REVIEW']):8s} "
             f"{'(same fronts)':9s}")
    L.append("")
    L.append(f"S rescued (main-alone PASS -> triple REVIEW, i.e. an S over-flag vetoed a main pass): {r['s_rescue_count']}")
    L.append(f"per-image back-verdict stability across {r['reps']} reps: {r['stable_backs']}")
    if r["unstable_backs"]:
        L.append("  UNSTABLE backs (verdict flips across reps):")
        for fn, v in r["unstable_backs"].items():
            L.append(f"     {fn:24s} {v}")
    L.append(f"review reasons (backs): {r['review_reasons']}")
    L.append(f"latency: avg {r['wall_avg']}s  p50 {r['wall_p50']}s  max {r['wall_max']}s  >5s {r['over_5s']}")
    L.append("")
    L.append("per-image-rep (main-alone = prior-default header_body_gate on the gpt-5.4-mini:A read; "
             "can FAIL):")
    for rec in r["records"]:
        mv = rec["main"]
        mvs = "ERR" if mv is None else f"hb={mv[0]}/{(mv[2] or '-')[:1]} bb={mv[1]}/{(mv[3] or '-')[:1]} leg={mv[4]}"
        wstr = f"{rec['wall']:.2f}s" if rec["wall"] is not None else "ERR"
        L.append(f"   {rec['img']:22s} r{rec['rep']} {wstr:7s} main-alone[{rec['main_alone']:9s}] "
                 f"main({mvs}) S1.bb={rec['s1_bb']} S2.bb={rec['s2_bb']} -> {rec['verdict']:13s}"
                 + (f" ({rec['reason']})" if rec["is_back"] else ""))
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"triple_gate_real_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written {os.path.relpath(base, ROOT)}.txt / .json")


if __name__ == "__main__":
    main()
