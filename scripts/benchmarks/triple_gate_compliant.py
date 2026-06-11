"""Three-parallel-read BOLD gate on COMPLIANT labels (baseline + clearer_baseline), 1x,
with TIME + ACCURACY. No production code touched; reuses bold_prompt_safety prompt A + _call and
prompt_S_test's PROMPT_S verbatim.

Three reads per image, run CONCURRENTLY (wall = max of the three call latencies = assumed parallel):
  main          : gpt-5.4-mini + main prompt A        (the only non-S / neutral witness)
  specialist_1  : gpt-4.1      + PROMPT_S ("traps")    } two REPEAT SAMPLES of the SAME S witness --
  specialist_2  : gpt-4.1      + PROMPT_S ("traps")    } NOT independent witnesses. May veto, never FAIL.

Arbitration under test (the user's, refined):
  PASS  -> main header_bold=True/high AND main body_bold=False/high AND main legibility good
           AND neither S specialist returns body_bold=True.
  FAIL  -> only with TWO independent non-S (or crop-quality) witnesses body_bold=True/high, OR one
           high-res crop body_bold=True/high + good legibility. No crop/independent-non-S witness is
           implemented here, so this branch CANNOT fire for body-bold (FAIL stays 0). A lone main
           body-bold is NOT a fail -> it routes to REVIEW.
  REVIEW-> any S specialist body_bold=True; OR main body_bold=True/high; OR main not a clean
           high-confidence pass; OR a specialist disagrees with main on header boldness; OR
           poor/limited legibility; OR timeout/error.

Labels are COMPLIANT, so on the warning-bearing backs: PASS = correct, FAIL = FALSE-fail,
REVIEW = over-caution. Fronts: correct = no warning. Also reports the MAIN-ALONE baseline
(production header_body_gate on the gpt-5.4-mini:A read, which CAN fail) from the same reads.

Usage: python scripts/benchmarks/triple_gate_compliant.py
Writes output/triple_gate_compliant_<ts>.{txt,json}.
"""
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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

PROMPT_S = PS.PROMPT_S
FOLDERS = ["baseline_labels", "clearer_baseline_labels"]
# (role, model, prompt)
READS = [
    ("main", "gpt-5.4-mini", B._prompt("A")),
    ("specialist_1", "gpt-4.1", PROMPT_S),
    ("specialist_2", "gpt-4.1", PROMPT_S),
]


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
    """MAIN-ALONE baseline: production header_body_gate on a single read (this one CAN fail)."""
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


def decide(main, s1, s2):
    """Three-read arbitration. Returns (verdict, reasons:list). FAIL is inert (no independent
    non-S / crop witness), so body-bold can never FAIL here."""
    if not main or not s1 or not s2:
        return "REVIEW", ["timeout/error"]
    hb_m, bb_m, hbc_m, bbc_m, leg_m = _vals(main)
    bb_s1 = B._eff_body_bold(s1)
    bb_s2 = B._eff_body_bold(s2)
    hb_s1 = B._eff_header_bold(s1)
    hb_s2 = B._eff_header_bold(s2)

    # FAIL branch -- requires >=2 independent NON-S witnesses (or a crop read) with body_bold
    # True/high. The only non-S witness here is `main` (one), and no crop read exists, so this
    # branch is inert by construction. Implemented explicitly for fidelity.
    independent_nonS_bodybold_high = sum(
        1 for (bb, bbc) in [(bb_m, bbc_m)] if bb is True and bbc == "high")
    crop_bodybold_high = False  # no crop read implemented
    if independent_nonS_bodybold_high >= 2 or crop_bodybold_high:
        return "FAIL", ["two-independent-nonS-body-bold"]

    reasons = []
    # REVIEW triggers (collect all that apply; verdict is REVIEW if any)
    if bb_s1 is True or bb_s2 is True:
        reasons.append("S body-bold veto")
    if bb_m is True and bbc_m == "high":
        reasons.append("main body-bold/high (->review, no 2nd non-S witness)")
    # header disagreement: an S confidently asserts the opposite header weight from main
    if hb_m is True and (hb_s1 is False or hb_s2 is False):
        reasons.append("header disagreement (S says header not bold)")
    if leg_m in ("poor", "limited"):
        reasons.append("limited/poor legibility (main)")
    main_clean = (hb_m is True and hbc_m == "high" and bb_m is False and bbc_m == "high"
                  and leg_m == "good")
    if not main_clean and not reasons:
        reasons.append("main not clean high-conf pass")

    if reasons:
        return "REVIEW", reasons
    # PASS: main clean high-conf AND neither S flags body-bold (already guaranteed by no reasons)
    return "PASS", ["main clean + both S no body-bold"]


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def _run_reads(path):
    """Fire the three reads concurrently; return role -> (fields, dt, err)."""
    out = {}
    with ThreadPoolExecutor(max_workers=len(READS)) as pool:
        futs = {role: pool.submit(B._call, model, prompt, [path]) for (role, model, prompt) in READS}
        for role, fut in futs.items():
            fields, dt, _retries, err = fut.result()
            out[role] = (fields, dt, err)
    return out


def main():
    images = _gather()
    n_back = sum(1 for *_, b in images if b)
    print(f"main=gpt-5.4-mini:A  specialist_1/2=gpt-4.1+S (repeat samples)  "
          f"images={len(images)} ({n_back} backs) over {FOLDERS}\n")

    backs = {"n": 0, "PASS": 0, "FAIL": 0, "REVIEW": 0}
    base_backs = {"n": 0, "PASS": 0, "FAIL": 0, "REVIEW": 0}     # main-alone baseline
    fronts = {"n": 0, "no_warning_ok": 0, "hallucinated": 0}
    by_folder = defaultdict(lambda: {"n": 0, "PASS": 0, "FAIL": 0, "REVIEW": 0})
    walls, per_image, reason_counts = [], [], defaultdict(int)

    for folder, fname, path, is_back in images:
        reads = _run_reads(path)
        m_f, m_dt, m_e = reads["main"]
        s1_f, s1_dt, s1_e = reads["specialist_1"]
        s2_f, s2_dt, s2_e = reads["specialist_2"]
        dts = [d for d in (m_dt, s1_dt, s2_dt) if d is not None]
        wall = max(dts) if dts else None       # assumed-parallel wall = slowest read
        if wall is not None:
            walls.append(wall)
        main_solo = _solo_gate(m_f)

        if is_back:
            verdict, reasons = decide(m_f, s1_f, s2_f)
            backs["n"] += 1
            backs[verdict] += 1
            by_folder[folder]["n"] += 1
            by_folder[folder][verdict] += 1
            reason_counts[reasons[0]] += 1
            base_backs["n"] += 1
            base_backs[main_solo if main_solo in base_backs else "REVIEW"] += 1
            tag, reason = verdict, "; ".join(reasons)
        else:
            wp_any = any((r[0] or {}).get("warning_present") is True for r in
                         (reads["main"], reads["specialist_1"], reads["specialist_2"]))
            fronts["n"] += 1
            if wp_any:
                fronts["hallucinated"] += 1; tag = "HALLUCINATED"; reason = "front warning seen"
            else:
                fronts["no_warning_ok"] += 1; tag = "no-warning-OK"; reason = "-"
            verdict = tag

        wstr = f"{wall:.2f}" if wall is not None else "ERR"
        per_image.append({
            "img": f"{folder}/{fname}", "is_back": is_back, "verdict": verdict, "reason": reason,
            "main_alone": main_solo, "wall": wall,
            "reads": {
                "main": {"dt": m_dt, "err": m_e, "vals": (None if not m_f else list(_vals(m_f))),
                         "warning_present": (m_f or {}).get("warning_present")},
                "specialist_1": {"dt": s1_dt, "err": s1_e, "body_bold": (None if not s1_f else B._eff_body_bold(s1_f)),
                                 "header_bold": (None if not s1_f else B._eff_header_bold(s1_f)),
                                 "bbc": (s1_f or {}).get("body_bold_confidence")},
                "specialist_2": {"dt": s2_dt, "err": s2_e, "body_bold": (None if not s2_f else B._eff_body_bold(s2_f)),
                                 "header_bold": (None if not s2_f else B._eff_header_bold(s2_f)),
                                 "bbc": (s2_f or {}).get("body_bold_confidence")},
            },
        })
        bb1 = None if not s1_f else B._eff_body_bold(s1_f)
        bb2 = None if not s2_f else B._eff_body_bold(s2_f)
        print(f"  {folder}/{fname:24s} wall={wstr}s  main-alone[{main_solo}]  "
              f"S1.bb={bb1} S2.bb={bb2} -> {tag}" + (f"  ({reason})" if is_back else ""))

    report = {
        "main": "gpt-5.4-mini:A", "specialists": "gpt-4.1+S x2 (repeat samples)",
        "backs": backs, "main_alone_backs": base_backs, "fronts": fronts,
        "by_folder": {k: dict(v) for k, v in by_folder.items()},
        "review_reasons": dict(reason_counts),
        "wall_avg": round(sum(walls) / len(walls), 2) if walls else None,
        "wall_p50": _pct(walls, 50), "wall_max": max(walls) if walls else None,
        "over_5s": sum(1 for x in walls if x > 5), "per_image": per_image,
    }
    print(f"\n  -> TRIPLE GATE backs: PASS {backs['PASS']}/{backs['n']}  FALSE-FAIL {backs['FAIL']}  "
          f"review {backs['REVIEW']}")
    print(f"     MAIN-ALONE   backs: PASS {base_backs['PASS']}/{base_backs['n']}  "
          f"FALSE-FAIL {base_backs['FAIL']}  review {base_backs['REVIEW']}")
    print(f"     fronts no-warning {fronts['no_warning_ok']}/{fronts['n']}  |  wall avg "
          f"{report['wall_avg']}s p50 {report['wall_p50']}s max {report['wall_max']}s "
          f"(>5s {report['over_5s']})\n")
    _write(report)


def _write(report):
    b, mb, f = report["backs"], report["main_alone_backs"], report["fronts"]
    L = ["", "=" * 100,
         "TRIPLE-READ gate (main + 2x S repeat samples) on COMPLIANT labels "
         "(baseline+clearer), 1x", "=" * 100,
         f"main = {report['main']}   specialists = {report['specialists']}",
         "rule: PASS = main clean(hdr bold/high + body not-bold/high + leg good) AND neither S "
         "body_bold=True; FAIL = inert (needs 2 independent non-S/crop body-bold; none here -> 0); "
         "REVIEW = any S body-bold, main body-bold/high, header disagreement, poor legibility, or "
         "not-clean main.",
         "labels are COMPLIANT -> PASS = correct, FAIL = FALSE-fail, REVIEW = over-caution.", ""]
    cp = f"{b['PASS']}/{b['n']}"
    mcp = f"{mb['PASS']}/{mb['n']}"
    fo = f"{f['no_warning_ok']}/{f['n']}"
    L.append(f"{'gate':30s} {'PASS':10s} {'FALSE-FAIL':12s} {'review':8s} {'frontOK':9s} "
             f"{'wall avg':9s} {'p50':6s} {'max':6s} {'>5s':4s}")
    L.append("-" * 98)
    L.append(f"{'triple (main + 2xS)':30s} {cp:10s} {str(b['FAIL']):12s} {str(b['REVIEW']):8s} "
             f"{fo:9s} {str(report['wall_avg'])+'s':9s} {str(report['wall_p50']):6s} "
             f"{str(report['wall_max']):6s} {str(report['over_5s']):4s}")
    L.append(f"{'main-alone baseline':30s} {mcp:10s} {str(mb['FAIL']):12s} {str(mb['REVIEW']):8s} "
             f"{'(same fronts)':9s}")
    L.append("")
    L.append(f"per-folder backs (triple gate): {report['by_folder']}")
    L.append(f"review/verdict reasons (backs): {report['review_reasons']}")
    L.append("")
    L.append("per-image (main-alone = production gate on the gpt-5.4-mini:A read; S1.bb/S2.bb = "
             "each S sample's body_bold; wall = parallel max):")
    for pi in report["per_image"]:
        wstr = f"{pi['wall']:.2f}s" if pi["wall"] is not None else "ERR"
        s1 = pi["reads"]["specialist_1"]; s2 = pi["reads"]["specialist_2"]
        mv = pi["reads"]["main"]["vals"]
        mvs = "ERR" if mv is None else f"hb={mv[0]}/{(mv[2] or '-')[:1]} bb={mv[1]}/{(mv[3] or '-')[:1]} leg={mv[4]}"
        L.append(f"   {pi['img']:42s} wall={wstr:7s} main-alone[{pi['main_alone']:9s}] "
                 f"main({mvs})  S1.bb={s1['body_bold']}/{(s1['bbc'] or '-')[:1]} "
                 f"S2.bb={s2['body_bold']}/{(s2['bbc'] or '-')[:1]} -> {pi['verdict']:7s} ({pi['reason']})")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"triple_gate_compliant_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written {os.path.relpath(base, ROOT)}.txt / .json")


if __name__ == "__main__":
    main()
