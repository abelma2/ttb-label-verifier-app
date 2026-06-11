"""Configs A vs B (helper-model) on REAL photographed labels, 3x. Same high-only gate as the
helper-model bold_safety run (HM.decide, with the high-conf header contradiction). No production
code touched.

  main = gpt-5.4-mini + prompt A  (shared across A and B)
  A: helper_1 = gpt-4.1+S      helper_2 = gpt-4.1+S
  B: helper_1 = gpt-4.1+S      helper_2 = gpt-5.4-mini+S

CONTROLLED: per image-rep, read main + g41_a + g41_b + g54_a (4 distinct reads, concurrent, bounded
to one image at a time). A = main+g41_a+g41_b; B = main+g41_a+g54_a (B reuses A's first gpt-4.1
helper). So the ONLY difference between A and B is helper_2: a 2nd gpt-4.1+S (A) vs a gpt-5.4-mini+S
(B). wall(config) = max of that config's 3 read latencies.

real_labels are COMPLIANT commercial bottles -> backs: PASS = correct, REVIEW = over-caution,
FAIL = false-fail (inert -> 0). Hypothesis: on compliant bodies gpt-5.4-mini+S OVER-flags body-bold
LESS than gpt-4.1+S, so B's 2nd helper should false-veto less -> B yields MORE compliant passes than
A (the throughput mirror of A's violation-safety edge). Reports per-config PASS/REVIEW, the A-vs-B
delta, the 2nd-helper over-flag rates, stability across reps, latency. Writes TXT + JSON.

Usage: python scripts/benchmarks/helper_AB_real.py
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
import helper_models_boldsafety as HM     # reuse decide() (exact gate) + _vals

PROMPT_A = B._prompt("A")
PROMPT_S = PS.PROMPT_S
FOLDER = "real_labels"
REPS = 3

READS = {
    "main":  ("gpt-5.4-mini", PROMPT_A),
    "g41_a": ("gpt-4.1", PROMPT_S),
    "g41_b": ("gpt-4.1", PROMPT_S),
    "g54_a": ("gpt-5.4-mini", PROMPT_S),
}
CONFIGS = {
    "A: 2x gpt-4.1+S":           ("main", "g41_a", "g41_b"),
    "B: gpt-4.1+S & 5.4-mini+S": ("main", "g41_a", "g54_a"),
}


def _gather():
    out = []
    d = os.path.join(ROOT, "test_labels", FOLDER)
    for fname in sorted(os.listdir(d)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            out.append((fname, os.path.join(d, fname), "_other" in fname.lower()))
    return out


def _read_image(path):
    out = {}
    with ThreadPoolExecutor(max_workers=len(READS)) as pool:
        futs = {role: pool.submit(B._call, model, prompt, [path]) for role, (model, prompt) in READS.items()}
        for role, fut in futs.items():
            fields, dt, _r, err = fut.result()
            out[role] = (fields, dt, err)
    return out


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def main():
    images = _gather()
    n_back = sum(1 for *_, b in images if b)
    print(f"configs A vs B on {FOLDER}  |  {len(images)} images ({n_back} backs) x {REPS} reps  "
          f"(4 reads/image, bounded)\n")

    backs = {c: defaultdict(int) for c in CONFIGS}          # config -> verdict -> n
    fronts = {"n": 0, "ok": 0, "hallucinated": 0}
    walls = {c: [] for c in CONFIGS}
    back_verdicts = {c: defaultdict(list) for c in CONFIGS}  # config -> fname -> [verdict per rep]
    # 2nd-helper over-flag on compliant backs (body_bold=True is a FALSE flag here)
    h2_overflag = {"A_g41_b": 0, "B_g54_a": 0, "n_back_reps": 0}
    diff = {"B_pass_A_review": 0, "A_pass_B_review": 0, "agree": 0}
    records = []

    for fname, path, is_back in images:
        for rep in range(1, REPS + 1):
            reads = _read_image(path)
            vresult = {}
            for cname, (mr, h1r, h2r) in CONFIGS.items():
                m_f, m_dt, _ = reads[mr]
                h1_f, h1_dt, _ = reads[h1r]
                h2_f, h2_dt, _ = reads[h2r]
                dts = [d for d in (m_dt, h1_dt, h2_dt) if d is not None]
                wall = max(dts) if dts else None
                if wall is not None:
                    walls[cname].append(wall)
                verdict, reasons = HM.decide(m_f, h1_f, h2_f)
                vresult[cname] = {"verdict": verdict, "reasons": reasons, "wall": wall,
                                  "h2_bb": (None if not h2_f else B._eff_body_bold(h2_f))}
            if is_back:
                for cname in CONFIGS:
                    v = vresult[cname]["verdict"]
                    backs[cname][v] += 1
                    back_verdicts[cname][fname].append(v)
                # 2nd-helper over-flag
                h2_overflag["n_back_reps"] += 1
                if reads["g41_b"][0] and B._eff_body_bold(reads["g41_b"][0]) is True:
                    h2_overflag["A_g41_b"] += 1
                if reads["g54_a"][0] and B._eff_body_bold(reads["g54_a"][0]) is True:
                    h2_overflag["B_g54_a"] += 1
                va, vb = vresult["A: 2x gpt-4.1+S"]["verdict"], vresult["B: gpt-4.1+S & 5.4-mini+S"]["verdict"]
                if va == vb:
                    diff["agree"] += 1
                elif vb == "PASS" and va == "REVIEW":
                    diff["B_pass_A_review"] += 1
                elif va == "PASS" and vb == "REVIEW":
                    diff["A_pass_B_review"] += 1
            else:
                wp = any((reads[r][0] or {}).get("warning_present") is True for r in ("main", "g41_a", "g41_b", "g54_a"))
                fronts["n"] += 1
                fronts["hallucinated" if wp else "ok"] += 1

            m_f = reads["main"][0]
            mv = None if not m_f else list(HM._vals(m_f))
            records.append({"img": fname, "rep": rep, "is_back": is_back, "main": mv,
                            "A": vresult["A: 2x gpt-4.1+S"], "B": vresult["B: gpt-4.1+S & 5.4-mini+S"],
                            "g41_b_bb": (None if not reads["g41_b"][0] else B._eff_body_bold(reads["g41_b"][0])),
                            "g54_a_bb": (None if not reads["g54_a"][0] else B._eff_body_bold(reads["g54_a"][0]))})
            if is_back:
                mvs = "ERR" if mv is None else f"hb={mv[0]}/{(mv[2] or '-')[:1]} bb={mv[1]}/{(mv[3] or '-')[:1]} leg={mv[4]}"
                a, b = vresult["A: 2x gpt-4.1+S"], vresult["B: gpt-4.1+S & 5.4-mini+S"]
                print(f"  {fname:22s} r{rep} main({mvs})  A={a['verdict']:6s}(g41_b.bb={records[-1]['g41_b_bb']})  "
                      f"B={b['verdict']:6s}(g54_a.bb={records[-1]['g54_a_bb']})")

    stable = {}
    for c in CONFIGS:
        st = {fn: (len(set(v)) == 1) for fn, v in back_verdicts[c].items()}
        stable[c] = f"{sum(1 for s in st.values() if s)}/{len(st)}"

    report = {
        "folder": FOLDER, "reps": REPS,
        "backs": {c: dict(v) for c, v in backs.items()}, "fronts": fronts,
        "A_vs_B_on_backs": diff, "second_helper_overflag": h2_overflag,
        "stable_backs": stable,
        "latency": {c: {"avg": round(sum(w) / len(w), 2) if w else None, "p50": _pct(w, 50),
                        "max": max(w) if w else None, "over_5s": sum(1 for x in w if x > 5)}
                    for c, w in walls.items()},
        "records": records,
    }
    _summary(report)
    _write(report)


def _summary(r):
    print("\n  ===== SUMMARY (real_labels backs, COMPLIANT -> PASS correct, REVIEW over-caution, FAIL false-fail) =====")
    for c in CONFIGS:
        b = r["backs"][c]; n = sum(b.values()); lat = r["latency"][c]
        print(f"  {c:28s} PASS {b.get('PASS',0)}/{n}  REVIEW {b.get('REVIEW',0)}  FAIL {b.get('FAIL',0)}  "
              f"|  wall p50 {lat['p50']}s max {lat['max']}s >5s {lat['over_5s']}  stable {r['stable_backs'][c]}")
    d = r["A_vs_B_on_backs"]
    print(f"  A vs B on backs: agree {d['agree']}, B-PASS/A-REVIEW {d['B_pass_A_review']}, "
          f"A-PASS/B-REVIEW {d['A_pass_B_review']}")
    o = r["second_helper_overflag"]
    print(f"  2nd-helper FALSE body-bold flags on {o['n_back_reps']} compliant back-reps: "
          f"A's g41_b (gpt-4.1) {o['A_g41_b']}  vs  B's g54_a (gpt-5.4-mini) {o['B_g54_a']}")
    print(f"  fronts no-warning OK {r['fronts']['ok']}/{r['fronts']['n']} (hallucinated {r['fronts']['hallucinated']})\n")


def _write(r):
    L = ["", "=" * 104, f"Configs A vs B (helper-model) on {r['folder']} (real photos), {r['reps']}x", "=" * 104,
         "main = gpt-5.4-mini:A (shared). A helper_2 = gpt-4.1+S; B helper_2 = gpt-5.4-mini+S. "
         "COMPLIANT -> PASS correct, REVIEW over-caution, FAIL false-fail (inert -> 0).", ""]
    L.append(f"{'config':28s} {'PASS':10s} {'REVIEW':8s} {'FAIL':6s} {'wall p50/max':14s} {'>5s':5s} {'stable':8s}")
    L.append("-" * 92)
    for c in CONFIGS:
        b = r["backs"][c]; n = sum(b.values()); lat = r["latency"][c]
        cp = f"{b.get('PASS',0)}/{n}"
        lats = f"{lat['p50']}/{lat['max']}s"
        L.append(f"{c:28s} {cp:10s} {str(b.get('REVIEW',0)):8s} {str(b.get('FAIL',0)):6s} {lats:14s} "
                 f"{str(lat['over_5s']):5s} {r['stable_backs'][c]:8s}")
    L.append("")
    d = r["A_vs_B_on_backs"]
    L.append(f"A vs B on backs: agree {d['agree']}  |  B-PASS where A-REVIEW {d['B_pass_A_review']}  |  "
             f"A-PASS where B-REVIEW {d['A_pass_B_review']}")
    o = r["second_helper_overflag"]
    L.append(f"2nd-helper FALSE body-bold flags on {o['n_back_reps']} compliant back-reps: "
             f"A's 2nd helper g41_b (gpt-4.1+S) = {o['A_g41_b']}   B's 2nd helper g54_a (gpt-5.4-mini+S) = {o['B_g54_a']}")
    L.append(f"fronts no-warning OK: {r['fronts']['ok']}/{r['fronts']['n']} (hallucinated {r['fronts']['hallucinated']})")
    L.append("")
    L.append("per back-rep (g41_b = A's 2nd helper; g54_a = B's 2nd helper; bb=True is a FALSE flag here):")
    for rec in r["records"]:
        if not rec["is_back"]:
            continue
        mv = rec["main"]
        mvs = "ERR" if mv is None else f"hb={mv[0]}/{(mv[2] or '-')[:1]} bb={mv[1]}/{(mv[3] or '-')[:1]} leg={mv[4]}"
        a, b = rec["A"], rec["B"]
        diff = "  <-- B PASS, A REVIEW" if (b["verdict"] == "PASS" and a["verdict"] == "REVIEW") else (
               "  <-- A PASS, B REVIEW" if (a["verdict"] == "PASS" and b["verdict"] == "REVIEW") else "")
        L.append(f"   {rec['img']:22s} r{rec['rep']} main({mvs})  A={a['verdict']:6s} "
                 f"B={b['verdict']:6s}  g41_b.bb={rec['g41_b_bb']} g54_a.bb={rec['g54_a_bb']}{diff}")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"helper_AB_real_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written {os.path.relpath(base, ROOT)}.txt / .json")


if __name__ == "__main__":
    main()
