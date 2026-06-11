"""Compare the strict warning-bold policy `header_body_gate` (the production default when this
benchmark was written) against `medium_pass_gate` (promoted to the production default 2026-06-11
per course-staff guidance -- see BENCHMARK_NOTES.md) -- BENCHMARK ONLY (does not modify production
code or change the default).

Why this design: both policies are DETERMINISTIC functions of the same extracted
`government_warning` observation (header_bold/header_bold_confidence/body_bold/body_bold_confidence).
So we extract each image ONCE with the production `extract_fields` (full schema) and apply BOTH
policies to the IDENTICAL observation. Any verdict difference is then PURELY the policy -- zero
extraction nondeterminism between the two columns. We repeat N rounds to characterize the bold
read's run-to-run variance under each policy (bold is the one machine-unreliable warning property).

Cases (ground-truth-controlled where possible):
  clean baselines (3, front+back)            -> compliant; must NOT FAIL, PASS preferred
  adversarial 01/02/03/04                    -> 01 PASS; 02 titlecase / 03 notbold / 04 reworded FAIL
  bold_safety boldbody  (ALL-BOLD-BODY)      -> FAIL (remainder/body bold = 27 CFR 16.22 violation)
  bold_safety bold_compliant                 -> PASS (header bold, body not bold)
  bold_safety notbold                        -> FAIL (header not bold)
  bold_safety titlecase                      -> FAIL (caps violation)

Reports, per policy: clean-baseline PASS/REVIEW/FAIL distribution; detection on each violation
class (FAIL preferred, REVIEW = "not auto-passed", PASS = FALSE-PASS); the false-pass count; and
the exact (case, round) cells where medium_pass_gate's verdict differs from header_body_gate's.

Run (calls the real model -- needs an API key, costs money):
  python scripts/benchmarks/medium_pass_gate_compare.py                 # default 3 rounds
  python scripts/benchmarks/medium_pass_gate_compare.py --rounds 5
  python scripts/benchmarks/medium_pass_gate_compare.py --bs-distortions clean,lowres
"""
import json
import os
import re
import sys
import time
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

import verification
from verification import PASS, REVIEW, FAIL, _check_warning
from extraction import extract_fields

ADV = os.path.join(ROOT, "adversarial")
BASE = os.path.join(ROOT, "test_labels", "baseline_labels")
BS = os.path.join(ROOT, "bold_safety")

POLICIES = ("header_body_gate", "medium_pass_gate")
_AB = {PASS: "PASS", REVIEW: "REVIEW", FAIL: "FAIL"}


def _media_type(path):
    return "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"


def _load_key():
    if os.environ.get("OPENAI_API_KEY"):
        return True
    secrets = os.path.join(ROOT, ".streamlit", "secrets.toml")
    if os.path.exists(secrets):
        with open(secrets, encoding="utf-8") as fh:
            m = re.search(r'OPENAI_API_KEY\s*=\s*"([^"]+)"', fh.read())
        if m and m.group(1) and m.group(1) != "sk-...":
            os.environ["OPENAI_API_KEY"] = m.group(1)
    return bool(os.environ.get("OPENAI_API_KEY"))


def _arg(args, flag, default):
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else default


def _cases(bs_distortions):
    """Return [(id, group, [paths], expected)]. expected: PASS / FAIL / None(=compliant-baseline)."""
    cases = [
        ("baseline_1", "clean_baseline",
         [os.path.join(BASE, "baseline_1_Front.png"), os.path.join(BASE, "baseline_1_Other.png")], None),
        ("baseline_2", "clean_baseline",
         [os.path.join(BASE, "baseline_2_Front.png"), os.path.join(BASE, "baseline_2_Other.png")], None),
        ("baseline_3", "clean_baseline",
         [os.path.join(BASE, "baseline_3_Front.png"), os.path.join(BASE, "baseline_3_Other.png")], None),
        ("adv_01_compliant", "adv_compliant", [os.path.join(ADV, "01_compliant.png")], PASS),
        ("adv_02_titlecase", "titlecase", [os.path.join(ADV, "02_titlecase.png")], FAIL),
        ("adv_03_notbold", "notbold", [os.path.join(ADV, "03_notbold.png")], FAIL),
        ("adv_04_reworded", "reworded", [os.path.join(ADV, "04_reworded.png")], FAIL),
    ]
    # bold_safety: variant -> (group, expected verdict)
    bs_variants = {
        "boldbody": ("all_bold_body", FAIL),
        "bold_compliant": ("bold_compliant", PASS),
        "notbold": ("notbold", FAIL),
        "titlecase": ("titlecase", FAIL),
    }
    ext = {"clean": "png", "lowres": "png", "curved": "png", "rotblur": "png", "jpeg": "jpg"}
    for variant, (group, expected) in bs_variants.items():
        for dist in bs_distortions:
            fn = "%s__%s.%s" % (variant, dist, ext.get(dist, "png"))
            p = os.path.join(BS, fn)
            if os.path.exists(p):
                cases.append(("bs_%s_%s" % (variant, dist), group, [p], expected))
    return [c for c in cases if all(os.path.exists(p) for p in c[2])]


def _verdict(policy, gw):
    """Apply ONE policy to an already-extracted government_warning observation."""
    old = verification.WARNING_BOLD_POLICY
    verification.WARNING_BOLD_POLICY = policy
    try:
        r = _check_warning(gw)
    finally:
        verification.WARNING_BOLD_POLICY = old
    return r.status, r.reason


def _obs(gw):
    if not isinstance(gw, dict):
        return {}
    return {k: gw.get(k) for k in ("header_bold", "header_bold_confidence",
                                   "body_bold", "body_bold_confidence")}


def main():
    args = sys.argv[1:]
    rounds = int(_arg(args, "--rounds", "3"))
    bs_distortions = [d.strip() for d in _arg(args, "--bs-distortions", "clean,lowres").split(",") if d.strip()]

    if not _load_key():
        sys.exit("ERROR: no OpenAI key (env OPENAI_API_KEY or .streamlit/secrets.toml).")

    cases = _cases(bs_distortions)
    print("rounds=%d  cases=%d  policies=%s  bs_distortions=%s\n"
          % (rounds, len(cases), ",".join(POLICIES), ",".join(bs_distortions)))

    # records[case_id] = {"group","expected","rounds":[{obs, verdicts:{policy:(status,reason)}, seconds}]}
    records = {}
    for cid, group, paths, expected in cases:
        round_recs = []
        for i in range(rounds):
            print("  %-22s round %d/%d ..." % (cid, i + 1, rounds), flush=True)
            imgs = [(open(p, "rb").read(), _media_type(p)) for p in paths]
            t = time.perf_counter()
            try:
                extracted = extract_fields(imgs)
                gw = extracted.get("government_warning")
                err = None
            except Exception as exc:
                gw, err = None, str(exc)[:160]
            dt = round(time.perf_counter() - t, 2)
            verdicts = {}
            if err is None:
                for pol in POLICIES:
                    st, why = _verdict(pol, gw)
                    verdicts[pol] = {"status": st, "reason": why}
            round_recs.append({"obs": _obs(gw), "verdicts": verdicts, "seconds": dt, "error": err})
        records[cid] = {"group": group, "expected": expected, "rounds": round_recs}

    report = _scorecard(records, rounds, bs_distortions)
    print(report)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(OUT_DIR, "medium_pass_gate_compare_%s.txt" % stamp), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    with open(os.path.join(OUT_DIR, "medium_pass_gate_compare_%s.json" % stamp), "w", encoding="utf-8") as fh:
        json.dump({"rounds": rounds, "bs_distortions": bs_distortions, "records": records}, fh, indent=2,
                  ensure_ascii=False)
    print("\nWritten to: output/medium_pass_gate_compare_%s.txt / .json" % stamp)


def _dist(records, group, policy):
    c = Counter()
    for cid, rec in records.items():
        if rec["group"] != group:
            continue
        for rr in rec["rounds"]:
            v = rr["verdicts"].get(policy)
            c[_AB.get(v["status"], "ERR") if v else "ERR"] += 1
    return c


def _scorecard(records, rounds, bs_distortions):
    L = ["", "=" * 100, "MEDIUM_PASS_GATE vs HEADER_BODY_GATE  (same extraction, both policies)", "=" * 100,
         "rounds=%d  bs_distortions=%s" % (rounds, ",".join(bs_distortions)),
         "Both policies applied to the IDENTICAL extracted observation -> any diff is the policy, not model noise.",
         ""]

    # 1. clean-baseline distribution
    L.append("1) CLEAN BASELINE distribution (compliant labels; PASS preferred, must not FAIL):")
    for pol in POLICIES:
        d = _dist(records, "clean_baseline", pol)
        tot = sum(d.values())
        L.append("   %-18s PASS=%d  REVIEW=%d  FAIL=%d   (of %d)"
                 % (pol, d.get("PASS", 0), d.get("REVIEW", 0), d.get("FAIL", 0), tot))

    # 2. violation detection + compliant pass-through by class
    L.append("")
    L.append("2) DETECTION by class (PASS on a violation class = FALSE-PASS; PASS on a compliant class is good):")
    L.append("   %-16s %-26s %-26s" % ("class", "header_body_gate", "medium_pass_gate"))
    groups = ["adv_compliant", "bold_compliant", "titlecase", "notbold", "reworded", "all_bold_body"]
    violation_groups = {"titlecase", "notbold", "reworded", "all_bold_body"}
    for g in groups:
        present = any(r["group"] == g for r in records.values())
        if not present:
            continue
        cells = []
        for pol in POLICIES:
            d = _dist(records, g, pol)
            tot = sum(d.values())
            cells.append("P=%d R=%d F=%d /%d" % (d.get("PASS", 0), d.get("REVIEW", 0), d.get("FAIL", 0), tot))
        tag = "  <- violation" if g in violation_groups else "  <- compliant"
        L.append("   %-16s %-26s %-26s%s" % (g, cells[0], cells[1], tag))

    # 3. false-pass risk: any should-FAIL case that PASSED, per policy
    L.append("")
    L.append("3) FALSE-PASS RISK (a should-FAIL violation that returned PASS):")
    for pol in POLICIES:
        fps = []
        for cid, rec in records.items():
            if rec["expected"] != FAIL:
                continue
            for i, rr in enumerate(rec["rounds"]):
                v = rr["verdicts"].get(pol)
                if v and v["status"] == PASS:
                    fps.append((cid, i + 1, rr["obs"]))
        L.append("   %-18s false-passes=%d" % (pol, len(fps)))
        for cid, rnd, obs in fps:
            L.append("       %s round %d  obs=%s" % (cid, rnd, obs))

    # 4. exact cells where the two policies DISAGREE (the whole point of the variant)
    L.append("")
    L.append("4) CELLS WHERE medium_pass_gate DIFFERS FROM header_body_gate (same observation):")
    diffs = 0
    for cid, rec in records.items():
        for i, rr in enumerate(rec["rounds"]):
            a = rr["verdicts"].get("header_body_gate")
            b = rr["verdicts"].get("medium_pass_gate")
            if a and b and a["status"] != b["status"]:
                diffs += 1
            L_diff = a and b and a["status"] != b["status"]
            if L_diff:
                L.append("   %-22s round %d  %s -> %s   obs=%s"
                         % (cid, i + 1, _AB.get(a["status"]), _AB.get(b["status"]), rr["obs"]))
    if diffs == 0:
        L.append("   (none this run)")
    L.append("")
    L.append("   Summary: %d of %d total (case,round) cells differ; by construction every diff is a"
             % (diffs, sum(len(r["rounds"]) for r in records.values())))
    L.append("   header_body_gate REVIEW that medium_pass_gate turns into PASS (medium-confidence,")
    L.append("   both-rules-satisfied), OR -- on a violation -- a medium-confidence misread that")
    L.append("   medium_pass_gate would PASS where header_body_gate reviews (the false-pass surface).")
    return "\n".join(L)


if __name__ == "__main__":
    main()
