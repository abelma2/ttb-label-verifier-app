"""High-only triple gate -- 3 HELPER-MODEL variants compared on bold_safety, 1x. No production code
touched. Reuses bold_prompt_safety (_call, prompt A) + prompt_S_test (PROMPT_S).

main is the SAME for all configs (gpt-5.4-mini + prompt A); only the 2 helpers vary:
  A. helper_1 = gpt-4.1 + S      helper_2 = gpt-4.1 + S          (both gpt-4.1)
  B. helper_1 = gpt-4.1 + S      helper_2 = gpt-5.4-mini + S     (one of each)
  C. helper_1 = gpt-5.4-mini + S helper_2 = gpt-5.4-mini + S     (both gpt-5.4-mini)

CONTROLLED: the shared `main` read is run ONCE per image and reused across A/B/C, so the only
variable between configs is the helper composition (no main-read sampling noise). The distinct
helper reads needed are 2x gpt-4.1+S and 2x gpt-5.4-mini+S; B reuses one of each. So 5 reads per
image total, run CONCURRENTLY (bounded to one image at a time -> no global rate-limit distortion).
wall(config) = max of THAT config's 3 read latencies.

Gate (high-only, exactly as specified):
  PASS  = main hb=True/high AND main bb=False/high AND main leg=good AND no helper bb=True
          AND no helper contradicts the header (helper hb=False AT HIGH conf).
  FAIL  = inert (no independent non-S/crop witness; S helpers cannot create FAIL).
  REVIEW= any helper bb=True; or main bb=True; or main not clean-high; or a helper hb=False/high;
          or poor/limited legibility; or timeout/error.

Score vs bold_safety/manifest.json: boldbody/notbold caught-vs-false-pass, bold_compliant
pass/review/false-fail, titlecase separately (out-of-scope for a bold gate), total violation leaks,
latency avg/p50/max/>5s, per-image decisions. Writes output/helper_models_boldsafety_<ts>.{txt,json}.
Usage: python scripts/benchmarks/helper_models_boldsafety.py
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

PROMPT_A = B._prompt("A")
PROMPT_S = PS.PROMPT_S
VIOLATIONS = ("boldbody", "notbold")

# the 5 distinct reads per image (role -> (model, prompt))
READS = {
    "main":  ("gpt-5.4-mini", PROMPT_A),
    "g41_a": ("gpt-4.1", PROMPT_S),
    "g41_b": ("gpt-4.1", PROMPT_S),
    "g54_a": ("gpt-5.4-mini", PROMPT_S),
    "g54_b": ("gpt-5.4-mini", PROMPT_S),
}
# config -> (main_role, helper1_role, helper2_role)
CONFIGS = {
    "A: 2x gpt-4.1+S":            ("main", "g41_a", "g41_b"),
    "B: gpt-4.1+S & 5.4-mini+S":  ("main", "g41_a", "g54_a"),
    "C: 2x gpt-5.4-mini+S":       ("main", "g54_a", "g54_b"),
}


def _vals(f):
    return (B._eff_header_bold(f), B._eff_body_bold(f),
            f.get("header_bold_confidence"), f.get("body_bold_confidence"), f.get("legibility"))


def decide(main, h1, h2):
    """High-only gate, exactly per spec. FAIL is inert. Returns (verdict, reasons)."""
    if not main or not h1 or not h2:
        return "REVIEW", ["timeout/error"]
    hb_m, bb_m, hbc_m, bbc_m, leg_m = _vals(main)
    bb_h1, bb_h2 = B._eff_body_bold(h1), B._eff_body_bold(h2)
    hb_h1, hbc_h1 = B._eff_header_bold(h1), h1.get("header_bold_confidence")
    hb_h2, hbc_h2 = B._eff_header_bold(h2), h2.get("header_bold_confidence")
    # FAIL inert (no independent non-S/crop witness)
    reasons = []
    if bb_h1 is True or bb_h2 is True:
        reasons.append("helper body-bold veto")
    if bb_m is True:
        reasons.append("main body-bold")
    if hb_m is True and ((hb_h1 is False and hbc_h1 == "high") or (hb_h2 is False and hbc_h2 == "high")):
        reasons.append("helper header contradiction (false/high)")
    if leg_m in ("poor", "limited"):
        reasons.append("poor/limited legibility")
    main_clean = (hb_m is True and hbc_m == "high" and bb_m is False and bbc_m == "high" and leg_m == "good")
    if not main_clean and not reasons:
        reasons.append("main not clean high-conf")
    if reasons:
        return "REVIEW", reasons
    return "PASS", ["clean"]


def _score(variant, verdict):
    if variant in VIOLATIONS:
        return "FALSE-PASS" if verdict == "PASS" else "caught"
    if variant == "bold_compliant":
        return "correct-PASS" if verdict == "PASS" else ("over-review" if verdict == "REVIEW" else "FALSE-FAIL")
    return "obs-" + verdict


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def _read_image(path):
    """Run the 5 distinct reads concurrently for ONE image. role -> (fields, dt, err)."""
    out = {}
    with ThreadPoolExecutor(max_workers=len(READS)) as pool:
        futs = {role: pool.submit(B._call, model, prompt, [path]) for role, (model, prompt) in READS.items()}
        for role, fut in futs.items():
            fields, dt, _retries, err = fut.result()
            out[role] = (fields, dt, err)
    return out


def main():
    images = B._bs_images()
    print(f"helper-model configs A/B/C on bold_safety  |  {len(images)} images x 1  "
          f"(5 reads/image, bounded to one image at a time)\n")
    tally = {c: defaultdict(lambda: defaultdict(int)) for c in CONFIGS}   # config -> variant -> outcome -> n
    walls = {c: [] for c in CONFIGS}
    leaks = {c: [] for c in CONFIGS}
    records = []

    for im in images:
        var = im["variant"]
        reads = _read_image(im["path"])
        row = {"image": im["name"], "variant": var}
        for cname, (mr, h1r, h2r) in CONFIGS.items():
            m_f, m_dt, _ = reads[mr]
            h1_f, h1_dt, _ = reads[h1r]
            h2_f, h2_dt, _ = reads[h2r]
            dts = [d for d in (m_dt, h1_dt, h2_dt) if d is not None]
            wall = max(dts) if dts else None
            if wall is not None:
                walls[cname].append(wall)
            verdict, reasons = decide(m_f, h1_f, h2_f)
            outcome = _score(var, verdict)
            tally[cname][var][outcome] += 1
            if var in VIOLATIONS and outcome == "FALSE-PASS":
                leaks[cname].append(im["name"])
            row[cname] = {"verdict": verdict, "outcome": outcome, "wall": wall,
                          "reasons": reasons,
                          "h1_bb": (None if not h1_f else B._eff_body_bold(h1_f)),
                          "h2_bb": (None if not h2_f else B._eff_body_bold(h2_f))}
        # main read context (shared)
        m_f = reads["main"][0]
        row["main"] = None if not m_f else list(_vals(m_f))
        records.append(row)
        mv = row["main"]
        mvs = "ERR" if mv is None else f"hb={mv[0]}/{(mv[2] or '-')[:1]} bb={mv[1]}/{(mv[3] or '-')[:1]}"
        cells = "  ".join(f"{c[0]}={row[c]['verdict'][:4]}" for c in CONFIGS)
        print(f"  {im['name']:26s} [{var:14s}] main({mvs})  {cells}")

    report = {
        "configs": list(CONFIGS.keys()),
        "tally": {c: {v: dict(o) for v, o in d.items()} for c, d in tally.items()},
        "violation_leaks": leaks,
        "latency": {c: {"avg": round(sum(w) / len(w), 2) if w else None, "p50": _pct(w, 50),
                        "max": max(w) if w else None, "over_5s": sum(1 for x in w if x > 5)}
                    for c, w in walls.items()},
        "records": records,
    }
    _summary(report)
    _write(report)


def _viol(t):
    caught = t.get("caught", 0); fp = t.get("FALSE-PASS", 0)
    return caught, fp, caught + fp


def _summary(r):
    print("\n  ===== SUMMARY (caught = routed to review = good; FALSE-PASS = leaked violation) =====")
    for c in r["configs"]:
        t = r["tally"][c]
        bb_c, bb_fp, bb_n = _viol(t.get("boldbody", {}))
        nb_c, nb_fp, nb_n = _viol(t.get("notbold", {}))
        bc = t.get("bold_compliant", {})
        lat = r["latency"][c]
        total_leak = bb_fp + nb_fp
        print(f"  {c:28s} boldbody {bb_c}/{bb_n} (leak {bb_fp})  notbold {nb_c}/{nb_n} (leak {nb_fp})  "
              f"TOTAL LEAKS {total_leak}")
        print(f"  {'':28s} bold_compliant: PASS {bc.get('correct-PASS',0)} review {bc.get('over-review',0)} "
              f"false-fail {bc.get('FALSE-FAIL',0)}  |  wall avg {lat['avg']}s p50 {lat['p50']}s "
              f"max {lat['max']}s >5s {lat['over_5s']}")
    print()


def _write(r):
    L = ["", "=" * 104, "HELPER-MODEL variants A/B/C -- high-only triple gate on bold_safety, 1x", "=" * 104,
         "main = gpt-5.4-mini+A (shared across configs); only the 2 helpers vary. "
         "caught=routed to review (good); FALSE-PASS=leaked violation; bold_compliant PASS=correct.",
         "titlecase = caps issue, OUT OF SCOPE for a bold gate (reported separately).", ""]
    L.append(f"{'config':28s} {'boldbody':16s} {'notbold':16s} {'TOTAL LEAKS':12s} "
             f"{'bold_compliant(P/R/F)':22s} {'wall p50/max':14s}")
    L.append("-" * 104)
    for c in r["configs"]:
        t = r["tally"][c]
        bb_c, bb_fp, bb_n = _viol(t.get("boldbody", {}))
        nb_c, nb_fp, nb_n = _viol(t.get("notbold", {}))
        bc = t.get("bold_compliant", {})
        lat = r["latency"][c]
        bbs = f"{bb_c}/{bb_n} leak{bb_fp}"
        nbs = f"{nb_c}/{nb_n} leak{nb_fp}"
        bcs = f"{bc.get('correct-PASS',0)}/{bc.get('over-review',0)}/{bc.get('FALSE-FAIL',0)}"
        lats = f"{lat['p50']}/{lat['max']}s"
        L.append(f"{c:28s} {bbs:16s} {nbs:16s} {str(bb_fp+nb_fp):12s} {bcs:22s} {lats:14s}")
    L.append("")
    for c in r["configs"]:
        t = r["tally"][c]
        tc = t.get("titlecase", {})
        L.append(f"   {c}: titlecase {dict(tc)}   violation leaks: {r['violation_leaks'][c] or 'none'}")
    L.append("")
    L.append("per-image decisions (verdict per config; main = shared gpt-5.4-mini:A read):")
    for row in r["records"]:
        mv = row["main"]
        mvs = "ERR" if mv is None else f"hb={mv[0]}/{(mv[2] or '-')[:1]} bb={mv[1]}/{(mv[3] or '-')[:1]} leg={mv[4]}"
        L.append(f"   {row['image']:26s} [{row['variant']:14s}] main({mvs})")
        for c in r["configs"]:
            cell = row[c]
            L.append(f"        {c:28s} -> {cell['verdict']:7s} [{cell['outcome']:12s}] "
                     f"h1.bb={cell['h1_bb']} h2.bb={cell['h2_bb']} wall={cell['wall']}s  "
                     f"({'; '.join(cell['reasons'])})")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"helper_models_boldsafety_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written {os.path.relpath(base, ROOT)}.txt / .json")


if __name__ == "__main__":
    main()
