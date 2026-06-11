"""Stage-2 sanity: the Stage-1 winners (gpt-4.1 + N, gpt-5.4-mini + N) on the CLEAN
clearer_baseline_labels, 3x. Tracks accuracy AND latency.

These are COMPLIANT labels: the back (_Other) carries a standard warning (BOLD header, NON-bold
body); the front (_Front) has NO warning. So there is no violation to catch -- accuracy here is:
  - backs : a CORRECT COMPLIANT read = warning_present + header_bold True + body_bold False
            (would PASS). A false body_bold=True or header_bold=False would FALSE-FAIL a clean label.
  - fronts: correct = warning_present False (no warning hallucinated).
Also: these are large ~6MB full labels (not tiny crops), so this tests whether the read stays fast.

Reuses the Stage-1 prompt N + normalized schema + _call (429 retry/backoff). No production code touched.
Usage: python scripts/benchmarks/stage2_baseline.py
Writes output/stage2_baseline_<ts>.{txt,json}.
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict
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


def _arg(flag, default):
    a = sys.argv[1:]
    return a[a.index(flag) + 1] if flag in a and a.index(flag) + 1 < len(a) else default


COMBOS = [("gpt-4.1", "N"), ("gpt-5.4-mini", "N")]
DIRNAME = _arg("--dir", "clearer_baseline_labels")   # e.g. baseline_labels
PREFIX = _arg("--prefix", "clear_baseline")          # e.g. baseline
REPS = int(_arg("--reps", "3"))
DIR = os.path.join(ROOT, "test_labels", DIRNAME)
IMAGES = [f"{PREFIX}_{n}_{side}.png" for n in (1, 2, 3) for side in ("Front", "Other")]


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def main():
    print(f"combos={COMBOS}  images={len(IMAGES)}  reps={REPS}\n")
    report = {}
    for model, pv in COMBOS:
        key = f"{model}|{pv}"
        print(f"=== {key} ===")
        per_image = defaultdict(list)
        lat, errs, rl = [], 0, 0
        for fname in IMAGES:
            path = os.path.join(DIR, fname)
            is_back = fname.endswith("Other.png")
            for rep in range(1, REPS + 1):
                fields, dt, retries, err = B._call(model, B._prompt(pv), [path])
                if not fields:
                    errs += 1
                    rl += int((err or "").startswith("RATELIMIT"))
                    per_image[fname].append({"error": err})
                    print(f"  {fname:26s} rep{rep}: ERROR {str(err)[:70]}")
                    continue
                lat.append(dt)
                wp = fields.get("warning_present")
                hb, bb = B._eff_header_bold(fields), B._eff_body_bold(fields)
                rec = {"warning_present": wp, "header_bold": hb, "body_bold": bb,
                       "hb_conf": fields.get("header_bold_confidence"),
                       "bb_conf": fields.get("body_bold_confidence"),
                       "legibility": fields.get("legibility"), "seconds": dt,
                       "basis": fields.get("short_basis")}
                per_image[fname].append(rec)
                if is_back:
                    verdict = ("COMPLIANT-read" if (wp and hb is True and bb is False)
                               else "FALSE-body-bold" if bb is True
                               else "missed-header-bold" if hb is False
                               else "unclear/review")
                else:
                    verdict = "no-warning-OK" if wp is False else "HALLUCINATED-warning"
                print(f"  {fname:26s} rep{rep}: {dt:5.2f}s  present={wp} hb={hb}/{rec['hb_conf']} "
                      f"bb={bb}/{rec['bb_conf']}  -> {verdict}")
        # aggregate
        backs = [r for f in IMAGES if f.endswith("Other.png") for r in per_image[f] if "error" not in r]
        fronts = [r for f in IMAGES if f.endswith("Front.png") for r in per_image[f] if "error" not in r]
        compliant_read = sum(1 for r in backs if r["warning_present"] and r["header_bold"] is True and r["body_bold"] is False)
        false_body_bold = sum(1 for r in backs if r["body_bold"] is True)
        missed_header = sum(1 for r in backs if r["header_bold"] is False)
        back_review = sum(1 for r in backs if r["body_bold"] is None or r["header_bold"] is None
                          or r["bb_conf"] == "low" or r["legibility"] == "poor")
        front_ok = sum(1 for r in fronts if r["warning_present"] is False)
        front_halluc = sum(1 for r in fronts if r["warning_present"] is True)
        # stability per image: do the per-rep (hb, bb) reads agree?
        stable = 0
        for f in IMAGES:
            reads = [(r.get("header_bold"), r.get("body_bold")) for r in per_image[f] if "error" not in r]
            if len(reads) >= 2 and len(set(reads)) == 1:
                stable += 1
        report[key] = {
            "model": model, "prompt": pv, "backs_n": len(backs), "fronts_n": len(fronts),
            "compliant_read": compliant_read, "false_body_bold": false_body_bold,
            "missed_header_bold": missed_header, "back_review": back_review,
            "front_no_warning_ok": front_ok, "front_hallucinated": front_halluc,
            "stable_images": stable, "n_images": len(IMAGES),
            "lat_avg": round(sum(lat) / len(lat), 2) if lat else None, "lat_p50": _pct(lat, 50),
            "lat_p90": _pct(lat, 90), "lat_max": max(lat) if lat else None,
            "over_5s": sum(1 for x in lat if x > 5), "errors": errs, "ratelimit": rl,
            "per_image": {f: per_image[f] for f in IMAGES},
        }
        r = report[key]
        print(f"  -> backs: compliant-read {compliant_read}/{r['backs_n']}, false-body-bold "
              f"{false_body_bold}, missed-header {missed_header}, review {back_review}  |  "
              f"fronts: no-warning {front_ok}/{r['fronts_n']} (halluc {front_halluc})  |  "
              f"stable {stable}/{r['n_images']}  |  lat avg {r['lat_avg']}s p50 {r['lat_p50']}s "
              f"max {r['lat_max']}s (>5s: {r['over_5s']})  errors {errs}\n")
    _write(report)


def _write(report):
    L = ["", "=" * 100, f"BASELINE SANITY -- winners on {DIRNAME} (CLEAN/compliant), {REPS}x",
         "=" * 100,
         "backs are COMPLIANT (bold header, non-bold body): correct = compliant-read; a false body_bold "
         "or missed header would FALSE-FAIL a clean label. fronts: correct = no warning.", ""]
    hdr = (f"{'model|prompt':18s} {'cmpRead':12s} {'falseBB':8s} {'missHdr':8s} {'review':7s} "
           f"{'frontOK':9s} {'stable':7s} {'avg':6s} {'p50':6s} {'p90':6s} {'max':6s} {'>5s':4s} {'err':4s}")
    L.append(hdr); L.append("-" * len(hdr))
    for key, r in report.items():
        cmpread = f"{r['compliant_read']}/{r['backs_n']}"
        frontok = f"{r['front_no_warning_ok']}/{r['fronts_n']}"
        stablestr = f"{r['stable_images']}/{r['n_images']}"
        L.append(f"{key:18s} {cmpread:12s} {str(r['false_body_bold']):8s} {str(r['missed_header_bold']):8s} "
                 f"{str(r['back_review']):7s} {frontok:9s} {stablestr:7s} "
                 f"{str(r['lat_avg']):6s} {str(r['lat_p50']):6s} {str(r['lat_p90']):6s} {str(r['lat_max']):6s} "
                 f"{str(r['over_5s']):4s} {str(r['errors']):4s}")
    L.append("")
    for key, r in report.items():
        L.append(f"--- {key} ---")
        for f, reads in r["per_image"].items():
            cells = []
            for rd in reads:
                if "error" in rd:
                    cells.append("ERR")
                else:
                    cells.append(f"{rd['seconds']}s pres={rd['warning_present']} hb={rd['header_bold']}/{rd['hb_conf'][:1] if rd['hb_conf'] else '-'} bb={rd['body_bold']}/{rd['bb_conf'][:1] if rd['bb_conf'] else '-'}")
            L.append(f"   {f:26s} " + "  |  ".join(cells))
        L.append("")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"stage2_baseline_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"Written {os.path.relpath(base, ROOT)}.txt / .json")


if __name__ == "__main__":
    main()
