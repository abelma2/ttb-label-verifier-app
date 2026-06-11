"""Prompt S ('traps') on COMPLIANT labels (baseline_labels + clearer_baseline_labels), gpt-5.4-mini
and gpt-4.1, 1x. Complementary to the bold_safety run: there S aced the VIOLATIONS; here we check
whether its aggressive "if both bold -> body_bold=true" trap FALSE-FLAGS compliant bodies.

These labels are compliant (back warning = bold header + non-bold body; front = no warning), so:
  backs : correct = compliant read (header_bold True, body_bold False). body_bold=True = a FALSE
          body-bold flag (the prompt-S risk); header_bold=False = missed header; null/low = review.
  fronts: correct = no warning.
Tracks time. No production code touched; PROMPT_S reused verbatim from prompt_S_test.

Usage: python scripts/benchmarks/prompt_S_compliant.py
Writes output/prompt_S_compliant_<ts>.{txt,json}.
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
import prompt_S_test as PS   # reuse the exact PROMPT_S verbatim

PROMPT_S = PS.PROMPT_S
MODELS = ["gpt-5.4-mini", "gpt-4.1"]
FOLDERS = ["baseline_labels", "clearer_baseline_labels", "real_labels"]


def _gather():
    """All images across the compliant folders. _Other = warning-bearing back; _Front = no warning."""
    out = []
    for folder in FOLDERS:
        d = os.path.join(ROOT, "test_labels", folder)
        for fname in sorted(os.listdir(d)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                out.append((folder, fname, os.path.join(d, fname), "_other" in fname.lower()))
    return out


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def main():
    images = _gather()   # (folder, fname, path, is_back)
    print(f"prompt=S  models={MODELS}  images={len(images)} ({len(FOLDERS)} compliant folders)  reps=1\n")
    report = {}
    for model in MODELS:
        print(f"=== {model} + S ===")
        backs = {"n": 0, "compliant_pass": 0, "false_body_bold": 0, "missed_header": 0, "review": 0}
        fronts = {"n": 0, "no_warning_ok": 0, "hallucinated": 0}
        lat, per_image, by_folder = [], [], defaultdict(lambda: {"n": 0, "pass": 0, "false_bb": 0})
        for folder, fname, path, is_back in images:
            fields, dt, retries, err = B._call(model, PROMPT_S, [path])
            if not fields:
                print(f"  {folder}/{fname:24s} ERROR {str(err)[:50]}")
                continue
            lat.append(dt)
            wp = fields.get("warning_present")
            hb, bb = B._eff_header_bold(fields), B._eff_body_bold(fields)
            hbc, bbc, leg = (fields.get("header_bold_confidence"), fields.get("body_bold_confidence"),
                             fields.get("legibility"))
            if is_back:
                backs["n"] += 1
                by_folder[folder]["n"] += 1
                if bb is True:
                    backs["false_body_bold"] += 1; by_folder[folder]["false_bb"] += 1; tag = "FALSE-body-bold"
                elif hb is False:
                    backs["missed_header"] += 1; tag = "missed-header"
                elif wp and hb is True and bb is False and hbc == "high" and bbc == "high":
                    backs["compliant_pass"] += 1; by_folder[folder]["pass"] += 1; tag = "compliant-PASS"
                elif leg == "poor" or bb is None or hb is None or bbc == "low" or hbc == "low":
                    backs["review"] += 1; tag = "review"
                else:
                    backs["review"] += 1; tag = "review(med)"
            else:
                fronts["n"] += 1
                if wp is False:
                    fronts["no_warning_ok"] += 1; tag = "no-warning-OK"
                else:
                    fronts["hallucinated"] += 1; tag = "HALLUCINATED-warning"
            per_image.append({"img": f"{folder}/{fname}", "tag": tag, "hb": hb, "bb": bb,
                              "hbc": hbc, "bbc": bbc, "secs": dt,
                              "rel": fields.get("header_body_relationship")})
            print(f"  {folder}/{fname:24s} {dt:5.2f}s hb={hb}/{(hbc or '-')[:1]} bb={bb}/{(bbc or '-')[:1]} "
                  f"rel={fields.get('header_body_relationship')} -> {tag}")
        report[model] = {
            "backs": backs, "fronts": fronts, "by_folder": {k: dict(v) for k, v in by_folder.items()},
            "lat_avg": round(sum(lat) / len(lat), 2) if lat else None, "lat_p50": _pct(lat, 50),
            "lat_max": max(lat) if lat else None, "over_5s": sum(1 for x in lat if x > 5),
            "per_image": per_image,
        }
        r = report[model]
        print(f"  -> backs: compliant-PASS {backs['compliant_pass']}/{backs['n']}, "
              f"FALSE-body-bold {backs['false_body_bold']}, missed-header {backs['missed_header']}, "
              f"review {backs['review']}  |  fronts no-warning {fronts['no_warning_ok']}/{fronts['n']}  |  "
              f"time avg {r['lat_avg']}s p50 {r['lat_p50']}s max {r['lat_max']}s\n")
    _write(report)


def _write(report):
    L = ["", "=" * 96, "PROMPT S on COMPLIANT labels (baseline + clearer_baseline, 1x): TIME + ACCURACY",
         "=" * 96,
         "labels are COMPLIANT -> correct = compliant-PASS on backs; FALSE-body-bold = the prompt-S "
         "over-flag risk (body read bold when it isn't); fronts: correct = no warning.", ""]
    L.append(f"{'model + S':16s} {'compliant-PASS':16s} {'FALSE-body-bold':16s} {'missed-hdr':11s} "
             f"{'review':8s} {'frontOK':9s} {'avg':6s} {'p50':6s} {'max':6s}")
    L.append("-" * 96)
    for m, r in report.items():
        b, f = r["backs"], r["fronts"]
        cp = f"{b['compliant_pass']}/{b['n']}"
        fo = f"{f['no_warning_ok']}/{f['n']}"
        L.append(f"{m:16s} {cp:16s} {str(b['false_body_bold']):16s} {str(b['missed_header']):11s} "
                 f"{str(b['review']):8s} {fo:9s} {str(r['lat_avg']):6s} {str(r['lat_p50']):6s} {str(r['lat_max']):6s}")
    L.append("")
    for m, r in report.items():
        L.append(f"--- {m} + S  (backs: {r['backs']}) ---")
        L.append(f"   per-folder backs: {r['by_folder']}")
        for pi in r["per_image"]:
            L.append(f"     {pi['img']:42s} {pi['secs']}s hb={pi['hb']}/{(pi['hbc'] or '-')[:1]} "
                     f"bb={pi['bb']}/{(pi['bbc'] or '-')[:1]} rel={pi['rel']} -> {pi['tag']}")
        L.append("")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"prompt_S_compliant_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written {os.path.relpath(base, ROOT)}.txt / .json")


if __name__ == "__main__":
    main()
