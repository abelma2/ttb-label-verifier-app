"""Time + accuracy of gpt-4.1+N and gpt-5.4-mini+N on bold_safety, once each (per-model, not the
panel). Scores against the font-controlled manifest ground truth.

Accuracy is over the 3 PRIMARY classes (15 images): bold_compliant (header bold + body not bold),
boldbody (body bold -> must be caught), notbold (header not bold -> must be caught). titlecase is
observational only (its real defect is CAPS, out of scope for a bold-only read). The safety-
critical number is the high-confidence FALSE-PASS: a violation confidently read as compliant.

Reuses bold_prompt_safety's prompt N + scoring (_correct/_fp/_review). No production code touched.
Usage: python scripts/benchmarks/bold_safety_accuracy.py
Writes output/bold_safety_accuracy_<ts>.{txt,json}.
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

BS = os.path.join(ROOT, "bold_safety")
COMBOS = [("gpt-4.1", "N"), ("gpt-5.4-mini", "N")]
PRIMARY = ("bold_compliant", "boldbody", "notbold")


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def main():
    with open(os.path.join(BS, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    images = sorted(man.items(), key=lambda kv: kv[0])
    print(f"combos={COMBOS}  images={len(images)}  reps=1\n")
    report = {}
    for model, pv in COMBOS:
        key = f"{model}|{pv}"
        print(f"=== {key} ===")
        by_cls = defaultdict(lambda: {"n": 0, "correct": 0, "false_pass": 0, "hi_false_pass": 0, "review": 0})
        lat, per_image = [], []
        for fname, gt in images:
            var = gt["variant"]
            fields, dt, retries, err = B._call(model, B._prompt(pv), [os.path.join(BS, fname)])
            if not fields:
                print(f"  {fname:26s} ERROR {str(err)[:60]}")
                continue
            lat.append(dt)
            hb, bb = B._eff_header_bold(fields), B._eff_body_bold(fields)
            c = by_cls[var]
            c["n"] += 1
            corr = B._correct(var, fields)        # None for titlecase
            fp = B._fp(var, fields)
            rev = B._review(var, fields)
            if corr is True:
                c["correct"] += 1
            if fp:
                c["false_pass"] += 1
                if B._fp_conf(var, fields) == "high":
                    c["hi_false_pass"] += 1
            if rev:
                c["review"] += 1
            tag = ("OK" if corr is True else "FALSE-PASS" if fp else "review" if rev else
                   "obs" if corr is None else "wrong")
            per_image.append({"image": fname, "variant": var, "hb": hb, "bb": bb,
                              "hbc": fields.get("header_bold_confidence"),
                              "bbc": fields.get("body_bold_confidence"), "secs": dt, "tag": tag})
            print(f"  {fname:26s} [{var:14s}] {dt:5.2f}s hb={hb}/{(fields.get('header_bold_confidence') or '-')[:1]} "
                  f"bb={bb}/{(fields.get('body_bold_confidence') or '-')[:1]} -> {tag}")
        prim_n = sum(by_cls[v]["n"] for v in PRIMARY)
        prim_correct = sum(by_cls[v]["correct"] for v in PRIMARY)
        hifp = sum(c["hi_false_pass"] for c in by_cls.values())
        fp = sum(c["false_pass"] for c in by_cls.values())
        report[key] = {
            "model": model, "prompt": pv,
            "accuracy_primary": f"{prim_correct}/{prim_n}",
            "accuracy_pct": round(100 * prim_correct / prim_n, 1) if prim_n else None,
            "false_pass": fp, "high_conf_false_pass": hifp,
            "by_class": {v: dict(by_cls[v]) for v in by_cls},
            "lat_avg": round(sum(lat) / len(lat), 2) if lat else None, "lat_p50": _pct(lat, 50),
            "lat_p90": _pct(lat, 90), "lat_max": max(lat) if lat else None,
            "over_5s": sum(1 for x in lat if x > 5), "per_image": per_image,
        }
        r = report[key]
        print(f"  -> accuracy(primary 15) {r['accuracy_primary']} ({r['accuracy_pct']}%)  "
              f"false-pass {fp} (high-conf {hifp})  |  time avg {r['lat_avg']}s p50 {r['lat_p50']}s "
              f"max {r['lat_max']}s (>5s {r['over_5s']})\n")
    _write(report)


def _write(report):
    L = ["", "=" * 96, "bold_safety -- gpt-4.1+N vs gpt-5.4-mini+N (1x): TIME + ACCURACY (per-model)", "=" * 96,
         "accuracy over 15 PRIMARY images (bold_compliant/boldbody/notbold). high-conf FALSE-PASS = a "
         "violation confidently read as compliant (the dangerous error).", ""]
    L.append(f"{'model|prompt':18s} {'accuracy':10s} {'false-pass':11s} {'HI-falsepass':13s} "
             f"{'avg':6s} {'p50':6s} {'p90':6s} {'max':6s} {'>5s':4s}")
    L.append("-" * 90)
    for key, r in report.items():
        acc = f"{r['accuracy_primary']} ({r['accuracy_pct']}%)"
        L.append(f"{key:18s} {acc:14s} {str(r['false_pass']):11s} {str(r['high_conf_false_pass']):13s} "
                 f"{str(r['lat_avg']):6s} {str(r['lat_p50']):6s} {str(r['lat_p90']):6s} {str(r['lat_max']):6s} "
                 f"{str(r['over_5s']):4s}")
    L.append("")
    for key, r in report.items():
        L.append(f"--- {key}  (accuracy {r['accuracy_primary']} = {r['accuracy_pct']}%, "
                 f"high-conf false-pass {r['high_conf_false_pass']}, avg {r['lat_avg']}s) ---")
        for v in ("bold_compliant", "boldbody", "notbold", "titlecase"):
            c = r["by_class"].get(v)
            if not c:
                continue
            extra = (f"correct {c['correct']}/{c['n']}" if v in ("bold_compliant",) else
                     f"caught {c['n']-c['false_pass']}/{c['n']}, FALSE-PASS {c['false_pass']} "
                     f"(high-conf {c['hi_false_pass']})" if v in ("boldbody", "notbold") else
                     f"observational ({c['n']} imgs)")
            L.append(f"     {v:16s} {extra}   review {c['review']}")
        L.append("")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"bold_safety_accuracy_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written {os.path.relpath(base, ROOT)}.txt / .json")


if __name__ == "__main__":
    main()
