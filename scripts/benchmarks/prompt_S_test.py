"""Test a refined 'traps' bold prompt (prompt S) on bold_safety, 1x, gpt-5.4-mini + gpt-4.1.
Scores accuracy vs the manifest ground truth + time. The prompt is used VERBATIM (it already
matches the normalized schema); _call enforces the strict schema. No production code touched.

Accuracy over the 3 PRIMARY classes (15 imgs): bold_compliant / boldbody / notbold. The headline
is the boldbody catch rate -- this prompt's explicit "if both bold -> body_bold=true, same"
instruction is aimed straight at the body-bold misses that sank prompt N.

Usage: python scripts/benchmarks/prompt_S_test.py
Writes output/prompt_S_test_<ts>.{txt,json}.
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
MODELS = ["gpt-5.4-mini", "gpt-4.1"]
PRIMARY = ("bold_compliant", "boldbody", "notbold")

PROMPT_S = """Inspect only the visible alcohol health warning text.

Find the words "GOVERNMENT WARNING" if visible. Treat those words as the header. Treat the sentence text after the colon as the body.

Report only visible stroke weight. Do not judge legal compliance. Do not use expectations about how alcohol warnings are usually formatted. Do not infer that the body is regular just because the header appears bold.

Check two things separately:
1. Relative weight: are the header strokes heavier than the body, similar to the body, lighter than the body, or unclear?
2. Absolute body weight: do the body letters themselves look bold/heavy, regular/thin, or unclear?

Important visual traps:
- If both header and body appear bold/heavy, report body_bold=true and header_body_relationship="same".
- If the image is too small, blurry, angled, compressed, glared, or low-contrast to compare stroke weights confidently, use null/unclear rather than guessing.
- Use high confidence only when the relevant letters are clearly readable and the stroke-weight difference is visually obvious.

Return JSON only:
{
  "warning_present": true | false | null,
  "header_text_seen": string | null,
  "header_body_relationship": "header_heavier" | "same" | "body_heavier" | "unclear" | null,
  "header_weight": "thin" | "regular" | "semibold_heavy" | "unclear",
  "body_weight": "thin" | "regular" | "semibold_heavy" | "unclear",
  "header_bold": true | false | null,
  "body_bold": true | false | null,
  "header_bold_confidence": "high" | "medium" | "low",
  "body_bold_confidence": "high" | "medium" | "low",
  "legibility": "good" | "limited" | "poor",
  "short_basis": string,
  "image_quality_notes": string | null
}"""


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def main():
    with open(os.path.join(BS, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    images = sorted(man.items(), key=lambda kv: kv[0])
    print(f"prompt=S(traps)  models={MODELS}  images={len(images)}  reps=1\n")
    report = {}
    for model in MODELS:
        print(f"=== {model} + S ===")
        by_cls = defaultdict(lambda: {"n": 0, "correct": 0, "false_pass": 0, "hi_false_pass": 0, "review": 0})
        lat = []
        for fname, gt in images:
            var = gt["variant"]
            fields, dt, retries, err = B._call(model, PROMPT_S, [os.path.join(BS, fname)])
            if not fields:
                print(f"  {fname:26s} ERROR {str(err)[:60]}")
                continue
            lat.append(dt)
            hb, bb = B._eff_header_bold(fields), B._eff_body_bold(fields)
            c = by_cls[var]
            c["n"] += 1
            corr, fp, rev = B._correct(var, fields), B._fp(var, fields), B._review(var, fields)
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
            print(f"  {fname:26s} [{var:14s}] {dt:5.2f}s hb={hb}/{(fields.get('header_bold_confidence') or '-')[:1]} "
                  f"bb={bb}/{(fields.get('body_bold_confidence') or '-')[:1]} rel={fields.get('header_body_relationship')} -> {tag}")
        prim_n = sum(by_cls[v]["n"] for v in PRIMARY)
        prim_correct = sum(by_cls[v]["correct"] for v in PRIMARY)
        hifp = sum(c["hi_false_pass"] for c in by_cls.values())
        fp = sum(c["false_pass"] for c in by_cls.values())
        report[model] = {
            "accuracy_primary": f"{prim_correct}/{prim_n}",
            "accuracy_pct": round(100 * prim_correct / prim_n, 1) if prim_n else None,
            "false_pass": fp, "high_conf_false_pass": hifp,
            "by_class": {v: dict(by_cls[v]) for v in by_cls},
            "lat_avg": round(sum(lat) / len(lat), 2) if lat else None, "lat_p50": _pct(lat, 50),
            "lat_max": max(lat) if lat else None, "over_5s": sum(1 for x in lat if x > 5),
        }
        r = report[model]
        print(f"  -> accuracy(primary 15) {r['accuracy_primary']} ({r['accuracy_pct']}%)  "
              f"false-pass {fp} (high-conf {hifp})  |  time avg {r['lat_avg']}s p50 {r['lat_p50']}s "
              f"max {r['lat_max']}s\n")
    _write(report)


def _write(report):
    L = ["", "=" * 96, "PROMPT S ('traps') on bold_safety (1x): TIME + ACCURACY -- gpt-5.4-mini & gpt-4.1",
         "=" * 96,
         "accuracy over 15 PRIMARY imgs. high-conf FALSE-PASS = a violation confidently read as "
         "compliant. boldbody catch rate is the headline (this prompt targets body-bold).", ""]
    L.append(f"{'model + S':16s} {'accuracy':14s} {'false-pass':11s} {'HI-falsepass':13s} "
             f"{'avg':6s} {'p50':6s} {'max':6s}")
    L.append("-" * 80)
    for m, r in report.items():
        acc = f"{r['accuracy_primary']} ({r['accuracy_pct']}%)"
        L.append(f"{m:16s} {acc:14s} {str(r['false_pass']):11s} {str(r['high_conf_false_pass']):13s} "
                 f"{str(r['lat_avg']):6s} {str(r['lat_p50']):6s} {str(r['lat_max']):6s}")
    L.append("")
    for m, r in report.items():
        L.append(f"--- {m} + S  (accuracy {r['accuracy_primary']} = {r['accuracy_pct']}%, "
                 f"high-conf false-pass {r['high_conf_false_pass']}) ---")
        for v in ("bold_compliant", "boldbody", "notbold", "titlecase"):
            c = r["by_class"].get(v)
            if not c:
                continue
            if v == "bold_compliant":
                extra = f"correct {c['correct']}/{c['n']}"
            elif v in ("boldbody", "notbold"):
                extra = f"caught {c['n']-c['false_pass']}/{c['n']}, FALSE-PASS {c['false_pass']} (high-conf {c['hi_false_pass']})"
            else:
                extra = f"observational ({c['n']})"
            L.append(f"     {v:16s} {extra}   review {c['review']}")
        L.append("")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"prompt_S_test_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written {os.path.relpath(base, ROOT)}.txt / .json")


if __name__ == "__main__":
    main()
