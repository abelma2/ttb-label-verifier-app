"""The body-bold decider: gpt-4o vs the gpt-5.x readers on a GENUINELY BOLD BODY.

The 01-04 adversarial set and the A/D/E cross-model table only vary the HEADER, so they cannot
separate models on 27 CFR 16.22's SECOND rule -- the body/remainder must NOT be bold. That is
exactly where gpt-4o and the gpt-5.x reasoning models disagreed (the rum body). bold_safety/
has font-controlled BODY-bold ground truth (boldbody__* = header bold AND body bold, a violation).

This runs the PRODUCTION pipeline (extract_fields full prompt -> verification._check_warning under
the live header_body_gate) on those fixtures, per model, and asks the decisive question:
  - boldbody__*      (body IS bold -> a violation)  -> a model that reads body-bold FAILs it;
                                                       a rubber-stamp (body_bold=False) PASSES it.
  - bold_compliant__* (bold header, non-bold body)  -> should PASS.
  - notbold__*        (header not bold)             -> should FAIL.

Sequential (no parallelism) to avoid the 30k-TPM throttle on gpt-4o / gpt-4.1.
Usage:  python scripts/benchmarks/bodybold_decider.py [runs]
Writes output/bodybold_decider_<ts>.{txt,json}.
"""
import json
import os
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

import model_image_experiment as M             # _extract_with_model + secrets key load
from verification import _check_warning, PASS, REVIEW, FAIL

BS = os.path.join(ROOT, "bold_safety")
MODELS = ["gpt-5.4-mini", "gpt-5.5", "gpt-4o", "gpt-4.1"]
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

# (file, variant, expected verdict). Focus on the DECISIVE body-bold cases + a compliant control
# + a header-not-bold control; include a couple distortions of boldbody for robustness.
CASES = [
    ("boldbody__clean.png",       "boldbody",       FAIL),   # body bold -> must FAIL (the decider)
    ("boldbody__lowres.png",      "boldbody",       FAIL),
    ("boldbody__jpeg.jpg",        "boldbody",       FAIL),
    ("bold_compliant__clean.png", "bold_compliant", PASS),   # header bold, body not -> PASS control
    ("notbold__clean.png",        "notbold",        FAIL),   # header not bold -> FAIL control
]


def _mt(p):
    return "image/jpeg" if p.lower().endswith((".jpg", ".jpeg")) else "image/png"


def main():
    print(f"models={MODELS}  runs={RUNS}  cases={[c[0] for c in CASES]}\n")
    report = {}
    for model in MODELS:
        print(f"=== {model} ===")
        per_case = {}
        for fname, variant, expected in CASES:
            path = os.path.join(BS, fname)
            imgs = [(open(path, "rb").read(), _mt(path))]
            verdicts, bodybold_obs, headerbold_obs, times = Counter(), Counter(), Counter(), []
            for rep in range(1, RUNS + 1):
                try:
                    t0 = time.perf_counter()
                    ex = M._extract_with_model(model, imgs)
                    dt = round(time.perf_counter() - t0, 2)
                    gw = ex.get("government_warning") or {}
                    r = _check_warning(gw)
                    verdicts[r.status] += 1
                    times.append(dt)
                    bb = gw.get("body_bold")
                    bodybold_obs[f"{bb}/{gw.get('body_bold_confidence')}"] += 1
                    headerbold_obs[f"{gw.get('header_bold')}/{gw.get('header_bold_confidence')}"] += 1
                    print(f"  {variant:14s} rep{rep}: {dt:5.2f}s  {r.status:12s} (cause {r.cause}) "
                          f"body_bold={bb}/{gw.get('body_bold_confidence')}")
                except Exception as exc:
                    verdicts["ERR"] += 1
                    print(f"  {variant:14s} rep{rep}: ERROR {str(exc)[:90]}")
            per_case[fname] = {"variant": variant, "expected": expected,
                               "verdicts": dict(verdicts), "body_bold_obs": dict(bodybold_obs),
                               "header_bold_obs": dict(headerbold_obs),
                               "time_mean": round(sum(times) / len(times), 2) if times else None}
        report[model] = per_case
        # per-model boldbody catch-rate
        bb_cases = [c for c in per_case.values() if c["variant"] == "boldbody"]
        bb_fail = sum(c["verdicts"].get(FAIL, 0) for c in bb_cases)
        bb_n = sum(sum(c["verdicts"].values()) for c in bb_cases)
        comp = per_case.get("bold_compliant__clean.png", {}).get("verdicts", {})
        print(f"  -> BODY-BOLD caught (boldbody FAIL): {bb_fail}/{bb_n}   "
              f"bold_compliant: {comp}\n")
    _write(report)


def _write(report):
    L = ["", "=" * 96, "BODY-BOLD DECIDER  (production pipeline on font-controlled body-bold ground truth)",
         "=" * 96,
         "boldbody__* = header bold AND body bold (a 27 CFR 16.22 violation) -> MUST FAIL.",
         "A model that READS body-bold FAILs it (body_bold=True); a rubber-stamp PASSES it (body_bold=False).",
         "bold_compliant -> should PASS ; notbold -> should FAIL.", ""]
    # headline: body-bold catch-rate + compliant pass-rate + the raw body_bold observation
    L.append(f"{'model':14s} {'boldbody FAIL (caught)':22s} {'compliant PASS':16s} {'notbold FAIL':14s} "
             f"{'body_bold read on boldbody':28s}")
    L.append("-" * 96)
    for model, pc in report.items():
        bb = [c for c in pc.values() if c["variant"] == "boldbody"]
        bb_fail = sum(c["verdicts"].get(FAIL, 0) for c in bb)
        bb_n = sum(sum(c["verdicts"].values()) for c in bb)
        comp = pc.get("bold_compliant__clean.png", {}).get("verdicts", {})
        comp_pass = comp.get(PASS, 0); comp_n = sum(comp.values())
        nb = pc.get("notbold__clean.png", {}).get("verdicts", {})
        nb_fail = nb.get(FAIL, 0); nb_n = sum(nb.values())
        bb_obs = Counter()
        for c in bb:
            for k, v in c["body_bold_obs"].items():
                bb_obs[k] += v
        L.append(f"{model:14s} {f'{bb_fail}/{bb_n}':22s} {f'{comp_pass}/{comp_n}':16s} "
                 f"{f'{nb_fail}/{nb_n}':14s} {dict(bb_obs)}")
    L.append("")
    L.append("  body_bold read key: 'True/high' = correctly saw the bold body; 'False/high' = rubber-stamped it.")
    L.append("")
    for model, pc in report.items():
        L.append(f"--- {model} ---")
        for fname, c in pc.items():
            L.append(f"   {fname:26s} exp={c['expected']:6s} verdicts={c['verdicts']}  "
                     f"body_bold={c['body_bold_obs']}  ({c['time_mean']}s)")
        L.append("")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt = os.path.join(OUT_DIR, f"bodybold_decider_{stamp}.txt")
    js = os.path.join(OUT_DIR, f"bodybold_decider_{stamp}.json")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written to:\n  {os.path.relpath(txt, ROOT)}\n  {os.path.relpath(js, ROOT)}")


if __name__ == "__main__":
    main()
