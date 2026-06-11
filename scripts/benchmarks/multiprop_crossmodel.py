"""Cross-model A/B of the MULTI-PROPERTY bold prompt (variant E) vs the existing prompt variants,
on the font-controlled adversarial set where bold is KNOWN ground truth.

The question: the gpt-4o / gpt-4.1 generation rubber-stamps the warning as bold-compliant even on
03_notbold (a header that is genuinely NOT bold). Does asking for several ORTHOGONAL typographic
properties (variant E: bold/italic/underline x header/body, binary) make those models actually
LOOK -- i.e. correctly FAIL 03_notbold -- or do they still pass it (confirming the blindness is
perceptual, not a priming artifact the prompt can fix)? And does E help / hurt the bold-reading
gpt-5.x models?

It reuses bold_variant_benchmark's variant prompts/schemas/gates (A baseline, D candidate, E new)
and the real wording/caps/Surgeon-General judging, scoring each (model x variant) on the
adversarial ground truth:
   01_compliant -> PASS   |   03_notbold -> FAIL (the bold trap)   |   02/04 -> FAIL

Verdicts only (no latency), so the run is PARALLELIZED. gpt-4o(-mini)/gpt-4.1 share low TPM, so a
modest pool + per-call error capture handles the occasional 429.

Usage:  python scripts/benchmarks/multiprop_crossmodel.py [runs]
Writes output/multiprop_crossmodel_<ts>.{txt,json}.
"""
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT
OUT_DIR = os.path.join(ROOT, "output")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import bold_variant_benchmark as B            # variant prompts/schemas/gates + _run + cases + key load
from verification import PASS, REVIEW, FAIL

# the user's 7: two gpt-5.4 reasoning models (default reasoning_effort=low) + five gpt-4.x/4o non-reasoning
MODELS = ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini"]
VARIANTS = ["a", "d", "e", "f", "g", "h", "i"]
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
CASES = B.ADV_CASES                            # font-controlled ground truth (bold is known)
_VLABEL = {"a": "A baseline", "d": "D fmt-quality", "e": "E multi-prop", "f": "F relative-scale",
           "g": "G describe-first", "h": "H weight-gap", "i": "I self-consist"}


def _job(model, variant, cid, paths, run):
    try:
        rec = B._run(variant, paths, grid=(3, 3), detail="high",
                     models={"a": model, "rich": model})
        return model, variant, cid, rec["status"], rec.get("fields", {}), None
    except Exception as exc:
        return model, variant, cid, "ERR", {}, str(exc)[:140]


def main():
    B._load_key()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: no OpenAI key (env OPENAI_API_KEY or .streamlit/secrets.toml).")
    jobs = [(m, v, cid, paths, r)
            for m in MODELS for v in VARIANTS
            for cid, paths, _exp in CASES for r in range(RUNS)]
    print(f"{len(MODELS)} models x {len(VARIANTS)} variants x {len(CASES)} adversarial cases "
          f"x {RUNS} runs = {len(jobs)} calls (parallel)\n")

    # results[(model,variant)][cid] = list of statuses ; ev[(model,variant)][cid] = sample fields
    results, ev, errors = {}, {}, []
    done, total = 0, len(jobs)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_job, m, v, cid, paths, r) for (m, v, cid, paths, r) in jobs]
        for fut in as_completed(futs):
            model, variant, cid, status, fields, err = fut.result()
            done += 1
            key = (model, variant)
            results.setdefault(key, {}).setdefault(cid, []).append(status)
            if cid in ("03_notbold", "01_compliant") and fields:
                ev.setdefault(key, {}).setdefault(cid, fields)   # keep a sample of the bold evidence
            if err:
                errors.append((model, variant, cid, err))
            if done % 15 == 0 or done == total:
                print(f"  [{done}/{total}] ...", flush=True)
    _write(results, ev, errors)


def _rate(results, key, cid, want):
    rs = results.get(key, {}).get(cid, [])
    return sum(1 for s in rs if s == want), len(rs)


def _write(results, ev, errors):
    L = ["", "=" * 100,
         "MULTI-PROPERTY (E) vs BASELINE (A) / FMT-QUALITY (D)  -- cross-model, adversarial ground truth",
         "=" * 100,
         f"runs={RUNS}  detail=high   the BOLD TRAP is 03_notbold (header genuinely NOT bold -> must FAIL).",
         "A model that reads bold catches it (03 FAIL); a bold-blind model rubber-stamps it (03 PASS).", ""]
    # headline: the bold trap (03_notbold -> FAIL) by model x variant
    L.append("BOLD-TRAP: 03_notbold FAIL-rate (higher = actually reads bold)  +  01_compliant PASS-rate")
    L.append(f"{'model':14s} | " + " | ".join(f"{_VLABEL[v]:^26s}" for v in VARIANTS))
    L.append(f"{'':14s} | " + " | ".join(f"{'03 FAIL':>12s} {'01 PASS':>12s} " for v in VARIANTS))
    L.append("-" * 100)
    for m in MODELS:
        cells = []
        for v in VARIANTS:
            f3, n3 = _rate(results, (m, v), "03_notbold", FAIL)
            p1, n1 = _rate(results, (m, v), "01_compliant", PASS)
            cells.append(f"{f'{f3}/{n3}':>12s} {f'{p1}/{n1}':>12s} ")
        L.append(f"{m:14s} | " + " | ".join(cells))
    L.append("")

    # full per (model,variant) scorecard across all 4 adversarial cases
    L.append("FULL SCORECARD (want: 01 PASS, 02/03/04 FAIL; false-pass = any should-FAIL that PASSed)")
    L.append(f"{'model':14s} {'variant':14s} {'01 PASS':8s} {'03 FAIL':8s} {'02 FAIL':8s} "
             f"{'04 FAIL':8s} {'false-pass':11s}")
    L.append("-" * 86)
    for m in MODELS:
        for v in VARIANTS:
            c1 = _rate(results, (m, v), "01_compliant", PASS)
            c3 = _rate(results, (m, v), "03_notbold", FAIL)
            c2 = _rate(results, (m, v), "02_titlecase", FAIL)
            c4 = _rate(results, (m, v), "04_reworded", FAIL)
            fp = sum(1 for cid in ("02_titlecase", "03_notbold", "04_reworded")
                     for s in results.get((m, v), {}).get(cid, []) if s == PASS)
            L.append(f"{m:14s} {_VLABEL[v]:14s} {f'{c1[0]}/{c1[1]}':8s} {f'{c3[0]}/{c3[1]}':8s} "
                     f"{f'{c2[0]}/{c2[1]}':8s} {f'{c4[0]}/{c4[1]}':8s} {str(fp):11s}")
        L.append("")

    # sample bold evidence on the trap, to SEE whether E changed what the model reports
    L.append("SAMPLE bold evidence on 03_notbold (header is NOT bold -- a reader should report header_bold False/0):")
    for m in MODELS:
        for v in VARIANTS:
            f = ev.get((m, v), {}).get("03_notbold")
            if not f:
                continue
            keys = ["header_bold", "body_bold", "header_bold_confidence", "formatting_quality",
                    "formatting_legibility", "header_vs_body", "body_vs_surround", "scale_confidence",
                    "header_vs_body_weight", "comparison_confidence", "header_weight_class",
                    "body_weight_class", "weight_gap_steps", "weight_legibility"]
            shown = {k: f.get(k) for k in keys if k in f}
            if "bold_trials" in f and isinstance(f["bold_trials"], list):   # variant I: summarize the 5 reads
                hb = [t.get("header_bold") for t in f["bold_trials"] if isinstance(t, dict)]
                shown["trials_header_bold"] = hb
            L.append(f"  {m:14s} {_VLABEL[v]:14s} {shown}")
    L.append("")
    if errors:
        L.append(f"({len(errors)} call errors, e.g. {errors[0][0]}/{errors[0][1]}: {errors[0][3]})")

    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt = os.path.join(OUT_DIR, f"multiprop_crossmodel_{stamp}.txt")
    js = os.path.join(OUT_DIR, f"multiprop_crossmodel_{stamp}.json")
    flat = {f"{m}|{v}": {cid: results.get((m, v), {}).get(cid, []) for cid, _, _ in CASES}
            for m in MODELS for v in VARIANTS}
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump({"models": MODELS, "variants": VARIANTS, "runs": RUNS, "results": flat,
                   "errors": errors}, fh, indent=2, ensure_ascii=False)
    print(f"Written to:\n  {os.path.relpath(txt, ROOT)}\n  {os.path.relpath(js, ROOT)}")


if __name__ == "__main__":
    main()
