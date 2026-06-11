"""Lightweight VLM warning-bold WITNESS benchmark (BENCHMARK ONLY) — model comparison.

Question: can a faster/lighter FULL-IMAGE warning-only variant-A call improve bold evidence
within the ~5s bottle budget, vs the current OCR-crop + variant-A path?

Models (exact, per the experiment spec): gpt-5.4-mini · gpt-5.4-nano · gpt-5.4
Same warning-only variant-A prompt and the same strict structured-output schema for every
model (reused verbatim from warning_bold_prompt_variants.py). detail="high".

TIMING IS THE HEADLINE. Calls run SEQUENTIALLY (zero client-side contention) and every
model/image/repeat records:
  call_seconds   — the successful API call duration only
  total_seconds  — full per-call elapsed: file load + base64 encode + request + parse,
                   INCLUDING any retry waits (the honest end-to-end figure)
  retries        — extra attempts beyond the first (transient API errors)
  error          — recorded if all attempts failed
Per model: avg/median/p50/p90/p95/max, fastest/slowest image, calls over 5s, and the
estimated 3-witness PARALLEL wall-clock per (image, repeat) = max over models (parallel
witnesses cost the max, not the sum). Token usage is summed per model as the cost proxy
(no $ pricing assumed).

Scoring: the bold_safety ground truth (filename-encoded): bold_compliant -> header bold +
body not bold; boldbody -> body_bold=true; notbold -> header_bold=false; titlecase ->
caps case (bold observations recorded, not the primary bold score).

NO production change, NO default enablement, NO model voting into PASS (disagreement
between witnesses would route to needs_review — that policy lives in the merge layer, not
here; this script only measures evidence quality and latency).

Run (real model calls — costs money):
  python scripts/benchmarks/warning_bold_model_witness_benchmark.py            # 3x20x3 = 180 calls
  python scripts/benchmarks/warning_bold_model_witness_benchmark.py --repeats 1 --models gpt-5.4-mini
Outputs: output/warning_bold_model_witness_benchmark_<ts>.json / .txt
"""
import json
import os
import statistics
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

import warning_bold_prompt_variants as wb           # _COMMON + PROMPT_A + schema (+ helpers), reused verbatim
from extraction import _get_client, _model_params, _create_with_fallbacks

BS = os.path.join(ROOT, "bold_safety")
MODELS = ("gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4")
BUDGET_S = 5.0

# Current OCR-crop + variant-A reference (crop_variant_a_benchmark_20260609_194741, 3x,
# gpt-5.4-mini): the incumbent this experiment compares against.
CROP_REF = {
    "boldbody_fp": "0/15 (hi-conf 0)", "notbold_fp": "6/18 (all medium/low)",
    "compliant_ok": "18/18", "latency": "total p50=2.38 p90=4.46 p95=6.91 max=33.7s "
    "(ocr 0.24 + crop 0.06 + model 2.51 avg)",
}


def _arg(args, flag, default):
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else default


def _call(model, paths):
    """One full witness call (one or more images) with split timing. total_seconds includes
    retry waits."""
    t0 = time.perf_counter()
    content = [{"type": "text", "text": wb._COMMON + wb.PROMPT_A}]
    for p in paths:
        b = open(p, "rb").read()
        content.append(wb._block(b, wb._media_type(p), "high"))
    attempts, last, out, call_s, usage = 0, None, None, None, None
    while attempts < 3:
        attempts += 1
        params = _model_params(model, response_format=wb._rf())
        tc = time.perf_counter()
        try:
            resp = _create_with_fallbacks(_get_client(), content, params)
            call_s = time.perf_counter() - tc
            out = json.loads(resp.choices[0].message.content)
            u = getattr(resp, "usage", None)
            if u is not None:
                usage = {"prompt_tokens": getattr(u, "prompt_tokens", None),
                         "completion_tokens": getattr(u, "completion_tokens", None)}
            break
        except Exception as exc:
            last = str(exc)[:160]
            time.sleep(1.5 * attempts)
    return {"output": out, "call_seconds": round(call_s, 2) if call_s is not None else None,
            "total_seconds": round(time.perf_counter() - t0, 2), "retries": attempts - 1,
            "error": None if out is not None else last, "usage": usage}


def _hb(o): return o.get("header_bold") if o else None
def _bb(o): return o.get("body_bold") if o else None
def _hc(o): return o.get("header_bold_confidence") if o else None
def _bc(o): return o.get("body_bold_confidence") if o else None


def _pcts(vals):
    v = sorted(vals)
    p = lambda q: v[min(len(v) - 1, int(len(v) * q))]
    return {"avg": round(statistics.mean(v), 2), "med": round(statistics.median(v), 2),
            "p50": p(.5), "p90": p(.9), "p95": p(.95), "max": v[-1]}


def _gate_proxy(o):
    """The production header_body_gate's BOLD outcome for one witness observation (wording/caps
    not judged here — the witness schema records header_text_seen only). FAIL = a confident
    violation read on a (presumed-compliant) baseline = the false-fail direction."""
    if o is None:
        return "ERR"
    if not o.get("warning_present"):
        return "FAIL"   # warning not found on a label that has one
    if (_hb(o) is False and _hc(o) == "high") or (_bb(o) is True and _bc(o) == "high"):
        return "FAIL"
    if _hb(o) is True and _hc(o) == "high" and _bb(o) is False and _bc(o) == "high":
        return "PASS"
    return "REVIEW"


def main():
    args = sys.argv[1:]
    repeats = int(_arg(args, "--repeats", "3"))
    models = [m for m in _arg(args, "--models", ",".join(MODELS)).split(",") if m]

    if not wb._load_key() and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: no OpenAI key.")
    baseline = "--baseline" in args
    if baseline:
        # production-shaped input: each clean baseline product as a front+back PAIR (the
        # full-size two-image read — also answers how each model's latency scales with real
        # input size vs the small bold_safety renders). Baselines are presumed compliant:
        # the metric is the false-fail / review rate under the gate proxy.
        BL = os.path.join(ROOT, "test_labels", "baseline_labels")
        cases = [("baseline_%d" % n,
                  [os.path.join(BL, "baseline_%d_Front.png" % n),
                   os.path.join(BL, "baseline_%d_Other.png" % n)], "baseline_pair")
                 for n in (1, 2, 3)
                 if os.path.exists(os.path.join(BL, "baseline_%d_Front.png" % n))]
    else:
        images = sorted(f for f in os.listdir(BS) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        cases = [(fn, [os.path.join(BS, fn)], wb._expected_class(fn)) for fn in images]
    case_labels = [c[0] for c in cases]
    total = len(models) * len(cases) * repeats
    print("mode=%s  models=%s  cases=%d  repeats=%d  SEQUENTIAL calls=%d  budget=%.0fs\n"
          % ("baseline-pairs" if baseline else "bold_safety", ",".join(models), len(cases),
             repeats, total, BUDGET_S), flush=True)

    rows, done = [], 0
    for model in models:
        for rep in range(1, repeats + 1):
            for label, paths, cls in cases:
                r = _call(model, paths)
                r.update({"model": model, "image": label, "repeat": rep, "cls": cls})
                rows.append(r)
                done += 1
                if done % 20 == 0 or done == total:
                    print("  %d/%d done (%s)" % (done, total, model), flush=True)

    # ---- per-model scoring + latency ----
    L = ["", "=" * 106,
         "WARNING-BOLD MODEL WITNESS BENCHMARK — full-image variant-A, per-model (sequential timing)",
         "=" * 106,
         "mode=%s  models=%s  cases=%d  repeats=%d  detail=high  prompt=variant A (shared)"
         % ("baseline-pairs (front+back, presumed compliant)" if baseline else "bold_safety",
            ",".join(models), len(cases), repeats), ""]
    summary = {}
    n_comp = sum(1 for _, _, c in cases if c == "bold_compliant") * repeats
    n_bb = sum(1 for _, _, c in cases if c == "boldbody") * repeats
    n_nb = sum(1 for _, _, c in cases if c == "notbold") * repeats

    for model in models:
        recs = [r for r in rows if r["model"] == model]
        valid = [r for r in recs if r["output"] is not None]
        errors = [r for r in recs if r["output"] is None]
        retries = sum(r["retries"] for r in recs)

        comp_ok = bb_fp = nb_fp = hi_fp = uncertain = med_corr = 0
        for r in valid:
            o, cls = r["output"], r["cls"]
            if cls == "bold_compliant":
                if _hb(o) is True and _bb(o) is False:
                    comp_ok += 1
                    if _hc(o) == "medium" or _bc(o) == "medium":
                        med_corr += 1
                if _hb(o) is None or _bb(o) is None or _hc(o) == "low" or _bc(o) == "low":
                    uncertain += 1
            elif cls == "boldbody":
                if _bb(o) is False:
                    bb_fp += 1
                    if _bc(o) == "high":
                        hi_fp += 1
                elif _bb(o) is True and _bc(o) == "medium":
                    med_corr += 1
                if _bb(o) is None or _bc(o) == "low":
                    uncertain += 1
            elif cls == "notbold":
                if _hb(o) is True:
                    nb_fp += 1
                    if _hc(o) == "high":
                        hi_fp += 1
                elif _hb(o) is False and _hc(o) == "medium":
                    med_corr += 1
                if _hb(o) is None or _hc(o) == "low":
                    uncertain += 1

        val_stable = val_changed = conf_changed = groups = 0
        for fn in case_labels:
            grp = [r for r in recs if r["image"] == fn and r["output"] is not None]
            if len(grp) < 2:
                continue
            groups += 1
            vals = {(_hb(r["output"]), _bb(r["output"])) for r in grp}
            confs = {(_hc(r["output"]), _bc(r["output"])) for r in grp}
            if len(vals) == 1 and len(grp) == repeats:
                val_stable += 1
            if len(vals) > 1:
                val_changed += 1
            if len(confs) > 1:
                conf_changed += 1

        tot = [r["total_seconds"] for r in recs]
        calls = [r["call_seconds"] for r in recs if r["call_seconds"] is not None]
        lp = _pcts(tot)
        by_img = {fn: statistics.mean([r["total_seconds"] for r in recs if r["image"] == fn])
                  for fn in case_labels}
        fastest = min(by_img, key=by_img.get)
        slowest = max(by_img, key=by_img.get)
        over = [r for r in recs if r["total_seconds"] > BUDGET_S]
        ptok = sum((r["usage"] or {}).get("prompt_tokens") or 0 for r in recs)
        ctok = sum((r["usage"] or {}).get("completion_tokens") or 0 for r in recs)

        summary[model] = {"comp_ok": comp_ok, "n_comp": n_comp, "bb_fp": bb_fp, "n_bb": n_bb,
                          "nb_fp": nb_fp, "n_nb": n_nb, "tot_fp": bb_fp + nb_fp, "hi_fp": hi_fp,
                          "uncertain": uncertain, "med_corr": med_corr,
                          "val_stable": val_stable, "groups": groups, "val_changed": val_changed,
                          "conf_changed": conf_changed, "errors": len(errors), "retries": retries,
                          "latency": lp, "call_avg": round(statistics.mean(calls), 2) if calls else None,
                          "fastest_image": [fastest, round(by_img[fastest], 2)],
                          "slowest_image": [slowest, round(by_img[slowest], 2)],
                          "over_budget": [len(over), round(100.0 * len(over) / len(recs), 1)],
                          "tokens": {"prompt": ptok, "completion": ctok}}

        L.append("--- %s ---" % model)
        L.append("  LATENCY total_s: avg=%.2f med=%.2f p50=%.2f p90=%.2f p95=%.2f max=%.2f "
                 "(call-only avg %.2f)" % (lp["avg"], lp["med"], lp["p50"], lp["p90"], lp["p95"],
                                           lp["max"], summary[model]["call_avg"] or 0))
        L.append("  over %.0fs: %d/%d (%.1f%%)  fastest=%s %.2fs  slowest=%s %.2fs"
                 % (BUDGET_S, len(over), len(recs), summary[model]["over_budget"][1],
                    fastest, by_img[fastest], slowest, by_img[slowest]))
        if baseline:
            gates = Counter(_gate_proxy(r["output"]) for r in recs)
            present_ok = sum(1 for r in valid if r["output"].get("warning_present") is True)
            ffs = [r for r in valid if _gate_proxy(r["output"]) == "FAIL"]
            summary[model]["gate"] = dict(gates)
            summary[model]["present_ok"] = [present_ok, len(valid)]
            L.append("  BASELINE GATE (compliant expected): PASS=%d REVIEW=%d FAIL=%d ERR=%d  "
                     "| warning found %d/%d"
                     % (gates.get("PASS", 0), gates.get("REVIEW", 0), gates.get("FAIL", 0),
                        gates.get("ERR", 0), present_ok, len(valid)))
            for r in ffs:
                o = r["output"]
                L.append("    FALSE-FAIL %s r%d: present=%s hb=%s[%s] bb=%s[%s] basis=%r"
                         % (r["image"], r["repeat"], o.get("warning_present"), _hb(o), _hc(o),
                            _bb(o), _bc(o), (o.get("short_basis") or "")[:60]))
        else:
            L.append("  SAFETY: compliant %d/%d  boldbodyFP %d/%d  notboldFP %d/%d  totFP %d  "
                     "HI-CONF FP %d  uncert %d  medCorr %d"
                     % (comp_ok, n_comp, bb_fp, n_bb, nb_fp, n_nb, bb_fp + nb_fp, hi_fp,
                        uncertain, med_corr))
        L.append("  STABILITY: value-stable %d/%d  valChg %d  confChg %d  errors %d  retries %d"
                 % (val_stable, groups, val_changed, conf_changed, len(errors), retries))
        L.append("  tokens: prompt %d  completion %d  (no $ pricing assumed)" % (ptok, ctok))
        L.append("")

    # ---- per-case gate verdicts by repeat (baseline mode) ----
    if baseline:
        L.append("PER-CASE gate verdicts by repeat (P/R/F/E):")
        for fn in case_labels:
            row = "  %-12s" % fn
            for model in models:
                seq = sorted((r for r in rows if r["model"] == model and r["image"] == fn),
                             key=lambda r: r["repeat"])
                row += "  %s=%s" % (model, "".join(_gate_proxy(r["output"])[0] for r in seq))
            L.append(row)
        L.append("")

    # ---- 3-witness parallel wall-clock estimate: max over models per (image, repeat) ----
    if len(models) > 1:
        walls = []
        for fn in case_labels:
            for rep in range(1, repeats + 1):
                ts = [r["total_seconds"] for r in rows if r["image"] == fn and r["repeat"] == rep]
                if len(ts) == len(models):
                    walls.append(max(ts))
        if walls:
            wp = _pcts(walls)
            over_w = sum(1 for w in walls if w > BUDGET_S)
            L.append("ESTIMATED %d-WITNESS PARALLEL WALL-CLOCK (max of the parallel calls, per image/repeat):"
                     % len(models))
            L.append("  p50=%.2f p90=%.2f p95=%.2f max=%.2f  over %.0fs: %d/%d (%.1f%%)"
                     % (wp["p50"], wp["p90"], wp["p95"], wp["max"], BUDGET_S, over_w, len(walls),
                        100.0 * over_w / len(walls)))
            summary["_parallel_wall"] = {**wp, "over_budget": [over_w, len(walls)]}
            L.append("")

    L.append("REFERENCE — current OCR crop + variant-A path (crop_variant_a_benchmark, 3x, gpt-5.4-mini):")
    for k, v in CROP_REF.items():
        L.append("  %s: %s" % (k, v))

    report = "\n".join(L)
    print(report)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, "warning_bold_model_witness_benchmark_%s%s"
                        % ("baselines_" if baseline else "", stamp))
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump({"mode": "baseline-pairs" if baseline else "bold_safety",
                   "models": models, "repeats": repeats, "budget_s": BUDGET_S,
                   "prompt": wb._COMMON + wb.PROMPT_A, "crop_reference": CROP_REF,
                   "summary": summary, "rows": rows}, fh, indent=2, ensure_ascii=False)
    print("\nWritten to: %s.txt / .json" % os.path.relpath(base, ROOT))


if __name__ == "__main__":
    main()
