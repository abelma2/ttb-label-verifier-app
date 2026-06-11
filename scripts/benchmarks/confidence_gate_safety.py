"""Benchmark-only: does the CURRENT production path auto-PASS any bold violation?

Runs the real production extraction (`extraction.extract_fields`, the production _PROMPT + model)
and the real verifier (`verification.verify_label_only`) on the controlled bold_safety set --
exactly as-is, nothing modified. It pins `WARNING_BOLD_POLICY=confidence_gate` by default (the gate
this benchmark is named for; the production default is now header_body_gate) -- set the env var to
measure another gate, and the report title/banner reflect whichever policy actually ran. For each
image it records the government_warning field's verdict (pass / needs_review / fail).

The safety question: on a NOT-bold header or an ALL-bold-body warning, does the warning verdict
come back PASS? That would be an automated false-pass of a violation. (Note: confidence_gate checks
the header's header_bold only -- it does NOT check that the body is non-bold -- so an all-bold-body
warning whose header is bold is expected to PASS, which is exactly the gap this measures.)

Diagnostic only -- nothing is wired into production. 3 runs/image (modest, to smooth the documented
run-to-run variance; NOT the deferred 540-call stability pass).

Run:  python scripts/benchmarks/confidence_gate_safety.py [runs]
Outputs: artifacts/confidence_gate_safety_results.md / .json
"""
import json
import os
import sys

import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from smoke_test import _load_key, _media_type


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
    man = os.path.join(ROOT, "bold_safety", "manifest.json")
    if not os.path.exists(man):
        sys.exit("No bold_safety set -- run: python scripts/benchmarks/generate_bold_safety.py")
    manifest = json.load(open(man, encoding="utf-8"))
    if not _load_key():
        sys.exit("ERROR: no OpenAI key (OPENAI_API_KEY env or .streamlit/secrets.toml).")

    # This benchmark is named for (and its prose describes) confidence_gate; pin it so the script
    # measures that gate even though the production default is now header_body_gate. Must be set
    # BEFORE the first import that pulls in config (extraction imports it). Override via the env var.
    os.environ.setdefault("WARNING_BOLD_POLICY", "confidence_gate")
    # Production path, used exactly as-is (only called, never modified).
    from extraction import extract_fields, EXTRACTION_MODEL
    from verification import verify_label_only, PASS, REVIEW, FAIL
    from config import WARNING_BOLD_POLICY

    per_image = []
    for fn in sorted(manifest):
        meta = manifest[fn]
        path = os.path.join(ROOT, "bold_safety", fn)
        data = open(path, "rb").read()
        mime = _media_type(path)
        runs_out = []
        for _ in range(runs):
            try:
                extracted = extract_fields([(data, mime)])
                result = verify_label_only(extracted)
                gw = next((f for f in result["fields"] if f.field == "government_warning"), None)
                raw = extracted.get("government_warning") or {}
                runs_out.append({"warn_status": gw.status if gw else "ERR",
                                 "warn_reason": (gw.reason if gw else "")[:90],
                                 "header_bold": raw.get("header_bold"),
                                 "header_bold_confidence": raw.get("header_bold_confidence")})
            except Exception as e:
                runs_out.append({"warn_status": "ERR", "warn_reason": str(e)[:90]})
        per_image.append({"file": fn, "variant": meta["variant"], "distortion": meta["distortion"],
                          "bold_gt": meta["bold_gt"], "runs": runs_out})
        sl = " ".join(f"{r['warn_status'][:4]}(hb={r.get('header_bold')})" for r in runs_out)
        print(f"{fn:30s} gt_bold={str(meta['bold_gt']):5s} {sl}")

    # ---- metrics over all runs ----
    def flat(pred):  # (variant_filter) -> list of run dicts
        return [r for im in per_image if pred(im) for r in im["runs"] if r["warn_status"] != "ERR"]

    notbold = flat(lambda im: im["variant"] == "notbold")
    boldbody = flat(lambda im: im["variant"] == "boldbody")
    compliant = flat(lambda im: im["variant"] == "bold_compliant")
    titlecase = flat(lambda im: im["variant"] == "titlecase")
    allruns = flat(lambda im: True)

    fp_nb = [r for r in notbold if r["warn_status"] == PASS]
    fp_bb = [r for r in boldbody if r["warn_status"] == PASS]
    ff_co = [r for r in compliant if r["warn_status"] == FAIL]

    def counts(rs):
        return {"pass": sum(1 for r in rs if r["warn_status"] == PASS),
                "needs_review": sum(1 for r in rs if r["warn_status"] == REVIEW),
                "fail": sum(1 for r in rs if r["warn_status"] == FAIL), "n": len(rs)}

    metrics = {
        "policy": WARNING_BOLD_POLICY, "model": EXTRACTION_MODEL, "runs_per_image": runs,
        "false_pass_notbold": [len(fp_nb), len(notbold)],
        "false_pass_boldbody": [len(fp_bb), len(boldbody)],
        "false_fail_compliant_bold": [len(ff_co), len(compliant)],
        "warning_counts_overall": counts(allruns),
        "warning_counts_by_variant": {v: counts(flat(lambda im, v=v: im["variant"] == v))
                                      for v in ("bold_compliant", "notbold", "titlecase", "boldbody")},
        # images where a violation auto-passed at least once (the existence question)
        "violations_autopassed": sorted({im["file"] for im in per_image
                                         if im["bold_gt"] is False
                                         and any(r["warn_status"] == PASS for r in im["runs"])}),
    }

    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    json.dump({"metrics": metrics, "per_image": per_image},
              open(os.path.join(ROOT, "artifacts", "confidence_gate_safety_results.json"),
                   "w", encoding="utf-8"), indent=2)

    def pct(xy):
        x, n = xy
        return f"{x}/{n} ({100*x/n:.0f}%)" if n else f"{x}/0 (n/a)"
    L = [f"# Production {metrics['policy']} — bold-safety measurement (benchmark-only)", "",
         f"Production path used as-is: `extract_fields` ({metrics['model']}) -> `verify_label_only` "
         f"(`WARNING_BOLD_POLICY={metrics['policy']}`). {runs} runs/image on `bold_safety/`. "
         "Diagnostic only; nothing wired into production.", "",
         "| metric | result |", "|---|---|",
         f"| FALSE-PASS — not-bold headers (warning verdict = PASS) | {pct(metrics['false_pass_notbold'])} |",
         f"| FALSE-PASS — all-bold-body violations (verdict = PASS) | {pct(metrics['false_pass_boldbody'])} |",
         f"| FALSE-FAIL — compliant bold (verdict = FAIL) | {pct(metrics['false_fail_compliant_bold'])} |", "",
         "Warning-verdict counts (over all runs): "
         f"PASS {metrics['warning_counts_overall']['pass']} · "
         f"REVIEW {metrics['warning_counts_overall']['needs_review']} · "
         f"FAIL {metrics['warning_counts_overall']['fail']} (n={metrics['warning_counts_overall']['n']})", "",
         "Per variant (PASS / REVIEW / FAIL):"]
    for v, c in metrics["warning_counts_by_variant"].items():
        L.append(f"- `{v}` (gt_bold={'True' if v in ('bold_compliant','titlecase') else 'False'}): "
                 f"{c['pass']} / {c['needs_review']} / {c['fail']}  (n={c['n']})")
    L += ["", "Violations that auto-PASSED at least once:"]
    L += [f"- `{f}`" for f in metrics["violations_autopassed"]] or ["- (none)"]
    open(os.path.join(ROOT, "artifacts", "confidence_gate_safety_results.md"),
         "w", encoding="utf-8").write("\n".join(L))

    print("\n" + "=" * 72)
    print(f"BOLD-GATE SAFETY  (policy={metrics['policy']}, {runs} runs/image)\n")
    print(f"  FALSE-PASS not-bold headers:     {pct(metrics['false_pass_notbold'])}")
    print(f"  FALSE-PASS all-bold-body:        {pct(metrics['false_pass_boldbody'])}")
    print(f"  FALSE-FAIL compliant bold:       {pct(metrics['false_fail_compliant_bold'])}")
    print(f"  warning verdicts: PASS {metrics['warning_counts_overall']['pass']} / "
          f"REVIEW {metrics['warning_counts_overall']['needs_review']} / "
          f"FAIL {metrics['warning_counts_overall']['fail']}")
    print(f"  violations auto-passed >=1x: {len(metrics['violations_autopassed'])} "
          f"-> {metrics['violations_autopassed']}")
    print("\nartifacts/confidence_gate_safety_results.md / .json")


if __name__ == "__main__":
    main()
