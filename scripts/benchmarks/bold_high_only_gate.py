"""Benchmark-only: compare the confidence_gate rule (the production default at the time) against a
stricter 'high-only' rule, re-using the raw header_bold / header_bold_confidence already captured
in artifacts/confidence_gate_safety_results.json (NO new API calls).

Gate rules compared (applied to the production reads, in isolation):
  current   : header_bold is True AND confidence in {medium, high}  -> auto-PASS bold
  high-only : header_bold is True AND confidence == high            -> auto-PASS bold
              anything else -> not auto-passed (fail-closed / needs review)

Nothing is modified or wired in: app.py / extraction.py / verification.py / config.py /
WARNING_BOLD_POLICY are untouched. This only re-scores existing data. Diagnostic only.

Run:  python scripts/benchmarks/bold_high_only_gate.py
Outputs: artifacts/bold_high_only_gate_results.md / .json
"""
import json
import os
import sys

import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT
ART = os.path.join(ROOT, "artifacts")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SRC = os.path.join(ART, "confidence_gate_safety_results.json")


def gate_current(hb, conf):
    return hb is True and conf in ("medium", "high")


def gate_high(hb, conf):
    return hb is True and conf == "high"


def main():
    if not os.path.exists(SRC):
        sys.exit("Missing artifacts/confidence_gate_safety_results.json -- run "
                 "scripts/benchmarks/confidence_gate_safety.py first.")
    per_image = json.load(open(SRC, encoding="utf-8"))["per_image"]

    # flatten runs, tagged by variant (only runs with a definite header_bold read are gate-relevant)
    runs = []
    for im in per_image:
        for r in im["runs"]:
            runs.append({"file": im["file"], "variant": im["variant"], "dist": im["distortion"],
                         "bold_gt": im["bold_gt"], "hb": r.get("header_bold"),
                         "conf": r.get("header_bold_confidence")})

    def subset(v):
        return [r for r in runs if r["variant"] == v]
    notbold, boldbody, compliant, titlecase = (subset("notbold"), subset("boldbody"),
                                              subset("bold_compliant"), subset("titlecase"))

    def rate(rows, fn):  # fraction of rows where fn(row) is True
        return [sum(1 for r in rows if fn(r)), len(rows)]

    # false-pass = gate auto-passes a violation; false-fail = gate fails to pass a compliant-bold label
    m = {
        "1_current_fp_notbold":  rate(notbold,  lambda r: gate_current(r["hb"], r["conf"])),
        "2_highonly_fp_notbold": rate(notbold,  lambda r: gate_high(r["hb"], r["conf"])),
        "3_current_fp_boldbody":  rate(boldbody, lambda r: gate_current(r["hb"], r["conf"])),
        "4_highonly_fp_boldbody": rate(boldbody, lambda r: gate_high(r["hb"], r["conf"])),
        "5_current_ff_compliant":  rate(compliant, lambda r: not gate_current(r["hb"], r["conf"])),
        "6_highonly_ff_compliant": rate(compliant, lambda r: not gate_high(r["hb"], r["conf"])),
    }
    # 7. current false-passes (notbold + boldbody) by confidence
    cur_fps = [r for r in notbold + boldbody if gate_current(r["hb"], r["conf"])]
    m["7_current_fp_by_conf"] = {"medium": sum(1 for r in cur_fps if r["conf"] == "medium"),
                                 "high": sum(1 for r in cur_fps if r["conf"] == "high"),
                                 "total": len(cur_fps)}
    # 8. compliant runs that newly fail under high-only (were medium-confidence True)
    newly = [r["file"] for r in compliant
             if gate_current(r["hb"], r["conf"]) and not gate_high(r["hb"], r["conf"])]
    m["8_compliant_newly_failing"] = {"count": len(newly), "of": len(compliant), "files": sorted(set(newly))}
    # residual high-confidence false-passes that high-only does NOT fix
    m["residual_high_fp"] = {
        "notbold": [r["file"] for r in notbold if gate_high(r["hb"], r["conf"])],
        "boldbody": [r["file"] for r in boldbody if gate_high(r["hb"], r["conf"])],
    }

    json.dump({"note": "re-scored from confidence_gate_safety_results.json; no new API calls",
               "metrics": m}, open(os.path.join(ART, "bold_high_only_gate_results.json"), "w",
                                   encoding="utf-8"), indent=2)

    def pct(xy):
        x, n = xy
        return f"{x}/{n} ({100*x/n:.0f}%)" if n else "n/a"
    fp7 = m["7_current_fp_by_conf"]
    L = ["# Stricter bold gate: current (medium+high) vs high-only (benchmark-only)", "",
         "Re-scored from `artifacts/confidence_gate_safety_results.json` (production reads on the "
         "controlled `bold_safety` set, 3 runs/image) -- **no new API calls**. Gate applied to the "
         "bold sub-decision in isolation; nothing wired into production.", "",
         "| metric | current (med+high) | high-only |",
         "|---|---|---|",
         f"| 1/2 false-pass — not-bold headers | {pct(m['1_current_fp_notbold'])} | {pct(m['2_highonly_fp_notbold'])} |",
         f"| 3/4 false-pass — all-bold-body | {pct(m['3_current_fp_boldbody'])} | {pct(m['4_highonly_fp_boldbody'])} |",
         f"| 5/6 false-fail — compliant bold | {pct(m['5_current_ff_compliant'])} | {pct(m['6_highonly_ff_compliant'])} |", "",
         f"**7. Current false-passes by confidence:** medium={fp7['medium']}, high={fp7['high']} (of {fp7['total']}).",
         f"**8. Compliant labels that NEWLY fail/review under high-only** (were medium-confidence True): "
         f"{m['8_compliant_newly_failing']['count']}/{m['8_compliant_newly_failing']['of']}.",
         f"**Residual HIGH-confidence false-passes high-only does NOT fix:** "
         f"not-bold={len(m['residual_high_fp']['notbold'])}, all-bold-body={len(m['residual_high_fp']['boldbody'])}.",
         f"  - all-bold-body (structural, header-only gate can't see the bold body): {m['residual_high_fp']['boldbody']}", ""]
    open(os.path.join(ART, "bold_high_only_gate_results.md"), "w", encoding="utf-8").write("\n".join(L))

    print("=" * 72)
    print("STRICTER BOLD GATE: current (med+high) vs high-only  (controlled bold_safety, re-scored)\n")
    print(f"  false-pass not-bold:      current {pct(m['1_current_fp_notbold'])}   high-only {pct(m['2_highonly_fp_notbold'])}")
    print(f"  false-pass all-bold-body: current {pct(m['3_current_fp_boldbody'])}   high-only {pct(m['4_highonly_fp_boldbody'])}")
    print(f"  false-fail compliant:     current {pct(m['5_current_ff_compliant'])}   high-only {pct(m['6_highonly_ff_compliant'])}")
    print(f"  current false-passes by conf: medium={fp7['medium']} high={fp7['high']} (of {fp7['total']})")
    print(f"  compliant newly failing under high-only: {m['8_compliant_newly_failing']['count']}/{m['8_compliant_newly_failing']['of']}")
    print(f"  residual HIGH-conf false-pass: not-bold={len(m['residual_high_fp']['notbold'])} "
          f"all-bold-body={len(m['residual_high_fp']['boldbody'])}")
    print("\nartifacts/bold_high_only_gate_results.md / .json")


if __name__ == "__main__":
    main()
