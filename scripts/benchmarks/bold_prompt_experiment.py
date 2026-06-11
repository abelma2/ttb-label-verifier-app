"""Benchmark-only: does a stricter, FORENSIC prompt improve gpt-5.4-mini's bold judgment?

Prompt-ONLY experiment (no cropping -- isolates the prompt effect). Sends each image as-is with
three prompt variants and asks the same strict-JSON bold question, then scores against controlled
ground truth. The safety metric is the FALSE-PASS rate (header not truly bolder than the body, but
the model says header_bold=true).

NOTHING here touches production: extraction.py / verification.py / app.py / WARNING_BOLD_POLICY are
untouched and not imported for judgment -- only the OpenAI client plumbing (_get_client /
_model_params / _create_with_fallbacks) is reused read-only. The model never decides compliance;
this only measures whether it can visually compare stroke thickness.

Image sets: adversarial/ (font-controlled GT), bold_safety/ (controlled GT + distortions, from
scripts/benchmarks/generate_bold_safety.py), and test_labels/baseline_labels/*Other* (compliant backs, GT bold).

Run:  python scripts/benchmarks/bold_prompt_experiment.py
Outputs: artifacts/bold_prompt_experiment_results.md / .json
"""
import base64
import json
import os
import sys
import time

import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from smoke_test import _gather, _media_type, _load_key

# Strict JSON the model must return (exactly the requested shape).
_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["header_bold", "header_bold_confidence", "header_bold_basis", "can_verify_bold"],
    "properties": {
        "header_bold": {"type": ["boolean", "null"]},
        "header_bold_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "header_bold_basis": {"type": ["string", "null"]},
        "can_verify_bold": {"type": "boolean"},
    },
}

# Variant 1 -- BASELINE: a faithful standalone of the production _PROMPT bold instruction
# (note: it still lists "DARKER" as sufficient, which the forensic prompt removes).
_BASELINE = (
    "Look at the government warning on this alcohol label. Compare the stroke weight of the printed "
    "'GOVERNMENT WARNING' header to the warning body text immediately after it. header_bold = true "
    "ONLY if the header strokes are visibly HEAVIER / THICKER / DARKER than that body text; false if "
    "the header is the SAME weight, lighter, or not bold; null only if you genuinely cannot compare "
    "(blur, glare, cropping, tiny text, or no body text to compare against). header_bold_basis: a "
    "short phrase on what you saw. header_bold_confidence: high/medium/low. can_verify_bold: whether "
    "the image is good enough to judge. Return strict JSON only; do not judge legal compliance."
)

# Variant 2 -- FORENSIC stroke-weight prompt.
_FORENSIC = (
    "Look ONLY at the government warning block. Compare the printed letter strokes of "
    "'GOVERNMENT WARNING:' against the body text immediately following it, beginning with "
    "'(1) According to the Surgeon General, women should not drink alcoholic beverages during "
    "pregnancy because of the risk of birth defects. (2) Consumption of alcoholic beverages impairs "
    "your ability to drive a car or operate machinery, and may cause health problems.' Question: are "
    "the letters in 'GOVERNMENT WARNING:' VISIBLY THICKER / HEAVIER than the adjacent body text?\n"
    "This is a VISUAL STROKE-WEIGHT comparison:\n"
    "- Bold means visibly thicker/heavier letter strokes, especially the VERTICAL strokes.\n"
    "- Do NOT infer bold from all-caps. All-caps is NOT bold.\n"
    "- Do NOT infer bold because government warnings are usually bold.\n"
    "- Do NOT rely on OCR/text alone -- judge the actual printed strokes.\n"
    "- Do NOT treat 'darker' as sufficient if the darkness could be caused by blur, shadow, image "
    "compression, ink density, or contrast.\n"
    "Return header_bold = true ONLY if the header strokes are CLEARLY thicker/heavier than the body. "
    "Return false if the header is all caps but the SAME weight as the body. Return false if the body "
    "text is equally bold. Return false if the header only appears darker because of image quality. "
    "Return null ONLY if the image is too blurry, tiny, cropped, glared, or low-resolution to compare.\n"
    "header_bold_basis MUST mention BOTH the header and the body text. "
    "header_bold_confidence: high = clear close-up where header/body stroke thickness compares easily; "
    "medium = comparison is possible but the image is somewhat small, angled, compressed, or marginal; "
    "low = small, blurry, glared, shadowed, cropped, low-resolution, or uncertain. "
    "can_verify_bold = whether the image is good enough to make this comparison at all. "
    "You do NOT decide legal compliance. Return strict JSON only."
)

# Variant 3 -- FORENSIC + explicit negative examples.
_FORENSIC_NEG = _FORENSIC + (
    "\nNEGATIVE EXAMPLES (these are NOT 'header thicker than body' -> header_bold=false):\n"
    "- 'GOVERNMENT WARNING' in all caps at the SAME stroke weight as the body.\n"
    "- Header and body the same stroke weight.\n"
    "- Body text equally bold, so the header is not thicker than the body.\n"
    "- Header looks darker only due to blur/shadow/compression (strokes not actually thicker).\n"
    "- Do not auto-pass just because government warnings are usually bold."
)

# Variant 4 -- FORENSIC + NEGATIVES + MULTIPLE-CHOICE: the model classifies the visual
# RELATIONSHIP (A-E); code maps it to header_bold, so the model never picks the conclusion.
_CHOICE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["visual_relationship", "header_bold_confidence", "header_bold_basis"],
    "properties": {
        "visual_relationship": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
        "header_bold_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "header_bold_basis": {"type": ["string", "null"]},
    },
}
_CHOICE = (
    "You are NOT judging compliance. You are ONLY classifying the visual stroke-weight relationship "
    "between the government-warning header and its body text.\n\n"
    "Look ONLY at the warning block. Compare the printed letter strokes of 'GOVERNMENT WARNING:' to "
    "the body text immediately after it, beginning with '(1) According to the Surgeon General, women "
    "should not drink alcoholic beverages during pregnancy because of the risk of birth defects. (2) "
    "Consumption of alcoholic beverages impairs your ability to drive a car or operate machinery, and "
    "may cause health problems.'\n\n"
    "Choose the ONE visual_relationship that best fits:\n"
    "A = header strokes are CLEARLY thicker/heavier than the body strokes\n"
    "B = header is all caps but the SAME stroke weight as the body\n"
    "C = header and body are BOTH bold / equally heavy\n"
    "D = header appears darker, but the thickness difference is unclear\n"
    "E = image quality prevents the comparison (too blurry, tiny, cropped, glared, low-res)\n\n"
    "Rules:\n"
    "- Focus on STROKE THICKNESS, not darkness. Darker text is NOT enough unless the letter strokes "
    "themselves are visibly thicker.\n"
    "- All-caps letters look visually denser than lowercase body text. Do NOT count uppercase density, "
    "letter height, or overall darkness as bold -- compare the thickness of the actual letter strokes.\n"
    "- If the body is equally bold, the header is NOT thicker than the body -> choose C.\n"
    "- Do NOT assume government warnings are usually bold. Assume this MAY be an adversarial test: the "
    "header may be all caps but not bold, or both header and body may be bold.\n"
    "- When uncertain between A and B/C/D, do NOT choose A. The safe answer is 'not clearly thicker'.\n\n"
    "header_bold_confidence: high ONLY if individual letter strokes in BOTH header and body are clearly "
    "visible; medium if strokes are visible enough to compare; low if the decision depends on overall "
    "darkness, blur, contrast, or guesswork.\n"
    "header_bold_basis MUST mention BOTH the header and the body. Return strict JSON only."
)


def _derive_direct(raw):
    return {"header_bold": raw.get("header_bold"), "confidence": raw.get("header_bold_confidence"),
            "basis": raw.get("header_bold_basis"), "can_verify": raw.get("can_verify_bold")}


def _derive_choice(raw):
    rel = raw.get("visual_relationship")
    hb = {"A": True, "B": False, "C": False, "D": False, "E": None}.get(rel)
    return {"header_bold": hb, "confidence": raw.get("header_bold_confidence"),
            "basis": raw.get("header_bold_basis"), "can_verify": rel != "E",
            "visual_relationship": rel}


# name -> (prompt, json schema, derive function mapping the raw response to header_bold)
VARIANTS = {
    "baseline": (_BASELINE, _SCHEMA, _derive_direct),
    "forensic": (_FORENSIC, _SCHEMA, _derive_direct),
    "forensic_neg": (_FORENSIC_NEG, _SCHEMA, _derive_direct),
    "forensic_neg_choice": (_CHOICE, _CHOICE_SCHEMA, _derive_choice),
}

ADV_GT = {"01_compliant": True, "02_titlecase": True, "03_notbold": False, "04_reworded": True}
ADV_NEG = {"03_notbold": "notbold"}  # the GT=False adversarial case is a regular-weight header


def _img_block(b, mime):
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{base64.b64encode(b).decode()}", "detail": "high"}}


def _gather_items():
    items = []
    for f in _gather([os.path.join(ROOT, "adversarial")]):
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem in ADV_GT:
            items.append({"path": f, "name": stem, "gt": ADV_GT[stem],
                          "neg": ADV_NEG.get(stem), "dist": "clean", "set": "adversarial"})
    man = os.path.join(ROOT, "bold_safety", "manifest.json")
    if os.path.exists(man):
        manifest = json.load(open(man, encoding="utf-8"))
        for fn, m in sorted(manifest.items()):
            items.append({"path": os.path.join(ROOT, "bold_safety", fn), "name": fn, "gt": m["bold_gt"],
                          "neg": m["variant"] if m["variant"] in ("notbold", "boldbody") else None,
                          "dist": m["distortion"], "set": "bold_safety"})
    for f in _gather([os.path.join(ROOT, "test_labels", "baseline_labels")]):
        if "other" in os.path.basename(f).lower():  # the warning is on the back/Other label
            items.append({"path": f, "name": os.path.splitext(os.path.basename(f))[0], "gt": True,
                          "neg": None, "dist": "clean", "set": "baseline"})
    return items


def _ask(model, prompt, schema, img_bytes, mime):
    from extraction import _get_client, _model_params, _create_with_fallbacks
    rf = {"type": "json_schema", "json_schema": {"name": "bold_judgment", "strict": True, "schema": schema}}
    params = _model_params(model, response_format=rf)
    content = [{"type": "text", "text": prompt}, _img_block(img_bytes, mime)]
    t = time.perf_counter()
    resp = _create_with_fallbacks(_get_client(), content, params)
    secs = time.perf_counter() - t
    return json.loads(resp.choices[0].message.content), round(secs, 2)


def main():
    if not _load_key():
        sys.exit("ERROR: no OpenAI key (OPENAI_API_KEY env or .streamlit/secrets.toml).")
    from config import EXTRACTION_MODEL
    model = EXTRACTION_MODEL

    items = _gather_items()
    if not items:
        sys.exit("No images found (adversarial/, bold_safety/, baseline_labels/).")

    per_image = []
    for it in items:
        data = open(it["path"], "rb").read()
        mime = _media_type(it["path"])
        res = {}
        for vname, (prompt, schema, derive) in VARIANTS.items():
            try:
                raw, secs = _ask(model, prompt, schema, data, mime)
                d = derive(raw)
                pred = d["header_bold"]
                res[vname] = {"header_bold": pred, "confidence": d["confidence"],
                              "basis": d["basis"], "can_verify": d["can_verify"],
                              "visual_relationship": d.get("visual_relationship"),
                              "seconds": secs, "correct": (it["gt"] is not None and pred == it["gt"])}
            except Exception as e:
                res[vname] = {"error": str(e)[:160]}
        per_image.append({**{k: it[k] for k in ("name", "set", "gt", "neg", "dist")}, "results": res})
        line = "  ".join(f"{v}={str(res[v].get('header_bold')):5s}/{(res[v].get('confidence') or '-')[:3]}"
                         for v in VARIANTS)
        print(f"{it['name']:30s} gt={str(it['gt']):5s} {line}")

    # ---- metrics per variant ----
    def definite(rows, v):
        return [r for r in rows if "error" not in r["results"][v] and r["results"][v]["header_bold"] is not None]

    metrics = {}
    for v in VARIANTS:
        notbold = [r for r in per_image if r["neg"] == "notbold"]
        boldbody = [r for r in per_image if r["neg"] == "boldbody"]
        compliant = [r for r in per_image if r["gt"] is True]
        def_nb, def_bb, def_co = definite(notbold, v), definite(boldbody, v), definite(compliant, v)
        fp_nb = [r for r in def_nb if r["results"][v]["header_bold"] is True]
        fp_bb = [r for r in def_bb if r["results"][v]["header_bold"] is True]
        ff = [r for r in def_co if r["results"][v]["header_bold"] is False]
        allrows = [r for r in per_image if "error" not in r["results"][v]]
        nulls = [r for r in allrows if r["results"][v]["header_bold"] is None or
                 r["results"][v]["confidence"] == "low" or r["results"][v]["can_verify"] is False]
        defall = definite(per_image, v)
        acc = sum(1 for r in defall if r["results"][v]["correct"])
        by_dist = {}
        for dn in sorted(set(r["dist"] for r in per_image if r["set"] == "bold_safety")):
            sub = definite([r for r in per_image if r["set"] == "bold_safety" and r["dist"] == dn], v)
            by_dist[dn] = [sum(1 for r in sub if r["results"][v]["correct"]), len(sub)]
        conf_wrong = [{"name": r["name"], "gt": r["gt"], "pred": r["results"][v]["header_bold"],
                       "conf": r["results"][v]["confidence"], "basis": r["results"][v]["basis"]}
                      for r in defall if not r["results"][v]["correct"]
                      and r["results"][v]["confidence"] in ("high", "medium")]
        lat = [r["results"][v]["seconds"] for r in allrows]
        metrics[v] = {
            "false_pass_notbold": [len(fp_nb), len(def_nb)],
            "false_pass_boldbody": [len(fp_bb), len(def_bb)],
            "false_pass_all": [len(fp_nb) + len(fp_bb), len(def_nb) + len(def_bb)],
            "false_fail_compliant": [len(ff), len(def_co)],
            "null_or_low": [len(nulls), len(allrows)],
            "accuracy": [acc, len(defall)],
            "by_distortion": by_dist,
            "avg_latency": round(sum(lat) / len(lat), 2) if lat else 0,
            "confident_wrong": conf_wrong,
        }

    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    out_json = os.path.join(ROOT, "artifacts", "bold_prompt_experiment_results.json")
    json.dump({"model": model, "variants": list(VARIANTS), "n_images": len(items),
               "per_image": per_image, "metrics": metrics},
              open(out_json, "w", encoding="utf-8"), indent=2)

    # ---- markdown report ----
    def pct(xy):
        x, n = xy
        return f"{x}/{n} ({100*x/n:.0f}%)" if n else f"{x}/0 (n/a)"
    L = ["# Bold prompt experiment (benchmark-only)", "",
         f"Model: `{model}` · prompt-only (no crop) · {len(items)} images "
         f"(adversarial + bold_safety + baseline backs). Variants: {', '.join(VARIANTS)}.",
         "Safety metric = **false-pass** (header not truly bolder than body, model says `true`). "
         "The model never decides compliance; `verification.py` remains the only judge.", "",
         "| metric | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    rows = [("1. false-pass — not-bold headers", "false_pass_notbold"),
            ("2. false-pass — bold-body violations", "false_pass_boldbody"),
            ("   false-pass — all not-bold", "false_pass_all"),
            ("3. false-fail — compliant bold", "false_fail_compliant"),
            ("4. null / low-confidence", "null_or_low"),
            ("6. accuracy (definite preds)", "accuracy")]
    for label, key in rows:
        L.append(f"| {label} | " + " | ".join(pct(metrics[v][key]) for v in VARIANTS) + " |")
    L.append("| 7. avg latency | " + " | ".join(f"{metrics[v]['avg_latency']}s" for v in VARIANTS) + " |")
    L += ["", "## 5. Accuracy by distortion (bold_safety, definite preds)", "",
          "| distortion | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    for dn in sorted(set(r["dist"] for r in per_image if r["set"] == "bold_safety")):
        L.append(f"| {dn} | " + " | ".join(pct(metrics[v]["by_distortion"][dn]) for v in VARIANTS) + " |")
    L += ["", "## 8. Confident wrong answers (high/medium confidence, wrong)"]
    for v in VARIANTS:
        L.append(f"\n**{v}** ({len(metrics[v]['confident_wrong'])}):")
        for c in metrics[v]["confident_wrong"]:
            L.append(f"- `{c['name']}` gt={c['gt']} pred={c['pred']} ({c['conf']}) — {c['basis']}")
    open(os.path.join(ROOT, "artifacts", "bold_prompt_experiment_results.md"),
         "w", encoding="utf-8").write("\n".join(L))

    # ---- console summary ----
    print("\n" + "=" * 72)
    print(f"BOLD PROMPT EXPERIMENT  ({len(items)} images, model {model})\n")
    print(f"{'metric':34s} " + " ".join(f"{v:>14s}" for v in VARIANTS))
    for label, key in rows:
        print(f"{label:34s} " + " ".join(f"{pct(metrics[v][key]):>14s}" for v in VARIANTS))
    print(f"{'7. avg latency':34s} " + " ".join(f"{str(metrics[v]['avg_latency'])+'s':>14s}" for v in VARIANTS))
    print(f"\nartifacts/bold_prompt_experiment_results.md / .json")


if __name__ == "__main__":
    main()
