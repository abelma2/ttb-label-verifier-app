"""Government-warning BOLD prompt-variant benchmark (BENCHMARK ONLY).

Tests several SHORT warning-only prompt styles for gpt-5.4-mini to see which elicits the SAFEST
bold evidence (lowest false-pass risk, most high-confidence-correct reads, most stable across
repeats). It does NOT judge compliance, does NOT use the verifier, sees NO application data, and
does NOT vote across models. It changes NO production code — `extraction.py` / `verification.py` /
`config.py` / `app.py` are untouched; this only reads the shared request helpers.

Each variant elicits the SAME warning-only structured schema:
  warning_present:boolean · header_text_seen:str|null · header_bold:true/false/null ·
  header_bold_confidence:high/medium/low · body_bold:true/false/null ·
  body_bold_confidence:high/medium/low · short_basis:str · image_quality_notes:str|null

Variants (E removed):
  P  production control  -- the current extraction.py government_warning bold wording, adapted
  A  direct minimal
  B  relative comparison
  C  anti-confusion
  D  conservative uncertainty
  F  same-weight trap
  G  evidence-first basis

Input: the controlled `bold_safety/` set (filenames encode the expected class):
  bold_compliant -> expect header_bold=true, body_bold=false
  boldbody       -> expect body_bold=true       (all-bold remainder = the dangerous violation)
  notbold        -> expect header_bold=false     (regular-weight header)
  titlecase      -> capitalization case; bold observations recorded but NOT scored as bold

Robustness: every (variant, image) is called `--repeats` (default 3) INDEPENDENT times; stability
across the repeats is reported (value-stable 3/3, value changed, confidence changed, API errors/
retries). A prompt that only looks good in one run but is unstable across repeats is not trusted.

Model: env `WARNING_BOLD_PROMPT_MODEL`, default `config.EXTRACTION_MODEL`.

Run (calls the real model -- needs an API key, costs money):
  python scripts/benchmarks/warning_bold_prompt_variants.py                       # 7 variants x 20 imgs x 3
  python scripts/benchmarks/warning_bold_prompt_variants.py --variants P,A,F --repeats 3
  python scripts/benchmarks/warning_bold_prompt_variants.py --workers 6 --detail high
Outputs: output/warning_bold_prompt_variants_<ts>.json / .txt
"""
import base64
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
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

import config
from extraction import _get_client, _model_params, _create_with_fallbacks

BS = os.path.join(ROOT, "bold_safety")
MODEL = os.environ.get("WARNING_BOLD_PROMPT_MODEL", config.EXTRACTION_MODEL)

# --- warning-only structured schema (Structured Outputs) ----------------------
_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["warning_present", "header_text_seen", "header_bold", "header_bold_confidence",
                 "body_bold", "body_bold_confidence", "short_basis", "image_quality_notes"],
    "properties": {
        "warning_present": {"type": "boolean"},
        "header_text_seen": {"type": ["string", "null"]},
        "header_bold": {"type": ["boolean", "null"]},
        "header_bold_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "body_bold": {"type": ["boolean", "null"]},
        "body_bold_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "short_basis": {"type": "string"},
        "image_quality_notes": {"type": ["string", "null"]},
    },
}


def _rf():
    return {"type": "json_schema", "json_schema": {"name": "warning_bold", "strict": True, "schema": _SCHEMA}}


# --- common field plumbing (kept NEUTRAL so only the per-variant judgment guidance varies) -----
_COMMON = (
    "You are shown ONE image of a U.S. alcohol beverage label. Look only at the government health "
    "warning (the paragraph beginning 'GOVERNMENT WARNING:'). Report ONLY what you visually SEE as "
    "a JSON object with EXACTLY these fields:\n"
    "  warning_present: true if the GOVERNMENT WARNING statement is visible, else false.\n"
    "  header_text_seen: the header words you see verbatim (e.g. 'GOVERNMENT WARNING'), or null.\n"
    "  header_bold: true / false / null  -- your reading of the HEADER's font weight.\n"
    "  header_bold_confidence: high / medium / low.\n"
    "  body_bold: true / false / null  -- your reading of the warning BODY's font weight.\n"
    "  body_bold_confidence: high / medium / low.\n"
    "  short_basis: one short sentence describing what you actually saw about the stroke weights.\n"
    "  image_quality_notes: any glare/blur/angle/crop/low-resolution that limited your reading, or null.\n"
    "Do NOT judge legal compliance. Output JSON only.\n\nINSTRUCTION:\n"
)

# Variant P -- production control: the current extraction.py government_warning bold wording,
# adapted to this warning-only schema (keeps the production "heavier/thicker/darker" cue, so the
# cleaner variants below have a faithful baseline to beat).
PROMPT_P = (
    "Compare the stroke weight of the printed 'GOVERNMENT WARNING' header to the warning body text "
    "IMMEDIATELY AFTER it. Set header_bold true ONLY if the header strokes are VISIBLY HEAVIER / "
    "THICKER / DARKER than that body text; false if the header is the SAME weight, lighter, or not "
    "bold; null ONLY if you genuinely cannot compare (blur, glare, cropping, tiny text, or no body "
    "text to compare against). Judge the ACTUAL printed strokes -- do not assume bold just because "
    "warning headers are usually bold. header_bold_confidence is how sure you are of THIS bold "
    "judgment specifically. For body_bold: does the warning body text itself (the sentences after "
    "the header) appear BOLD / heavy? Set body_bold true if the body's OWN letter strokes are "
    "visibly bold/heavy; false if the body is normal/regular weight; null only if you cannot tell. "
    "short_basis: one short phrase describing what you actually saw about the relative stroke weights."
)

PROMPT_A = (
    "Look only at the government warning. Compare the visible stroke weight of GOVERNMENT WARNING "
    "with the warning body text immediately after it. Is the header bold? Is the body bold? Do not "
    "infer from capitalization, size, darkness, contrast, or expectation. If uncertain, use "
    "null/low confidence."
)

PROMPT_B = (
    "Find GOVERNMENT WARNING and the warning body below/after it. Judge bold only by relative letter "
    "stroke thickness. Header is bold only if its letters are visibly thicker/heavier than the body "
    "letters. Body is bold only if the body letters themselves are heavy/bold. Ignore capitalization, "
    "font size, color, contrast, and layout prominence. If the strokes cannot be compared clearly, "
    "use null."
)

PROMPT_C = (
    "Report visual font weight only. All caps is not bold. Larger text is not bold. Darker ink is "
    "not bold. High contrast is not bold. A prominent header is not necessarily bold. For GOVERNMENT "
    "WARNING, report header_bold true only when the header strokes are clearly heavier than nearby "
    "warning body text. Report body_bold true only when the warning body strokes are clearly "
    "bold/heavy."
)

PROMPT_D = (
    "Inspect the government warning typography. Return bold=true only when you are clearly sure from "
    "visible letter strokes. Return false only when you are clearly sure it is regular/not bold. "
    "Otherwise return null with low confidence. Do not guess. Give a short basis describing the "
    "stroke-weight comparison."
)

PROMPT_F = (
    "Look at the government warning text. If GOVERNMENT WARNING appears to have the same stroke "
    "thickness as the warning body, report header_bold=false. Report header_bold=true only if the "
    "header strokes are clearly thicker than the body strokes. Report body_bold=true only if the "
    "body sentences are also heavy/bold. If you cannot compare stroke thickness, use null."
)

PROMPT_G = (
    "Inspect only the government warning. Before choosing true/false/null, identify the visual "
    "evidence: are the header letter strokes thicker, the same, or unclear compared with the body "
    "letter strokes? Then return header_bold, body_bold, confidence, and a short_basis that must "
    "mention stroke thickness. Do not use capitalization, size, darkness, contrast, or typical legal "
    "formatting."
)

# Variant H -- pixel-thickness / null
PROMPT_H = (
    "Look only at the government warning. Compare the actual pixel/stroke thickness of the letters in "
    "GOVERNMENT WARNING and the warning body. If the header strokes are clearly thicker than the body "
    "strokes, header_bold=true. If they are the same or unclear, header_bold=false or null. For "
    "body_bold, answer true only if the body strokes themselves are visibly thick/heavy. If exact "
    "stroke thickness cannot be compared, return null rather than guessing."
)

# Variant I -- body-first all-bold safety
PROMPT_I = (
    "Inspect the warning body first, then the header. Is the warning body itself bold/heavy? Then "
    "compare GOVERNMENT WARNING to that body text. Return body_bold=true if the body sentences have "
    "heavy/bold strokes, even if the header is also bold. Return header_bold=true only if the header "
    "is clearly heavier than the body. Do not assume the body is regular just because the header "
    "looks bold."
)

# Variant J -- relation categories
PROMPT_J = (
    "Inspect only visible stroke weight in the government warning. Decide the relationship between the "
    "header and body strokes: thicker, same, thinner, or unclear. Use that relationship to fill the "
    "fields. header_bold=true only for thicker. header_bold=false for same or thinner. body_bold=true "
    "if the body strokes are visibly heavy/bold on their own. Use null when the relationship is unclear."
)

VARIANTS = {"P": PROMPT_P, "A": PROMPT_A, "B": PROMPT_B, "C": PROMPT_C,
            "D": PROMPT_D, "F": PROMPT_F, "G": PROMPT_G,
            "H": PROMPT_H, "I": PROMPT_I, "J": PROMPT_J}


# --- image plumbing -----------------------------------------------------------
def _media_type(path):
    return "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"


def _block(b, media_type, detail):
    return {"type": "image_url",
            "image_url": {"url": "data:%s;base64,%s" % (media_type, base64.b64encode(b).decode()),
                          "detail": detail}}


def _load_key():
    if os.environ.get("OPENAI_API_KEY"):
        return True
    secrets = os.path.join(ROOT, ".streamlit", "secrets.toml")
    if os.path.exists(secrets):
        with open(secrets, encoding="utf-8") as fh:
            m = re.search(r'OPENAI_API_KEY\s*=\s*"([^"]+)"', fh.read())
        if m and m.group(1) and m.group(1) != "sk-...":
            os.environ["OPENAI_API_KEY"] = m.group(1)
    return bool(os.environ.get("OPENAI_API_KEY"))


def _arg(args, flag, default):
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else default


def _expected_class(fn):
    n = fn.lower()
    if n.startswith("bold_compliant"):
        return "bold_compliant"
    if n.startswith("boldbody"):
        return "boldbody"
    if n.startswith("notbold"):
        return "notbold"
    if n.startswith("titlecase"):
        return "titlecase"
    return "unknown"


# --- one call (own params dict per call; outer retry for transient errors) ----
def _call(variant, path, detail, max_attempts):
    b = open(path, "rb").read()
    content = [{"type": "text", "text": _COMMON + VARIANTS[variant]},
               _block(b, _media_type(path), detail)]
    attempts, last_err = 0, None
    while attempts < max_attempts:
        attempts += 1
        params = _model_params(MODEL, response_format=_rf())  # fresh dict (caller mutates on fallback)
        t = time.perf_counter()
        try:
            resp = _create_with_fallbacks(_get_client(), content, params)
            dt = round(time.perf_counter() - t, 2)
            out = json.loads(resp.choices[0].message.content)
            return {"output": out, "latency": dt, "attempts": attempts, "error": None}
        except Exception as exc:
            last_err = str(exc)[:200]
            time.sleep(1.5 * attempts)
    return {"output": None, "latency": None, "attempts": attempts, "error": last_err}


# --- scoring helpers ----------------------------------------------------------
def _hb(o):
    return o.get("header_bold") if o else None


def _bb(o):
    return o.get("body_bold") if o else None


def _hc(o):
    return o.get("header_bold_confidence") if o else None


def _bc(o):
    return o.get("body_bold_confidence") if o else None


def _scorecard(records, repeats, variants, images):
    classes = {}
    for fn in images:
        classes.setdefault(_expected_class(fn), 0)
        classes[_expected_class(fn)] += 1
    n_compliant = classes.get("bold_compliant", 0) * repeats
    n_boldbody = classes.get("boldbody", 0) * repeats
    n_notbold = classes.get("notbold", 0) * repeats

    L = ["", "=" * 104, "WARNING-BOLD PROMPT-VARIANT BENCHMARK", "=" * 104,
         "model=%s  repeats=%d  images=%d (%s)  variants=%s" %
         (MODEL, repeats, len(images),
          " ".join("%s:%d" % (k, v) for k, v in sorted(classes.items())), ",".join(variants)),
         "Warning-only structured read; NO compliance judgment, NO verifier, NO voting. "
         "false-pass = a violation read as compliant.",
         "Per class, counts are over (images-in-class x %d repeats): compliant=%d  boldbody=%d  notbold=%d" %
         (repeats, n_compliant, n_boldbody, n_notbold), ""]

    summary = {}
    hdr = ("%-3s %-9s %-12s %-11s %-9s %-9s %-10s %-11s %-10s %-13s %-9s %-7s"
           % ("var", "compl✓", "boldbodyFP", "notboldFP", "hiconfFP", "uncert",
              "medCorr", "valStable", "valChg", "confChg", "err/rty", "trust"))

    rows = []
    for v in variants:
        recs = [r for r in records if r["variant"] == v]
        valid = [r for r in recs if r["output"] is not None]
        errors = [r for r in recs if r["output"] is None]
        retries = sum(max(0, r["attempts"] - 1) for r in recs)

        # per-class correctness / false-pass
        compl_ok = 0
        compl_total = 0
        boldbody_fp = boldbody_hi_fp = 0
        notbold_fp = notbold_hi_fp = 0
        uncertain = 0
        med_correct = 0
        fp_examples = []
        for r in valid:
            o, cls = r["output"], r["expected_class"]
            if cls == "bold_compliant":
                compl_total += 1
                if _hb(o) is True and _bb(o) is False:
                    compl_ok += 1
                    if _hc(o) == "medium" or _bc(o) == "medium":
                        med_correct += 1
                if _hb(o) is None or _bb(o) is None or _hc(o) == "low" or _bc(o) == "low":
                    uncertain += 1
            elif cls == "boldbody":
                if _bb(o) is False:                      # expected body_bold=True -> FALSE-PASS
                    boldbody_fp += 1
                    if _bc(o) == "high":
                        boldbody_hi_fp += 1
                        fp_examples.append((r["filename"], r["repeat"], "boldbody hi-conf", o))
                elif _bb(o) is True and _bc(o) == "medium":
                    med_correct += 1
                if _bb(o) is None or _bc(o) == "low":
                    uncertain += 1
            elif cls == "notbold":
                if _hb(o) is True:                       # expected header_bold=False -> FALSE-PASS
                    notbold_fp += 1
                    if _hc(o) == "high":
                        notbold_hi_fp += 1
                        fp_examples.append((r["filename"], r["repeat"], "notbold hi-conf", o))
                elif _hb(o) is False and _hc(o) == "medium":
                    med_correct += 1
                if _hb(o) is None or _hc(o) == "low":
                    uncertain += 1
        hi_fp = boldbody_hi_fp + notbold_hi_fp
        tot_fp = boldbody_fp + notbold_fp

        # stability per (filename) group across repeats
        val_stable = val_changed = conf_changed = 0
        n_groups = 0
        for fn in images:
            grp = sorted((r for r in recs if r["filename"] == fn), key=lambda r: r["repeat"])
            valid_grp = [r for r in grp if r["output"] is not None]
            if len(valid_grp) < 2:
                continue
            n_groups += 1
            vals = {(_hb(r["output"]), _bb(r["output"])) for r in valid_grp}
            confs = {(_hc(r["output"]), _bc(r["output"])) for r in valid_grp}
            if len(vals) == 1 and len(valid_grp) == repeats:
                val_stable += 1
            if len(vals) > 1:
                val_changed += 1
            if len(confs) > 1:
                conf_changed += 1

        lat = [r["latency"] for r in valid if r["latency"] is not None]
        lat_avg = round(statistics.mean(lat), 2) if lat else 0
        lat_med = round(statistics.median(lat), 2) if lat else 0
        lat_max = round(max(lat), 2) if lat else 0

        # trust heuristic: no high-confidence false-pass, no errors, value-stable on most
        # images, AND a meaningfully low total false-pass rate (<=20% of the violation reads —
        # per-class counts can never exceed the class totals, so comparing against the raw
        # totals would be vacuously true and let a stably-wrong variant earn "yes").
        stable_frac = (val_stable / n_groups) if n_groups else 0
        fp_budget = 0.2 * (n_boldbody + n_notbold)
        if errors:
            trust = "no(err)"
        elif hi_fp > 0:
            trust = "NO(hiFP)"
        elif tot_fp > fp_budget:
            trust = "no(FPrate)"
        elif stable_frac >= 0.7:
            trust = "yes" if stable_frac >= 0.85 else "border"
        else:
            trust = "border"

        summary[v] = {"compl_ok": compl_ok, "compl_total": compl_total, "boldbody_fp": boldbody_fp,
                      "notbold_fp": notbold_fp, "hi_fp": hi_fp, "tot_fp": tot_fp, "uncertain": uncertain,
                      "med_correct": med_correct, "val_stable": val_stable, "val_changed": val_changed,
                      "conf_changed": conf_changed, "n_groups": n_groups, "errors": len(errors),
                      "retries": retries, "lat_avg": lat_avg, "lat_med": lat_med, "lat_max": lat_max,
                      "stable_frac": round(stable_frac, 2), "trust": trust, "fp_examples": fp_examples}
        rows.append("%-3s %-9s %-12s %-11s %-9s %-9s %-10s %-11s %-9s %-13s %-9s %-7s"
                    % (v, "%d/%d" % (compl_ok, compl_total),
                       "%d/%d" % (boldbody_fp, n_boldbody), "%d/%d" % (notbold_fp, n_notbold),
                       str(hi_fp), str(uncertain), str(med_correct),
                       "%d/%d" % (val_stable, n_groups), str(val_changed), str(conf_changed),
                       "%d/%d" % (len(errors), retries), trust))

    L.append(hdr)
    L.append("-" * len(hdr))
    L += rows
    L.append("")
    L.append("Legend: compl✓ = bold_compliant read correctly (header bold & body not). "
             "boldbodyFP = all-bold body read as body_bold=false. notboldFP = regular header read as "
             "header_bold=true. hiconfFP = those false-passes at HIGH confidence (the dangerous ones). "
             "uncert = null-or-low on the class-relevant field. medCorr = correct but only medium conf. "
             "valStable = images with identical (header_bold,body_bold) across all %d repeats. "
             "valChg/confChg = images whose value/confidence changed across repeats." % repeats)

    # control comparison + recommendation
    L.append("")
    L.append("CONTROL = P (production wording). A variant is a candidate ONLY if it does NOT increase")
    L.append("high-confidence false-pass risk and does NOT increase total false-pass risk vs P, while")
    L.append("being at least as stable; preferred if it lowers false-pass or raises high-conf-correct.")
    p = summary.get("P")
    if p:
        L.append("  P (control): hiFP=%d totFP=%d compl=%d/%d valStable=%d/%d errors=%d"
                 % (p["hi_fp"], p["tot_fp"], p["compl_ok"], p["compl_total"],
                    p["val_stable"], p["n_groups"], p["errors"]))
        candidates = []
        for v in variants:
            if v == "P":
                continue
            s = summary[v]
            safer = (s["hi_fp"] <= p["hi_fp"] and s["tot_fp"] <= p["tot_fp"]
                     and s["val_stable"] >= p["val_stable"] - 1 and s["errors"] == 0)
            improves = (s["tot_fp"] < p["tot_fp"] or s["hi_fp"] < p["hi_fp"]
                        or s["compl_ok"] > p["compl_ok"])
            tag = "CANDIDATE" if (safer and improves) else ("safe-tie" if safer else "-")
            L.append("  %s: hiFP=%d totFP=%d compl=%d/%d valStable=%d/%d errors=%d  -> %s"
                     % (v, s["hi_fp"], s["tot_fp"], s["compl_ok"], s["compl_total"],
                        s["val_stable"], s["n_groups"], s["errors"], tag))
            if safer and improves:
                candidates.append(v)
        # rank candidates: lowest hiFP, lowest totFP, highest compl, most stable
        candidates.sort(key=lambda v: (summary[v]["hi_fp"], summary[v]["tot_fp"],
                                       -summary[v]["compl_ok"], -summary[v]["val_stable"]))
        L.append("")
        if candidates:
            L.append("RECOMMENDATION: candidate(s) that beat P without adding danger: %s. Safest = %s."
                     % (", ".join(candidates), candidates[0]))
            L.append("  (Recommendation only -- production extraction.py is NOT modified by this script.)")
        else:
            L.append("RECOMMENDATION: NO variant safely beats the production control P -- keep production "
                     "wording. None reduced false-pass risk without adding danger or instability.")

    # false-pass detail (the dangerous reads), capped
    L.append("")
    L.append("HIGH-CONFIDENCE FALSE-PASSES (dangerous: a violation called compliant at high confidence):")
    any_fp = False
    for v in variants:
        ex = summary[v]["fp_examples"]
        if ex:
            any_fp = True
            L.append("  variant %s:" % v)
            for fn, rep, kind, o in ex[:6]:
                L.append("    %-26s r%d %-16s hb=%s[%s] bb=%s[%s] basis=%r"
                         % (fn, rep, kind, _hb(o), _hc(o), _bb(o), _bc(o),
                            (o.get("short_basis") or "")[:80]))
    if not any_fp:
        L.append("  (none -- no variant produced a HIGH-confidence false-pass this run)")
    return "\n".join(L), summary


def main():
    args = sys.argv[1:]
    repeats = int(_arg(args, "--repeats", "3"))
    detail = _arg(args, "--detail", "high")
    workers = int(_arg(args, "--workers", "4"))
    max_attempts = int(_arg(args, "--max-attempts", "3"))
    variants = [v.strip().upper() for v in _arg(args, "--variants", ",".join(VARIANTS)).split(",") if v.strip()]
    variants = [v for v in variants if v in VARIANTS]

    if not os.path.isdir(BS):
        sys.exit("ERROR: missing bold_safety folder: %s" % BS)
    if not _load_key():
        sys.exit("ERROR: no OpenAI key (env OPENAI_API_KEY or .streamlit/secrets.toml).")

    images = sorted(f for f in os.listdir(BS) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    tasks = [(v, fn, r) for v in variants for fn in images for r in range(1, repeats + 1)]
    print("model=%s  variants=%s  images=%d  repeats=%d  workers=%d  total calls=%d\n"
          % (MODEL, ",".join(variants), len(images), repeats, workers, len(tasks)), flush=True)

    records = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_call, v, os.path.join(BS, fn), detail, max_attempts): (v, fn, r)
                for (v, fn, r) in tasks}
        for fut in as_completed(futs):
            v, fn, r = futs[fut]
            res = fut.result()
            records.append({"variant": v, "repeat": r, "filename": fn,
                            "expected_class": _expected_class(fn), **res})
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print("  %d/%d done" % (done, len(tasks)), flush=True)

    report, summary = _scorecard(records, repeats, variants, images)
    print(report)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt = os.path.join(OUT_DIR, "warning_bold_prompt_variants_%s.txt" % stamp)
    js = os.path.join(OUT_DIR, "warning_bold_prompt_variants_%s.json" % stamp)
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump({"model": MODEL, "repeats": repeats, "detail": detail, "variants": variants,
                   "prompts": {v: _COMMON + VARIANTS[v] for v in variants},
                   "summary": summary, "records": records}, fh, indent=2, ensure_ascii=False)
    print("\nWritten to:\n  %s\n  %s" % (txt, js))


if __name__ == "__main__":
    main()
