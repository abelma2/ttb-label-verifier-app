"""Benchmark-only: does a PRODUCTION-SCHEMA-COMPATIBLE minimal style-feature prompt beat the production
prompt? Unlike minimal_style_prompt_experiment.py (which used a different output schema), B here keeps the
EXACT production schema, the EXACT _coerce(), and the EXACT verifier -- the ONLY change is the
WORDING of the government_warning portion of the prompt (de-primed for bold, transcribe-first). So
if B wins, adoption is a one-string prompt swap, not a schema/verifier rewrite.

  A = production extraction.extract_fields(), used exactly as-is.
  B = extract_B(): production _PROMPT with ONLY the government_warning instruction block replaced by
      the minimal style-feature wording (_MINIMAL_WARN). Built by string-substituting that block out of the
      real _PROMPT, so every other field instruction and the JSON example stay byte-identical.
      Same _STRUCTURED_RF (production schema) + same _parse_response/_coerce + same verifier.

Both scored through the SAME unchanged verification.verify_label_only / _check_warning (live
WARNING_BOLD_POLICY). NOTHING in production is modified; this only reuses extraction.py /
verification.py read-only, the same pattern as the other benchmarks.

BENCHMARK ONLY. Does not touch app.py / verification.py / config.py / the production _PROMPT, does
not change WARNING_BOLD_POLICY, is not wired into Streamlit, and does not overwrite other artifacts.

Run:  python scripts/benchmarks/minimal_schema_prompt_experiment.py [runs]
Outputs: artifacts/minimal_schema_prompt_experiment_results.md / .json
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rapidfuzz import fuzz
from smoke_test import _gather, _group_by_product, _media_type, _load_key
from config import EXTRACTION_MODEL
from extraction import (_PROMPT, _build_content, _model_params, _get_client,
                        _create_with_fallbacks, _parse_response, extract_fields)
from verification import (_check_warning, PASS, REVIEW, FAIL, _normalize, _warning_body,
                          _CANONICAL_WARNING_BODY_NORM)

# --------------------------------------------------------- B = production prompt, warning reworded
# The de-primed warning block. Fills EVERY production government_warning field, but transcribe-first
# and explicitly anti-priming on bold (the minimal-prompt wording).
_MINIMAL_WARN = """- government_warning: the federal government health warning. Your job here is to \
transcribe the visible text and report visible style observations only -- NOT to decide compliance.
    * text: transcribe the ENTIRE warning exactly as printed, INCLUDING the "GOVERNMENT WARNING:" \
header words. Preserve original capitalization, punctuation, spacing, and symbols. Do NOT \
reconstruct the warning from memory and do NOT correct or normalize it to the expected federal \
wording -- transcribe ONLY the visible printed words. If a word is unreadable, transcribe what you \
can and set confidence to "low".
    * header_all_caps: for the header words "GOVERNMENT WARNING", are they in ALL CAPITAL LETTERS? \
true / false / null if not determinable.
    * header_bold: report true ONLY if the header's printed strokes are clearly heavier / thicker \
than the warning body text immediately after it; false if they look the same weight or lighter; \
null if image quality or text size prevents a clear judgment. Do NOT infer bold from ALL CAPS, \
darkness, image contrast, or because government warnings are usually bold.
    * header_bold_confidence: "high" / "medium" / "low" for the header_bold observation; if styling \
is uncertain, prefer null/low over guessing.
    * header_bold_basis: one short phrase describing what you ACTUALLY saw, or null.
    * body_bold: report true ONLY if the remainder/body of the warning itself appears bold/heavy; \
false ONLY if the body appears normal / non-bold; null if image quality or text size prevents a \
clear style judgment. Judge the body's OWN stroke weight -- do not infer it from the header.
    * body_bold_confidence: "high" / "medium" / "low" for the body_bold observation; prefer null/low \
if uncertain.
    Report visible text and style observations, not compliance."""

_WARN_START = "- government_warning: the federal health warning."
_WARN_END = "    Report what you SEE; do not judge compliance."


def _build_minimal_prompt():
    """Derive B's prompt: production _PROMPT with ONLY the government_warning block swapped out, so
    all other field instructions + the JSON example stay byte-identical. Fail loudly if the anchors
    ever move (so the benchmark can't silently run the wrong prompt)."""
    i = _PROMPT.find(_WARN_START)
    j = _PROMPT.find(_WARN_END)
    if i < 0 or j < 0:
        sys.exit("ERROR: could not locate the government_warning block in the production _PROMPT "
                 "(anchors moved) -- refusing to run with a wrong prompt.")
    j += len(_WARN_END)
    return _PROMPT[:i] + _MINIMAL_WARN + _PROMPT[j:]


_MINIMAL_PROMPT_B = _build_minimal_prompt()


def extract_B(images, model, media_type="image/png"):
    """Strategy B: identical to extract_fields() except the prompt. Same production schema, same
    _parse_response/_coerce -> returns a production-shaped extraction dict."""
    content = _build_content(images, media_type, _MINIMAL_PROMPT_B)
    params = _model_params(model)   # default response_format = production _STRUCTURED_RF
    return _parse_response(_create_with_fallbacks(_get_client(), content, params))


# --------------------------------------------------------------------------- scoring
_FIELDS = ["brand_name", "class_type", "alcohol_content", "net_contents", "name_and_address",
           "country_of_origin", "appellation", "vintage"]


def _wording(text):
    if not text:
        return (False, 0.0)
    bn = _normalize(_warning_body(text))
    return (bn == _CANONICAL_WARNING_BODY_NORM, round(fuzz.ratio(bn, _CANONICAL_WARNING_BODY_NORM), 1))


def _caps_read(text, flag):
    m = re.search(r"government\s+warning", text or "", re.IGNORECASE)
    if m:
        return text[m.start():m.end()].isupper()
    return flag if isinstance(flag, bool) else None


def score(ex, secs):
    """One comparable record from a PRODUCTION-shaped extraction dict (A or B alike)."""
    gw = ex.get("government_warning") or {}
    fr = _check_warning(gw)
    exact, sim = _wording(gw.get("text"))
    return {
        "verdict": fr.status, "reason": fr.reason[:80],
        "wording_exact": exact, "wording_sim": sim,
        "caps_read": _caps_read(gw.get("text"), gw.get("header_all_caps")),
        "header_bold": gw.get("header_bold"), "body_bold": gw.get("body_bold"),
        "hb_conf": gw.get("header_bold_confidence"), "bb_conf": gw.get("body_bold_confidence"),
        "latency": round(secs, 2),
        "fields": {f: ((ex.get(f) or {}).get("value") if f != "alcohol_content"
                       else (ex.get("alcohol_content") or {}).get("abv_percent")) for f in _FIELDS},
    }


def _norm_txt(s):
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _field_agree(av, bv, abv=False):
    if abv:
        return (av is None and bv is None) or (av is not None and bv is not None and abs(av - bv) <= 0.1)
    a, b = _norm_txt(av), _norm_txt(bv)
    if not a and not b:
        return True
    if not a or not b:
        return False
    return fuzz.ratio(a, b) >= 85


# --------------------------------------------------------------------------- ground truth
def _adv_gt():
    return {
        "01_compliant": dict(compliant=True, bad=None, wording_good=True, caps_good=True, hb=True, bb=False),
        "02_titlecase": dict(compliant=False, bad="caps", wording_good=True, caps_good=False, hb=True, bb=False),
        "03_notbold": dict(compliant=False, bad="notbold", wording_good=True, caps_good=True, hb=False, bb=False),
        "04_reworded": dict(compliant=False, bad="wording", wording_good=False, caps_good=True, hb=True, bb=False),
        "Correct_Goverment_Warning": dict(compliant=True, bad=None, wording_good=True, caps_good=True, hb=True, bb=False),
    }


def _boldsafety_gt():
    man = os.path.join(ROOT, "bold_safety", "manifest.json")
    if not os.path.exists(man):
        return {}
    out = {}
    for fn, m in json.load(open(man, encoding="utf-8")).items():
        v = m.get("variant")
        bad = {"bold_compliant": None, "notbold": "notbold", "titlecase": "caps", "boldbody": "boldbody"}.get(v, v)
        out[fn] = dict(compliant=(v == "bold_compliant"), bad=bad, wording_good=True,
                       caps_good=(v != "titlecase"), hb=m.get("header_bold_font"), bb=m.get("body_bold_font"))
    return out


def _products():
    adv_gt, bs_gt = _adv_gt(), _boldsafety_gt()
    items = []
    for f in _gather([os.path.join(ROOT, "adversarial")]):
        lab = os.path.splitext(os.path.basename(f))[0]
        items.append((lab, "adversarial", [(open(f, "rb").read(), _media_type(f))],
                      adv_gt.get(lab, dict(compliant=True, bad=None, wording_good=True, caps_good=True, hb=None, bb=None))))
    for sub in ("baseline_labels", "real_labels"):
        files = _gather([os.path.join(ROOT, "test_labels", sub)])
        for key, group in sorted(_group_by_product(files).items()):
            imgs = [(open(f, "rb").read(), _media_type(f)) for f in group]
            items.append((key, sub, imgs, dict(compliant=True, bad=None, wording_good=True, caps_good=True, hb=None, bb=None)))
    if os.environ.get("MINIMAL_PROMPT_SKIP_BOLD_SAFETY") != "1":
        for f in _gather([os.path.join(ROOT, "bold_safety")]):
            fn = os.path.basename(f)
            if fn in bs_gt:
                items.append((os.path.splitext(fn)[0], "bold_safety",
                              [(open(f, "rb").read(), _media_type(f))], bs_gt[fn]))
    return items


def _with_retry(fn, *a, tries=4):
    last = None
    for k in range(tries):
        try:
            return fn(*a)
        except Exception as e:   # transient (rate-limit 429 / timeout): patient backoff, then retry
            last = e
            time.sleep(4 + 6 * k)
    raise last


# --------------------------------------------------------------------------- run
def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
    model = (os.environ.get("MINIMAL_PROMPT_MODEL") or EXTRACTION_MODEL).strip()
    if not _load_key():
        sys.exit("ERROR: no OpenAI key (OPENAI_API_KEY env or .streamlit/secrets.toml).")
    products = _products()
    print(f"minimal_schema_prompt_experiment: {len(products)} products, runs={runs}, model={model}\n")

    def run_A(imgs):
        t = time.perf_counter(); ex = extract_fields(imgs); return ex, time.perf_counter() - t

    def run_B(imgs):
        t = time.perf_counter(); ex = extract_B(imgs, model); return ex, time.perf_counter() - t

    jobs = []
    for (lab, st, imgs, gt) in products:
        nr = 1 if st == "bold_safety" else runs   # bold_safety is a SECONDARY single-read signal
        for r in range(nr):
            jobs.append((lab, st, gt, "A", r, imgs))
            jobs.append((lab, st, gt, "B", r, imgs))

    def execute(job):
        lab, st, gt, strat, r, imgs = job
        try:
            ex, secs = _with_retry(run_A if strat == "A" else run_B, imgs)
            return (lab, st, gt, strat, r, score(ex, secs), None)
        except Exception as e:
            return (lab, st, gt, strat, r, None, str(e)[:160])

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(execute, jobs))
    wall = time.perf_counter() - t0

    data, errors = {}, []
    for (lab, st, gt, strat, r, rec, err) in results:
        d = data.setdefault(lab, {"set": st, "gt": gt, "A": [], "B": []})
        if err:
            errors.append({"label": lab, "strat": strat, "run": r, "error": err})
        else:
            d[strat].append(rec)

    # ---------- metrics, SEPARATED: main label task vs bold_safety stress test ----------
    MAIN_SETS = {"adversarial", "baseline_labels", "real_labels"}

    def in_group(d, group):
        return (d["set"] in MAIN_SETS) if group == "main" else (d["set"] == "bold_safety")

    def first(recs):
        return recs[0] if recs else None

    def pct(xy):
        x, n = xy
        return f"{x}/{n} ({100*x/n:.0f}%)" if n else "n/a"

    def agg(strat, group):
        M = {k: [0, 0] for k in ("ff_compliant", "fp_wording", "fp_caps", "fp_notbold",
                                 "fp_boldbody", "review", "wording", "caps")}
        lat, confwrong = [], []
        for lab, d in data.items():
            if not in_group(d, group):
                continue
            gt, rec = d["gt"], first(d[strat])
            if not rec:
                continue
            lat.append(rec["latency"]); v = rec["verdict"]
            M["review"][1] += 1; M["review"][0] += (v == REVIEW)
            if gt.get("compliant") is True:
                M["ff_compliant"][1] += 1; M["ff_compliant"][0] += (v == FAIL)
            bad = gt.get("bad")
            if bad == "wording":
                M["fp_wording"][1] += 1; M["fp_wording"][0] += (v == PASS)
            if bad == "caps":
                M["fp_caps"][1] += 1; M["fp_caps"][0] += (v == PASS)
            if bad == "notbold":
                M["fp_notbold"][1] += 1; M["fp_notbold"][0] += (v == PASS)
            if bad == "boldbody":
                M["fp_boldbody"][1] += 1; M["fp_boldbody"][0] += (v == PASS)
            if gt.get("wording_good") is not None:
                M["wording"][1] += 1; M["wording"][0] += (rec["wording_exact"] == bool(gt["wording_good"]))
            if gt.get("caps_good") is not None and rec["caps_read"] is not None:
                M["caps"][1] += 1; M["caps"][0] += (rec["caps_read"] == bool(gt["caps_good"]))
            # confident-wrong: high-confidence bold read that contradicts known GT, or a reworded
            # warning silently "normalized" to the canonical text (wording_exact True on bad=wording).
            if gt.get("hb") is not None and rec["hb_conf"] == "high" and rec["header_bold"] is not None \
                    and rec["header_bold"] != gt["hb"]:
                confwrong.append({"label": lab, "kind": "bold", "claim": f"header_bold={rec['header_bold']} (high)",
                                  "gt": f"header_bold={gt['hb']}", "verdict": v})
            if bad == "wording" and rec["wording_exact"]:
                confwrong.append({"label": lab, "kind": "normalized_reworded",
                                  "claim": "transcribed warning == canonical", "gt": "warning is reworded", "verdict": v})
        return M, lat, confwrong

    def stability(strat, group):
        fv = fb = tot = 0
        for lab, d in data.items():
            if not in_group(d, group):
                continue
            recs = d[strat]
            if len(recs) < 2:
                continue
            tot += 1
            fv += (len({r["verdict"] for r in recs}) > 1)
            fb += (len({r["header_bold"] for r in recs}) > 1)
        return dict(products=tot, verdict_flip=fv, bold_flip=fb)

    def lat_stats(L):
        return dict(avg=round(sum(L) / len(L), 2) if L else None, max=round(max(L), 2) if L else None, n=len(L))

    def pack(strat, group):
        M, lat, cw = agg(strat, group)
        return {"false_fail_compliant": M["ff_compliant"], "false_pass_wording": M["fp_wording"],
                "false_pass_caps": M["fp_caps"], "false_pass_notbold": M["fp_notbold"],
                "false_pass_boldbody": M["fp_boldbody"], "review_rate": M["review"],
                "wording_accuracy": M["wording"], "caps_accuracy": M["caps"],
                "latency": lat_stats(lat), "stability": stability(strat, group), "confident_wrong": cw}

    # mandatory-field agreement B vs A (run #0) -- MAIN label task only (real fields)
    fa = {f: [0, 0] for f in _FIELDS}
    for lab, d in data.items():
        if d["set"] not in MAIN_SETS:
            continue
        a, b = first(d["A"]), first(d["B"])
        if not a or not b:
            continue
        for f in _FIELDS:
            fa[f][1] += 1
            fa[f][0] += _field_agree(a["fields"].get(f), b["fields"].get(f), abv=(f == "alcohol_content"))

    metrics = {"model": model, "runs": runs, "n_products": len(data), "wall_seconds": round(wall, 1),
               "errors": len(errors),
               "main": {"A": pack("A", "main"), "B": pack("B", "main")},
               "bold_safety": {"A": pack("A", "bold_safety"), "B": pack("B", "bold_safety")},
               "mandatory_field_agreement_BvsA_main": fa}

    out = {"note": "BENCHMARK-ONLY; production _PROMPT/schema/_coerce/verifier/policy unchanged. B = "
                   "production prompt with ONLY the government_warning block reworded; same schema, "
                   "same _coerce, same verifier, same model, same images. Metrics SEPARATED: "
                   "'main' = adversarial+baseline+real_labels; 'bold_safety' = secondary stress test.",
           "prompt_B_government_warning_block": _MINIMAL_WARN,
           "metrics": metrics, "errors": errors,
           "per_product": [{"label": l, "set": d["set"], "gt": d["gt"], "A": d["A"], "B": d["B"]}
                           for l, d in sorted(data.items())]}
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "artifacts", "minimal_schema_prompt_experiment_results.json"),
                        "w", encoding="utf-8"), indent=2, default=str)

    rows_main = [("wording accuracy", "wording_accuracy"), ("caps accuracy", "caps_accuracy"),
                 ("false-FAIL compliant", "false_fail_compliant"),
                 ("false-PASS reworded wording", "false_pass_wording"),
                 ("false-PASS title-case caps", "false_pass_caps"),
                 ("false-PASS not-bold header", "false_pass_notbold"),
                 ("review rate", "review_rate")]
    rows_bs = [("wording accuracy", "wording_accuracy"), ("caps accuracy", "caps_accuracy"),
               ("false-FAIL compliant-bold", "false_fail_compliant"),
               ("false-PASS not-bold header", "false_pass_notbold"),
               ("false-PASS all-bold-body", "false_pass_boldbody"),
               ("false-PASS title-case caps", "false_pass_caps"),
               ("review rate", "review_rate")]

    def md_table(title, grp, rows):
        A, B = metrics[grp]["A"], metrics[grp]["B"]
        out = [f"### {title}", "", "| metric | A production | B reworded-warning |", "|---|---|---|"]
        for name, key in rows:
            out.append(f"| {name} | {pct(A[key])} | {pct(B[key])} |")
        out += [f"| latency avg / max (s) | {A['latency']['avg']} / {A['latency']['max']} | {B['latency']['avg']} / {B['latency']['max']} |",
                f"| verdict flips across runs | {A['stability']['verdict_flip']}/{A['stability']['products']} | {B['stability']['verdict_flip']}/{B['stability']['products']} |",
                f"| confident-wrong reads | {len(A['confident_wrong'])} | {len(B['confident_wrong'])} |", ""]
        return out

    L = ["# Minimal schema-compatible prompt — production (A) vs reworded-warning prompt (B)  (benchmark-only)", "",
         f"Model `{model}`, {runs} runs/strategy, {metrics['n_products']} products, wall "
         f"{metrics['wall_seconds']}s, {metrics['errors']} call errors. **B is a true drop-in: same "
         f"production schema + `_coerce` + verifier + model + images; ONLY the government_warning "
         f"instruction wording differs** (derived from the real `_PROMPT`). Both scored through the same "
         f"unchanged `verification._check_warning`. Results SEPARATED into the main label task and the "
         f"secondary bold stress test.", ""]
    L += md_table("Main label task (adversarial + baseline + real_labels)", "main", rows_main)
    L += md_table("Secondary bold stress test (bold_safety)", "bold_safety", rows_bs)
    L += ["### Mandatory-field agreement (B vs A, main label task, run #0)"]
    for f in _FIELDS:
        L.append(f"- `{f}`: {pct(metrics['mandatory_field_agreement_BvsA_main'][f])}")
    L += ["", "### Confident-wrong examples (main ∥ stress)"]
    allcw = [("main", e) for e in metrics["main"]["A"]["confident_wrong"] + metrics["main"]["B"]["confident_wrong"]]
    allcw += [("stress", e) for e in metrics["bold_safety"]["A"]["confident_wrong"] + metrics["bold_safety"]["B"]["confident_wrong"]]
    for grp, e in allcw[:12]:
        L.append(f"- [{grp}] {e['label']} [{e['kind']}]: {e['claim']} vs {e['gt']} → {e['verdict']}")
    open(os.path.join(ROOT, "artifacts", "minimal_schema_prompt_experiment_results.md"), "w", encoding="utf-8").write("\n".join(L))

    def cprint(title, grp, rows):
        A, B = metrics[grp]["A"], metrics[grp]["B"]
        print(f"\n--- {title} ---")
        print(f"{'metric':<32}{'A production':<16}{'B reworded'}")
        print("-" * 62)
        for name, key in rows:
            print(f"{name:<32}{pct(A[key]):<16}{pct(B[key])}")
        print(f"{'latency avg/max':<32}{str(A['latency']['avg'])+'/'+str(A['latency']['max']):<16}{str(B['latency']['avg'])+'/'+str(B['latency']['max'])}")
        print(f"{'verdict flips/runs':<32}{str(A['stability']['verdict_flip'])+'/'+str(A['stability']['products']):<16}{str(B['stability']['verdict_flip'])+'/'+str(B['stability']['products'])}")
        print(f"{'confident-wrong reads':<32}{len(A['confident_wrong']):<16}{len(B['confident_wrong'])}")

    print("=" * 80)
    print(f"MINIMAL SCHEMA PROMPT EXPERIMENT  (model={model}, {runs} runs)  wall={metrics['wall_seconds']}s errors={metrics['errors']}")
    cprint("MAIN label task (adversarial+baseline+real)", "main", rows_main)
    cprint("SECONDARY bold stress test (bold_safety)", "bold_safety", rows_bs)
    print(f"\nartifacts/minimal_schema_prompt_experiment_results.md / .json")


if __name__ == "__main__":
    main()
