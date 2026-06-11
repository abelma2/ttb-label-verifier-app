"""Benchmark-only: the prompt-simplification hypothesis -- does a LESS bold-primed, minimal extraction prompt
reduce false positives while keeping text / caps / mandatory-field accuracy and still catching
real government-warning errors?

Compares three strategies on the same images, same model, same (unchanged) verifier:
  A = the PRODUCTION extraction prompt (extraction.extract_fields, used exactly as-is).
  B = a minimal style-feature MINIMAL prompt (_MINIMAL_PROMPT below) that does NOT prime the model toward
      bold; it asks for the warning text + visible style features (bold/italic/underline/all-caps/
      normal) "only if visually clear", with its own output schema (_MINIMAL_SCHEMA).
  C = strategy B repeated N times (default 3) on the default model to measure stability. The
      MODELS list is structured so extra models can be enabled later via MINIMAL_PROMPT_MODELS env
      (comma-separated), without code changes.

Fair A/B: B's style schema is MAPPED into the production government_warning shape and scored
through the SAME unchanged verifier (`verification._check_warning`, live WARNING_BOLD_POLICY), so
the only variable is the prompt -- not the gate. NOTHING in production is modified; this module
only *reuses* extraction.py / verification.py read-only (the same pattern as the other benchmarks).

BENCHMARK ONLY. Does not touch app.py / verification.py / config.py / the production prompt, does
not change WARNING_BOLD_POLICY, is not wired into Streamlit, and does not overwrite other artifacts.

Run:  python scripts/benchmarks/minimal_style_prompt_experiment.py [runs]
      MINIMAL_PROMPT_MODELS="gpt-5.4-mini,gpt-4.1-mini" python scripts/benchmarks/minimal_style_prompt_experiment.py 3
Outputs: artifacts/minimal_style_prompt_experiment_results.md / .json
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
from extraction import (_get_client, _model_params, _create_with_fallbacks, _build_content,
                        extract_fields)
from verification import (_check_warning, PASS, REVIEW, FAIL, _normalize, _warning_body,
                          _CANONICAL_WARNING_BODY_NORM)

# --------------------------------------------------------------------------- B: Minimal prompt
_MINIMAL_PROMPT = """You are reading alcohol beverage label images.
Extract the actual printed text and required label elements.
Do not judge legal compliance.
Transcribe text exactly as printed.
For the government warning, extract the warning text exactly and separately report visible style \
features for the warning header and warning body only if visually clear.
Possible style features include: bold, italic, underline, all caps, normal.
Do not infer styling. Do not assume warnings are usually bold.
Use null for any value you cannot read or cannot judge from the image (do not guess).
Return JSON only."""

_CONF = {"type": "string", "enum": ["high", "medium", "low"]}
_STYLE = {"type": "object", "additionalProperties": False,
          "properties": {"bold": {"type": ["boolean", "null"]}, "italic": {"type": ["boolean", "null"]},
                         "underline": {"type": ["boolean", "null"]}, "normal": {"type": ["boolean", "null"]}},
          "required": ["bold", "italic", "underline", "normal"]}
_MF = {"type": "object", "additionalProperties": False,
       "properties": {"text": {"type": ["string", "null"]}, "confidence": _CONF},
       "required": ["text", "confidence"]}
_MF_ABV = {"type": "object", "additionalProperties": False,
           "properties": {"text": {"type": ["string", "null"]}, "abv_percent": {"type": ["number", "null"]},
                          "confidence": _CONF},
           "required": ["text", "abv_percent", "confidence"]}
_MINIMAL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "beverage_type": {"type": "string", "enum": ["beer", "wine", "spirits", "unknown"]},
        "mandatory_fields": {
            "type": "object", "additionalProperties": False,
            "properties": {"brand_name": _MF, "class_type": _MF, "alcohol_content": _MF_ABV,
                           "net_contents": _MF, "name_and_address": _MF, "country_of_origin": _MF,
                           "appellation": _MF, "vintage": _MF},
            "required": ["brand_name", "class_type", "alcohol_content", "net_contents",
                         "name_and_address", "country_of_origin", "appellation", "vintage"]},
        "government_warning": {
            "type": "object", "additionalProperties": False,
            "properties": {"text": {"type": ["string", "null"]}, "header_text": {"type": ["string", "null"]},
                           "header_all_caps": {"type": ["boolean", "null"]},
                           "header_style_features": _STYLE, "body_style_features": _STYLE,
                           "style_confidence": _CONF, "style_basis": {"type": ["string", "null"]}},
            "required": ["text", "header_text", "header_all_caps", "header_style_features",
                         "body_style_features", "style_confidence", "style_basis"]},
        "additional_statements": {"type": "array", "items": {"type": "string"}},
        "image_quality_notes": {"type": ["string", "null"]},
    },
    "required": ["beverage_type", "mandatory_fields", "government_warning", "additional_statements",
                 "image_quality_notes"],
}
_MINIMAL_RF = {"type": "json_schema",
              "json_schema": {"name": "minimal_label", "strict": True, "schema": _MINIMAL_SCHEMA}}

_FIELDS = ["brand_name", "class_type", "alcohol_content", "net_contents", "name_and_address",
           "country_of_origin", "appellation", "vintage"]


def extract_minimal(images, model, media_type="image/png"):
    """Strategy B: one Minimal-prompt extraction. Returns (raw_dict, seconds)."""
    params = _model_params(model, response_format=_MINIMAL_RF)
    content = _build_content(images, media_type, _MINIMAL_PROMPT)
    t = time.perf_counter()
    resp = _create_with_fallbacks(_get_client(), content, params)
    secs = time.perf_counter() - t
    raw = json.loads(resp.choices[0].message.content)
    return raw, secs


def _bool_or_none(x):
    return x if isinstance(x, bool) else None


def _style_bold(sf):
    """Map a Minimal style-feature object to a header/body 'is it bold?' tri-state.
    Explicit normal=True (and bold not True) counts as 'not bold' -- the model committed to a
    normal-weight read. bold null with no 'normal' signal stays None -> routes to review."""
    sf = sf if isinstance(sf, dict) else {}
    b = _bool_or_none(sf.get("bold"))
    if b is True:
        return True
    if b is False:
        return False
    if sf.get("normal") is True:
        return False
    return None


def minimal_to_production_gw(pgw):
    """Map the minimal-prompt government_warning (style-feature schema) into the production
    government_warning shape, so the SAME verification._check_warning can score it. The overall
    `confidence` is set to 'high' (neutral) so _escalate doesn't double-count style uncertainty --
    the bold gate already consumes style_confidence; this isolates the prompt's effect on the gate."""
    pgw = pgw if isinstance(pgw, dict) else {}
    conf = pgw.get("style_confidence") if pgw.get("style_confidence") in ("high", "medium", "low") else "low"
    text = pgw.get("text")
    text = text if isinstance(text, str) else None
    return {
        "present": bool(text), "text": text,
        "header_all_caps": _bool_or_none(pgw.get("header_all_caps")),
        "header_bold": _style_bold(pgw.get("header_style_features")), "header_bold_confidence": conf,
        "header_bold_basis": pgw.get("style_basis") if isinstance(pgw.get("style_basis"), str) else None,
        "body_bold": _style_bold(pgw.get("body_style_features")), "body_bold_confidence": conf,
        "confidence": "high",
    }


# --------------------------------------------------------------------------- shared scoring
def _wording(text):
    """(exact_match_bool, similarity 0-100) of the warning BODY vs the canonical text."""
    if not text:
        return (False, 0.0)
    bn = _normalize(_warning_body(text))
    return (bn == _CANONICAL_WARNING_BODY_NORM, round(fuzz.ratio(bn, _CANONICAL_WARNING_BODY_NORM), 1))


def _caps_read(text, flag):
    """Header caps as the verifier derives it: from the literal header in `text` if present, else
    the model's boolean. Returns True/False/None."""
    m = re.search(r"government\s+warning", text or "", re.IGNORECASE)
    if m:
        return text[m.start():m.end()].isupper()
    return _bool_or_none(flag)


def _read_record(gw, latency):
    """Compact, comparable record from a production-shaped government_warning dict."""
    fr = _check_warning(gw)
    exact, sim = _wording(gw.get("text"))
    return {
        "verdict": fr.status, "reason": fr.reason[:80],
        "wording_exact": exact, "wording_sim": sim,
        "caps_read": _caps_read(gw.get("text"), gw.get("header_all_caps")),
        "header_bold": gw.get("header_bold"), "body_bold": gw.get("body_bold"),
        "style_conf": gw.get("header_bold_confidence"),
        "latency": round(latency, 2),
    }


def _norm_txt(s):
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _field_agree(a_val, b_text):
    """Do production field value `a_val` and Minimal field text `b_text` agree?"""
    a, b = _norm_txt(a_val), _norm_txt(b_text)
    if not a and not b:
        return True
    if not a or not b:
        return False
    return fuzz.ratio(a, b) >= 85


# --------------------------------------------------------------------------- ground truth
def _adv_gt():
    return {
        "01_compliant": dict(kind="adv", compliant=True, wording_good=True, caps_good=True, hb=True, bb=False),
        "02_titlecase": dict(kind="adv", compliant=False, bad="caps", wording_good=True, caps_good=False, hb=True, bb=False),
        "03_notbold": dict(kind="adv", compliant=False, bad="bold", wording_good=True, caps_good=True, hb=False, bb=False),
        "04_reworded": dict(kind="adv", compliant=False, bad="wording", wording_good=False, caps_good=True, hb=True, bb=False),
        "Correct_Goverment_Warning": dict(kind="adv", compliant=True, wording_good=True, caps_good=True, hb=True, bb=False),
    }


def _boldsafety_gt():
    man = os.path.join(ROOT, "bold_safety", "manifest.json")
    if not os.path.exists(man):
        return {}
    out = {}
    for fn, m in json.load(open(man, encoding="utf-8")).items():
        v = m.get("variant")
        out[fn] = dict(kind="boldsafety", compliant=(v == "bold_compliant"),
                       bad=(None if v == "bold_compliant" else v),
                       wording_good=True, caps_good=(v != "titlecase"),
                       hb=m.get("header_bold_font"), bb=m.get("body_bold_font"))
    return out


# --------------------------------------------------------------------------- product collection
def _products():
    """Returns a list of (label, set, [(bytes, mime), ...], gt) tuples."""
    adv_gt, bs_gt = _adv_gt(), _boldsafety_gt()
    items = []
    # adversarial: each png is its own product
    for f in _gather([os.path.join(ROOT, "adversarial")]):
        lab = os.path.splitext(os.path.basename(f))[0]
        items.append((lab, "adversarial", [(open(f, "rb").read(), _media_type(f))],
                      adv_gt.get(lab, dict(kind="adv", compliant=True, wording_good=True, caps_good=True, hb=None, bb=None))))
    # baseline + real: group front/back into one product
    for sub in ("baseline_labels", "real_labels"):
        files = _gather([os.path.join(ROOT, "test_labels", sub)])
        for key, group in sorted(_group_by_product(files).items()):
            imgs = [(open(f, "rb").read(), _media_type(f)) for f in group]
            items.append((key, sub, imgs, dict(kind="real", compliant=True, wording_good=True, caps_good=True, hb=None, bb=None)))
    # bold_safety (secondary): each image its own product
    if os.environ.get("MINIMAL_PROMPT_SKIP_BOLD_SAFETY") != "1":
        for f in _gather([os.path.join(ROOT, "bold_safety")]):
            fn = os.path.basename(f)
            if fn in bs_gt:
                items.append((os.path.splitext(fn)[0], "bold_safety",
                              [(open(f, "rb").read(), _media_type(f))], bs_gt[fn]))
    return items


def _with_retry(fn, *a, tries=3):
    last = None
    for i in range(tries):
        try:
            return fn(*a)
        except Exception as e:  # transient (rate limit / timeout): brief backoff, then retry
            last = e
            time.sleep(2 + 3 * i)
    raise last


# --------------------------------------------------------------------------- run
def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
    models = [m.strip() for m in os.environ.get("MINIMAL_PROMPT_MODELS", EXTRACTION_MODEL).split(",") if m.strip()]
    primary = models[0]
    if not _load_key():
        sys.exit("ERROR: no OpenAI key (OPENAI_API_KEY env or .streamlit/secrets.toml).")
    products = _products()
    print(f"minimal_style_prompt_experiment: {len(products)} products, runs={runs}, models={models}\n")

    # ---- build the (label, strategy, model, run) job list; A=production, B=minimal ----
    def run_A(imgs):
        t = time.perf_counter()
        ex = extract_fields(imgs)
        return ex, time.perf_counter() - t

    def run_B(imgs, model):
        return extract_minimal(imgs, model)  # (raw, secs)

    jobs = []
    for (lab, st, imgs, gt) in products:
        nr = 1 if st == "bold_safety" else runs   # bold_safety is a SECONDARY single-read signal -- don't let it dominate cost/conclusion
        for r in range(nr):
            jobs.append((lab, st, gt, "A", primary, r, ("A", imgs)))
        for model in models:
            for r in range(nr):
                jobs.append((lab, st, gt, "B", model, r, ("B", imgs, model)))

    def execute(job):
        lab, st, gt, strat, model, r, payload = job
        try:
            if payload[0] == "A":
                ex, secs = _with_retry(run_A, payload[1])
                rec = _read_record(ex["government_warning"], secs)
                fields = {f: ((ex.get(f) or {}).get("value") if f != "alcohol_content"
                              else (ex.get("alcohol_content") or {}).get("abv_percent")) for f in _FIELDS}
            else:
                raw, secs = _with_retry(run_B, payload[1], payload[2])
                gw = minimal_to_production_gw(raw.get("government_warning"))
                rec = _read_record(gw, secs)
                mf = raw.get("mandatory_fields") or {}
                fields = {f: ((mf.get(f) or {}).get("text") if f != "alcohol_content"
                              else (mf.get("alcohol_content") or {}).get("abv_percent")) for f in _FIELDS}
            rec["fields"] = fields
            return (lab, st, gt, strat, model, r, rec, None)
        except Exception as e:
            return (lab, st, gt, strat, model, r, None, str(e)[:160])

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(execute, jobs))
    wall = time.perf_counter() - t0

    # ---- collect per product/strategy ----
    data = {}  # (lab) -> {set, gt, "A":[recs], "B":{model:[recs]}, "A_fields0":..., ...}
    errors = []
    for (lab, st, gt, strat, model, r, rec, err) in results:
        d = data.setdefault(lab, {"set": st, "gt": gt, "A": [], "B": {}})
        if err:
            errors.append({"label": lab, "strat": strat, "model": model, "run": r, "error": err})
            continue
        if strat == "A":
            d["A"].append(rec)
        else:
            d["B"].setdefault(model, []).append(rec)

    # ---- metrics (head-to-head uses run #0; stability uses all runs) ----
    def first(recs):
        return recs[0] if recs else None

    def pct(x, n):
        return f"{x}/{n} ({100*x/n:.0f}%)" if n else f"{x}/0 (n/a)"

    def agg(strat_recs_getter):
        """strat_recs_getter(d) -> list[rec] for the chosen strategy (B uses primary model)."""
        m = dict(false_fail=[0, 0], false_pass=[0, 0], review=[0, 0],
                 wording=[0, 0], caps=[0, 0], bold_fp=[0, 0], bold_committed=[0, 0],
                 lat=[], conf_wrong=[])
        field_agree = {f: [0, 0] for f in _FIELDS}
        for lab, d in data.items():
            gt = d["gt"]
            recs = strat_recs_getter(d)
            rec = first(recs)
            if not rec:
                continue
            m["lat"].append(rec["latency"])
            v = rec["verdict"]
            m["review"][1] += 1; m["review"][0] += (v == REVIEW)
            if gt.get("compliant") is True:
                m["false_fail"][1] += 1; m["false_fail"][0] += (v == FAIL)
            if gt.get("compliant") is False:
                m["false_pass"][1] += 1; m["false_pass"][0] += (v == PASS)
            if gt.get("wording_good") is not None:
                ok = rec["wording_exact"] == bool(gt["wording_good"])
                m["wording"][1] += 1; m["wording"][0] += ok
            if gt.get("caps_good") is not None and rec["caps_read"] is not None:
                m["caps"][1] += 1; m["caps"][0] += (rec["caps_read"] == bool(gt["caps_good"]))
            # bold false-positive: committed header_bold=True when GT says NOT bold
            if gt.get("hb") is not None:
                if rec["header_bold"] is not None:
                    m["bold_committed"][1] += 1; m["bold_committed"][0] += 1
                else:
                    m["bold_committed"][1] += 1
                if gt["hb"] is False:
                    m["bold_fp"][1] += 1; m["bold_fp"][0] += (rec["header_bold"] is True)
            # confident-wrong: high style-confidence but the bold read contradicts GT
            if gt.get("hb") is not None and rec["style_conf"] == "high" and rec["header_bold"] is not None \
                    and rec["header_bold"] != gt["hb"]:
                m["conf_wrong"].append({"label": lab, "claim": f"header_bold={rec['header_bold']}",
                                        "gt": f"header_bold={gt['hb']}", "verdict": v})
        return m, field_agree

    A_m, _ = agg(lambda d: d["A"])
    B_m, _ = agg(lambda d: d["B"].get(primary, []))

    # mandatory-field agreement: B(primary, run0) vs A(run0)
    fa = {f: [0, 0] for f in _FIELDS}
    for lab, d in data.items():
        a, b = first(d["A"]), first(d["B"].get(primary, []))
        if not a or not b:
            continue
        for f in _FIELDS:
            av, bv = a["fields"].get(f), b["fields"].get(f)
            fa[f][1] += 1
            if f == "alcohol_content":
                agree = (av is None and bv is None) or (av is not None and bv is not None and abs(av - bv) <= 0.1)
            else:
                agree = _field_agree(av, bv)
            fa[f][0] += agree

    # stability across runs (per strategy): fraction of products whose verdict is NOT constant
    def stability(getter):
        flip_v = flip_b = tot = 0
        for lab, d in data.items():
            recs = getter(d)
            if len(recs) < 2:
                continue
            tot += 1
            flip_v += (len({r["verdict"] for r in recs}) > 1)
            flip_b += (len({r["header_bold"] for r in recs}) > 1)
        return dict(products=tot, verdict_flip=flip_v, bold_flip=flip_b)

    A_stab = stability(lambda d: d["A"])
    B_stab = stability(lambda d: d["B"].get(primary, []))

    def lat(m):
        L = m["lat"]
        return dict(avg=round(sum(L)/len(L), 2) if L else None, max=round(max(L), 2) if L else None, n=len(L))

    metrics = {
        "model_primary": primary, "models": models, "runs": runs,
        "n_products": len(data), "wall_seconds": round(wall, 1), "errors": len(errors),
        "A_production": {
            "false_fail_known_good": A_m["false_fail"], "false_pass_known_bad": A_m["false_pass"],
            "review_rate": A_m["review"], "wording_accuracy": A_m["wording"], "caps_accuracy": A_m["caps"],
            "bold_false_positive": A_m["bold_fp"], "bold_committed": A_m["bold_committed"],
            "latency": lat(A_m), "stability": A_stab, "confident_wrong": A_m["conf_wrong"]},
        "B_minimal": {
            "false_fail_known_good": B_m["false_fail"], "false_pass_known_bad": B_m["false_pass"],
            "review_rate": B_m["review"], "wording_accuracy": B_m["wording"], "caps_accuracy": B_m["caps"],
            "bold_false_positive": B_m["bold_fp"], "bold_committed": B_m["bold_committed"],
            "latency": lat(B_m), "stability": B_stab, "confident_wrong": B_m["conf_wrong"]},
        "mandatory_field_agreement_BvsA": {f: fa[f] for f in _FIELDS},
    }

    per_product = []
    for lab, d in sorted(data.items()):
        per_product.append({"label": lab, "set": d["set"], "gt": d["gt"],
                            "A": d["A"], "B": d["B"]})

    out = {"note": "BENCHMARK-ONLY; production prompt/verifier/policy unchanged. B mapped into the "
                   "production government_warning shape and scored through verification._check_warning.",
           "prompt_B": _MINIMAL_PROMPT, "metrics": metrics, "errors": errors, "per_product": per_product}

    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    jpath = os.path.join(ROOT, "artifacts", "minimal_style_prompt_experiment_results.json")
    json.dump(out, open(jpath, "w", encoding="utf-8"), indent=2, default=str)

    # ---- markdown ----
    A, B = metrics["A_production"], metrics["B_minimal"]
    L = [f"# Minimal prompt experiment — production (A) vs minimal style-feature (B)  (benchmark-only)", "",
         f"Model `{primary}`, {runs} runs/strategy, {metrics['n_products']} products "
         f"(adversarial + baseline + real + bold_safety), wall {metrics['wall_seconds']}s, "
         f"{metrics['errors']} call errors. Both prompts scored through the SAME unchanged "
         f"`verification._check_warning` (live WARNING_BOLD_POLICY); B's style schema mapped into "
         f"the production warning shape. Nothing in production changed.", "",
         "| metric (head-to-head, run #0) | A production | B Minimal |", "|---|---|---|",
         f"| false-FAIL on known-good labels | {pct(*A['false_fail_known_good'])} | {pct(*B['false_fail_known_good'])} |",
         f"| false-PASS on known-bad warnings | {pct(*A['false_pass_known_bad'])} | {pct(*B['false_pass_known_bad'])} |",
         f"| review rate (uncertain → review) | {pct(*A['review_rate'])} | {pct(*B['review_rate'])} |",
         f"| wording accuracy (match vs GT) | {pct(*A['wording_accuracy'])} | {pct(*B['wording_accuracy'])} |",
         f"| caps accuracy | {pct(*A['caps_accuracy'])} | {pct(*B['caps_accuracy'])} |",
         f"| bold false-positive (claimed bold on not-bold) | {pct(*A['bold_false_positive'])} | {pct(*B['bold_false_positive'])} |",
         f"| committed to a bold call (vs left null) | {pct(*A['bold_committed'])} | {pct(*B['bold_committed'])} |",
         f"| latency avg / max (s) | {A['latency']['avg']} / {A['latency']['max']} | {B['latency']['avg']} / {B['latency']['max']} |",
         f"| verdict flips across runs | {A['stability']['verdict_flip']}/{A['stability']['products']} | {B['stability']['verdict_flip']}/{B['stability']['products']} |",
         f"| bold-read flips across runs | {A['stability']['bold_flip']}/{A['stability']['products']} | {B['stability']['bold_flip']}/{B['stability']['products']} |", "",
         "### Mandatory-field extraction agreement (B vs A, run #0)"]
    for f in _FIELDS:
        L.append(f"- `{f}`: {pct(*metrics['mandatory_field_agreement_BvsA'][f])}")
    L += ["", "### Confident-wrong bold reads (style_confidence=high but contradicts ground truth)",
          f"- A production: {len(A['confident_wrong'])}", f"- B Minimal: {len(B['confident_wrong'])}"]
    for e in (A["confident_wrong"] + B["confident_wrong"])[:8]:
        L.append(f"  - {e['label']}: {e['claim']} vs {e['gt']} → verdict {e['verdict']}")
    open(os.path.join(ROOT, "artifacts", "minimal_style_prompt_experiment_results.md"), "w", encoding="utf-8").write("\n".join(L))

    # ---- console ----
    print("=" * 78)
    print(f"MINIMAL PROMPT EXPERIMENT  (model={primary}, {runs} runs)  wall={metrics['wall_seconds']}s errors={metrics['errors']}\n")
    print(f"{'metric':<40}{'A production':<16}{'B Minimal'}")
    print("-" * 72)
    rows = [("false-FAIL known-good", "false_fail_known_good"), ("false-PASS known-bad", "false_pass_known_bad"),
            ("review rate", "review_rate"), ("wording accuracy", "wording_accuracy"),
            ("caps accuracy", "caps_accuracy"), ("bold false-positive", "bold_false_positive"),
            ("committed to bold call", "bold_committed")]
    for name, key in rows:
        print(f"{name:<40}{pct(*A[key]):<16}{pct(*B[key])}")
    print(f"{'latency avg/max':<40}{str(A['latency']['avg'])+'/'+str(A['latency']['max']):<16}{str(B['latency']['avg'])+'/'+str(B['latency']['max'])}")
    print(f"{'verdict flips/runs':<40}{str(A['stability']['verdict_flip'])+'/'+str(A['stability']['products']):<16}{str(B['stability']['verdict_flip'])+'/'+str(B['stability']['products'])}")
    print(f"{'confident-wrong bold reads':<40}{len(A['confident_wrong']):<16}{len(B['confident_wrong'])}")
    print(f"\nartifacts/minimal_style_prompt_experiment_results.md / .json")


if __name__ == "__main__":
    main()
