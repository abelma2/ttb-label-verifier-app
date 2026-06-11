"""Benchmark-only: how many model reads per label (k = 1..N) are worth it?

Runs several independent READERS once-or-twice per product, stores each reader's raw warning read,
verdict (via the UNCHANGED verification._check_warning), latency, tokens, and errors -- then SIMULATES
ensemble sizes k=1..N and four combination policies from the collected data (no k* re-runs). Readers:

  A  production_current  -- extraction.extract_fields() exactly as production (gpt-5.4-mini)
  B  minimal_schema      -- the production-schema minimal-prompt variant (gpt-5.4-mini), reused from
                            minimal_schema_prompt_experiment._MINIMAL_PROMPT_B (same _coerce, verifier)
  D  higher_accuracy     -- production prompt+schema on a higher-accuracy model (ENSEMBLE_MODEL_D, default
                            gpt-5.5), via env override -- config.py NOT changed; PROBED, skipped if absent
  (A former reader E used Gemini; it was removed with the Google teardown -- this project
  does not use Google services.)

Policies: single_best (k=1 baseline), majority_vote (ties -> REVIEW), conservative_pass (PASS only if
ALL pass; FAIL on a deterministic wording/caps violation; else REVIEW), reviewer_triage (FAIL on clear
text/caps violation; PASS only if all readers PASS with high-confidence clean bold; else REVIEW).

Results are SEPARATED: main label task (adversarial + baseline + real_labels) is the primary
conclusion; bold_safety is a SECONDARY safety-critical stress test only.

BENCHMARK ONLY. Does not touch app.py / verification.py / config.py / the production prompt, does not
change WARNING_BOLD_POLICY, is not wired into Streamlit, and does not overwrite other artifacts.

Run:  python scripts/benchmarks/ensemble_size_experiment.py [repeats_main]
      ENSEMBLE_MODEL_D=gpt-5.5 python scripts/benchmarks/ensemble_size_experiment.py 2
Outputs: artifacts/ensemble_size_experiment_results.md / .json
"""
import json
import os
import re
import sys
import time
from collections import Counter
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
from extraction import (_PROMPT, _build_content, _model_params, _get_client, _create_with_fallbacks,
                        _coerce)
from verification import (_check_warning, PASS, REVIEW, FAIL, _normalize, _warning_body,
                          _CANONICAL_WARNING_BODY_NORM)
from minimal_schema_prompt_experiment import _MINIMAL_PROMPT_B

MODEL_A = EXTRACTION_MODEL                       # gpt-5.4-mini (A/B)
MODEL_D = os.environ.get("ENSEMBLE_MODEL_D", "gpt-5.5")

_FIELDS = ["brand_name", "class_type", "alcohol_content", "net_contents", "name_and_address",
           "country_of_origin", "appellation", "vintage"]
# rough RELATIVE price weights per model (gpt-5.4-mini = 1.0) -- a stated assumption for the cost
# multiplier; absolute $ needs live prices.
PRICE_W = {"gpt-5.4-mini": 1.0, "gpt-5.5": 8.0, "gpt-5": 6.0}
RETRIES = []   # appended to on each retry (thread-safe enough for counting)

# --------------------------------------------------------------------------- readers
def _fields_of(ex):
    return {f: ((ex.get(f) or {}).get("value") if f != "alcohol_content"
                else (ex.get("alcohol_content") or {}).get("abv_percent")) for f in _FIELDS}


def _openai_full(images, prompt, model, mt="image/png"):
    content = _build_content(images, mt, prompt)
    params = _model_params(model)
    t = time.perf_counter()
    resp = _create_with_fallbacks(_get_client(), content, params)
    secs = time.perf_counter() - t
    ex = _coerce(json.loads(resp.choices[0].message.content))
    tok = getattr(getattr(resp, "usage", None), "total_tokens", None)
    return {"gw": ex["government_warning"], "fields": _fields_of(ex), "model": model, "scope": "full",
            "latency": secs, "tokens": tok}


def _with_retry(fn, *a, tries=4):
    last = None
    for k in range(tries):
        try:
            return fn(*a)
        except Exception as e:
            last = e
            if k:
                RETRIES.append(1)
            time.sleep(4 + 6 * k)
    raise last


# --------------------------------------------------------------------------- per-reader state
def reader_state(rec):
    """Derive comparable warning state from one reader record's government_warning."""
    gw = rec["gw"]
    fr = _check_warning(gw)
    text = gw.get("text")
    bn = _normalize(_warning_body(text)) if text else ""
    sim = round(fuzz.ratio(bn, _CANONICAL_WARNING_BODY_NORM), 1) if text else 0.0
    exact = bn == _CANONICAL_WARNING_BODY_NORM
    wstate = "match" if exact else ("near" if sim >= 90 else "miss")
    m = re.search(r"government\s+warning", text or "", re.IGNORECASE)
    caps = text[m.start():m.end()].isupper() if m else gw.get("header_all_caps")
    cstate = caps if isinstance(caps, bool) else None
    hb, hbc = gw.get("header_bold"), gw.get("header_bold_confidence")
    bb, bbc = gw.get("body_bold"), gw.get("body_bold_confidence")
    if (hb is False and hbc == "high") or (bb is True and bbc == "high"):
        bstate = "violation"
    elif hb is True and hbc == "high" and bb is False and bbc == "high":
        bstate = "clean"
    else:
        bstate = "uncertain"
    return {"verdict": fr.status, "wstate": wstate, "cstate": cstate, "bstate": bstate, "sim": sim,
            "fields": rec.get("fields"), "latency": rec["latency"], "tokens": rec["tokens"],
            "model": rec["model"]}


def ensemble_verdict(policy, S):
    """Combine the first-k readers' states (list S, S[0] = reader A) into one verdict."""
    verdicts = [s["verdict"] for s in S]
    det_violation = any(s["wstate"] == "miss" or s["cstate"] is False for s in S)
    if policy == "single_best":
        return S[0]["verdict"]
    if policy == "majority_vote":
        top = Counter(verdicts).most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            return REVIEW
        return top[0][0]
    if policy == "conservative_pass":
        if det_violation:
            return FAIL
        if all(v == PASS for v in verdicts):
            return PASS
        return REVIEW
    if policy == "reviewer_triage":
        if det_violation:
            return FAIL
        if all(s["verdict"] == PASS and s["bstate"] == "clean" for s in S):
            return PASS
        return REVIEW
    raise ValueError(policy)


# --------------------------------------------------------------------------- ground truth + products
def _adv_gt():
    return {
        "01_compliant": dict(compliant=True, bad=None, wording_good=True, caps_good=True),
        "02_titlecase": dict(compliant=False, bad="caps", wording_good=True, caps_good=False),
        "03_notbold": dict(compliant=False, bad="notbold", wording_good=True, caps_good=True),
        "04_reworded": dict(compliant=False, bad="wording", wording_good=False, caps_good=True),
        "Correct_Goverment_Warning": dict(compliant=True, bad=None, wording_good=True, caps_good=True),
    }


def _bs_gt():
    man = os.path.join(ROOT, "bold_safety", "manifest.json")
    if not os.path.exists(man):
        return {}
    out = {}
    for fn, mm in json.load(open(man, encoding="utf-8")).items():
        v = mm.get("variant")
        bad = {"bold_compliant": None, "notbold": "notbold", "titlecase": "caps", "boldbody": "boldbody"}.get(v, v)
        out[fn] = dict(compliant=(v == "bold_compliant"), bad=bad, wording_good=True, caps_good=(v != "titlecase"))
    return out


def _products():
    adv, bs = _adv_gt(), _bs_gt()
    items = []
    for f in _gather([os.path.join(ROOT, "adversarial")]):
        lab = os.path.splitext(os.path.basename(f))[0]
        items.append((lab, "adversarial", [(open(f, "rb").read(), _media_type(f))],
                      adv.get(lab, dict(compliant=True, bad=None, wording_good=True, caps_good=True))))
    for sub in ("baseline_labels", "real_labels"):
        for key, group in sorted(_group_by_product(_gather([os.path.join(ROOT, "test_labels", sub)])).items()):
            imgs = [(open(f, "rb").read(), _media_type(f)) for f in group]
            items.append((key, sub, imgs, dict(compliant=True, bad=None, wording_good=True, caps_good=True)))
    for f in _gather([os.path.join(ROOT, "bold_safety")]):
        fn = os.path.basename(f)
        if fn in bs:
            items.append((os.path.splitext(fn)[0], "bold_safety",
                          [(open(f, "rb").read(), _media_type(f))], bs[fn]))
    return items


MAIN_SETS = {"adversarial", "baseline_labels", "real_labels"}


# --------------------------------------------------------------------------- run
def main():
    repeats_main = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2
    if not _load_key():
        sys.exit("ERROR: no OpenAI key (OPENAI_API_KEY env or .streamlit/secrets.toml).")
    products = _products()
    sample = products[0][2]

    # reader registry (ordered: cheap same-model first, expensive/other last)
    readers = [("A_production", lambda im: _openai_full(im, _PROMPT, MODEL_A)),
               ("B_minimal", lambda im: _openai_full(im, _MINIMAL_PROMPT_B, MODEL_A))]
    skipped = []
    # probe D (higher-accuracy model)
    try:
        _with_retry(lambda im: _openai_full(im, _PROMPT, MODEL_D), sample, tries=2)
        readers.append(("D_higher_accuracy", lambda im: _openai_full(im, _PROMPT, MODEL_D)))
    except Exception as e:
        skipped.append(("D_higher_accuracy", f"{MODEL_D} unavailable: {str(e)[:100]}"))

    reader_names = [n for n, _ in readers]
    print(f"ensemble_size_experiment: {len(products)} products, readers={reader_names}, "
          f"repeats_main={repeats_main}, skipped={[s[0] for s in skipped]}\n")

    # build jobs: each (product, reader, repeat). bold_safety = 1 repeat (secondary).
    jobs = []
    for (lab, st, imgs, gt) in products:
        nr = 1 if st == "bold_safety" else repeats_main
        for ri, (rname, rfn) in enumerate(readers):
            for rep in range(nr):
                jobs.append((lab, st, gt, ri, rname, rfn, rep, imgs))

    def execute(job):
        lab, st, gt, ri, rname, rfn, rep, imgs = job
        try:
            rec = _with_retry(rfn, imgs)
            return (lab, st, gt, ri, rname, rep, reader_state(rec), None)
        except Exception as e:
            return (lab, st, gt, ri, rname, rep, None, str(e)[:160])

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(execute, jobs))
    wall = time.perf_counter() - t0

    # data[lab] = {set, gt, reads: {reader_idx: {rep: state}}}
    data, errors = {}, []
    for (lab, st, gt, ri, rname, rep, state, err) in results:
        d = data.setdefault(lab, {"set": st, "gt": gt, "reads": {}})
        if err:
            errors.append({"label": lab, "reader": rname, "rep": rep, "error": err})
        else:
            d["reads"].setdefault(ri, {})[rep] = state

    # per-reader latency/token aggregates (repeat 0, all products that succeeded)
    reader_stats = []
    for ri, (rname, _) in enumerate(readers):
        lats, toks, errs = [], [], 0
        for lab, d in data.items():
            s = d["reads"].get(ri, {}).get(0)
            if s:
                lats.append(s["latency"])
                if s["tokens"]:
                    toks.append(s["tokens"])
        errs = sum(1 for e in errors if e["reader"] == rname)
        model = readers[ri][0]
        mdl = (next((d["reads"][ri][0]["model"] for d in data.values() if d["reads"].get(ri, {}).get(0)), MODEL_A))
        reader_stats.append({"reader": rname, "model": mdl,
                             "avg_latency": round(sum(lats) / len(lats), 2) if lats else None,
                             "max_latency": round(max(lats), 2) if lats else None,
                             "avg_tokens": round(sum(toks) / len(toks)) if toks else None, "errors": errs})

    # ----- ensemble metrics per (k, policy, group) -----
    POLICIES = ["single_best", "majority_vote", "conservative_pass", "reviewer_triage"]

    def states_k(d, k, rep):
        """ordered list of available reader states (first k readers) for a product+repeat."""
        return [d["reads"][ri][rep] for ri in range(k) if d["reads"].get(ri, {}).get(rep)]

    def group_labels(group):
        return [lab for lab, d in data.items()
                if (d["set"] in MAIN_SETS) == (group == "main")]

    def maj_true(S, pred):
        n = sum(1 for s in S if pred(s))
        return n > len(S) / 2

    def metrics(k, policy, group):
        M = {x: [0, 0] for x in ("fp_wording", "fp_caps", "fp_notbold", "fp_boldbody",
                                 "ff_compliant", "review", "wording", "caps")}
        par_lat, seq_lat, tok_sum = [], [], []
        for lab in group_labels(group):
            d = data[lab]; gt = d["gt"]
            S = states_k(d, k, 0)
            if not S:
                continue
            v = ensemble_verdict(policy, S)
            M["review"][1] += 1; M["review"][0] += (v == REVIEW)
            if gt.get("compliant") is True:
                M["ff_compliant"][1] += 1; M["ff_compliant"][0] += (v == FAIL)
            bad = gt.get("bad")
            for key, name in (("fp_wording", "wording"), ("fp_caps", "caps"),
                              ("fp_notbold", "notbold"), ("fp_boldbody", "boldbody")):
                if bad == name:
                    M[key][1] += 1; M[key][0] += (v == PASS)
            if gt.get("wording_good") is not None:
                M["wording"][1] += 1
                M["wording"][0] += (maj_true(S, lambda s: s["wstate"] == "miss") == (not gt["wording_good"]))
            if gt.get("caps_good") is not None:
                M["caps"][1] += 1
                M["caps"][0] += (maj_true(S, lambda s: s["cstate"] is False) == (not gt["caps_good"]))
            par_lat.append(max(s["latency"] for s in S))
            seq_lat.append(sum(s["latency"] for s in S))
            tok_sum.append(sum((s["tokens"] or 0) for s in S))
        # stability (main only, needs >=2 repeats)
        flips = stot = 0
        if group == "main" and repeats_main >= 2:
            for lab in group_labels(group):
                d = data[lab]
                S0, S1 = states_k(d, k, 0), states_k(d, k, 1)
                if S0 and S1:
                    stot += 1
                    flips += (ensemble_verdict(policy, S0) != ensemble_verdict(policy, S1))

        def stat(L):
            L = sorted(L)
            return {"avg": round(sum(L) / len(L), 2) if L else None, "max": round(max(L), 2) if L else None,
                    "over5s": sum(1 for x in L if x > 5), "n": len(L)} if L else {"avg": None, "max": None, "over5s": 0, "n": 0}
        return {**{x: M[x] for x in M}, "parallel_latency": stat(par_lat), "sequential_latency": stat(seq_lat),
                "avg_tokens": round(sum(tok_sum) / len(tok_sum)) if tok_sum else None,
                "stability_flips": [flips, stot]}

    # cost multiplier vs k=1 (token-weighted, stated price weights)
    def reader_cost(ri):
        rs = reader_stats[ri]
        w = PRICE_W.get(rs["model"], 1.0)
        return (rs["avg_tokens"] or 0) * w
    base_cost = reader_cost(0) or 1
    def cost_mult(k):
        return round(sum(reader_cost(ri) for ri in range(k)) / base_cost, 2)

    K = len(readers)
    grid = {}
    for group in ("main", "bold_safety"):
        for k in range(1, K + 1):
            for policy in POLICIES:
                grid[f"{group}|k{k}|{policy}"] = metrics(k, policy, group)

    # confident-wrong: a reader's read contradicts known ground truth (bold / reworded wording)
    confident_wrong = []
    for lab, d in data.items():
        gt = d["gt"]; bad = gt.get("bad")
        for ri, reps in d["reads"].items():
            s = reps.get(0)
            if not s:
                continue
            issue = None
            if bad in ("notbold", "boldbody") and s["bstate"] == "clean":
                issue = f"read bold as CLEAN on a {bad} violation"
            elif gt.get("compliant") is True and s["bstate"] == "violation":
                issue = "read a bold VIOLATION on a compliant label"
            elif bad == "wording" and s["wstate"] in ("match", "near"):
                issue = "matched canonical wording on a REWORDED warning"
            if issue:
                confident_wrong.append({"label": lab, "set": d["set"], "reader": reader_names[ri],
                                        "issue": issue, "verdict": s["verdict"]})
    out = {
        "note": "BENCHMARK-ONLY; production prompt/schema/_coerce/verifier/policy unchanged. Readers run "
                "independently; ensemble sizes k=1..N and policies SIMULATED from collected reads. Main "
                "conclusion = adversarial+baseline+real; bold_safety is a secondary stress test.",
        "readers": reader_names, "skipped_readers": skipped, "repeats_main": repeats_main,
        "wall_seconds": round(wall, 1), "errors": len(errors), "retries": len(RETRIES),
        "reader_stats": reader_stats, "cost_multiplier_by_k": {k: cost_mult(k) for k in range(1, K + 1)},
        "price_weights_assumed": PRICE_W, "grid": grid, "error_detail": errors[:40],
        "confident_wrong": confident_wrong,
        "per_product": [{"label": l, "set": d["set"], "gt": d["gt"],
                         "reads": {reader_names[ri]: {rep: {kk: st[kk] for kk in ("verdict", "wstate", "cstate", "bstate", "sim")}
                                                      for rep, st in reps.items()}
                                   for ri, reps in d["reads"].items()}}
                        for l, d in sorted(data.items())],
    }
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "artifacts", "ensemble_size_experiment_results.json"),
                        "w", encoding="utf-8"), indent=2, default=str)

    # ---- markdown ----
    def pct(xy):
        x, n = xy
        return f"{x}/{n} ({100*x/n:.0f}%)" if n else "n/a"

    def fp_all(group, k, policy):
        m = grid[f"{group}|k{k}|{policy}"]
        x = sum(m[c][0] for c in ("fp_wording", "fp_caps", "fp_notbold", "fp_boldbody"))
        n = sum(m[c][1] for c in ("fp_wording", "fp_caps", "fp_notbold", "fp_boldbody"))
        return [x, n]

    L = ["# Ensemble-size experiment — how many model reads per label?  (benchmark-only)", "",
         f"Readers: {', '.join(reader_names)}  (skipped: {', '.join(n for n, _ in skipped) or 'none'}). "
         f"{len(data)} products, repeats_main={repeats_main}, wall {round(wall,1)}s, {len(errors)} errors, "
         f"{len(RETRIES)} retries. Ensemble sizes k=1..{K} and 4 policies SIMULATED from the collected reads; "
         f"both prompts/verdicts scored through the unchanged `verification._check_warning`. "
         f"**Main conclusion = adversarial+baseline+real_labels; bold_safety is a secondary stress test.**", "",
         "### Per-reader latency / tokens / errors", "",
         "| reader | model | avg latency | max | avg tokens | errors |", "|---|---|---|---|---|---|"]
    for rs in reader_stats:
        L.append(f"| {rs['reader']} | {rs['model']} | {rs['avg_latency']} | {rs['max_latency']} | {rs['avg_tokens']} | {rs['errors']} |")
    L += ["", f"Cost multiplier vs k=1 (token-weighted, price weights {PRICE_W}): "
          + ", ".join(f"k{k}={cost_mult(k)}x" for k in range(1, K + 1)), ""]

    for group, title in (("main", "MAIN label task (adversarial + baseline + real_labels)"),
                         ("bold_safety", "SECONDARY bold stress test (bold_safety)")):
        L += [f"### {title}", "",
              "| k | policy | wording acc | caps acc | false-pass known-bad | false-fail good | review rate | par latency avg/max | >5s | cost× |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k in range(1, K + 1):
            for policy in POLICIES:
                if policy == "single_best" and k > 1:
                    continue
                m = grid[f"{group}|k{k}|{policy}"]
                pl = m["parallel_latency"]
                L.append(f"| {k} | {policy} | {pct(m['wording'])} | {pct(m['caps'])} | {pct(fp_all(group,k,policy))} | "
                         f"{pct(m['ff_compliant'])} | {pct(m['review'])} | {pl['avg']}/{pl['max']} | {pl['over5s']} | {cost_mult(k)} |")
        L.append("")
        # stability (main only)
        if group == "main" and repeats_main >= 2:
            L += ["Verdict flips across the 2 repeats (lower = more stable):"]
            for k in range(1, K + 1):
                row = []
                for policy in POLICIES:
                    if policy == "single_best" and k > 1:
                        continue
                    fl = grid[f"main|k{k}|{policy}"]["stability_flips"]
                    row.append(f"{policy}={fl[0]}/{fl[1]}")
                L.append(f"- k={k}: " + ", ".join(row))
            L.append("")
    L += ["### Confident-wrong reads (a reader contradicts known ground truth)", "",
          f"{len(confident_wrong)} total across readers:"]
    for e in confident_wrong[:12]:
        L.append(f"- [{e['set']}] {e['label']}: {e['reader']} {e['issue']} (verdict {e['verdict']})")
    open(os.path.join(ROOT, "artifacts", "ensemble_size_experiment_results.md"), "w", encoding="utf-8").write("\n".join(L))

    # ---- console ----
    print("=" * 88)
    print(f"ENSEMBLE-SIZE EXPERIMENT  readers={reader_names}  wall={round(wall,1)}s errors={len(errors)} retries={len(RETRIES)}")
    print("cost× vs k=1:", ", ".join(f"k{k}={cost_mult(k)}" for k in range(1, K + 1)))
    for group in ("main", "bold_safety"):
        print(f"\n--- {group} ---  (k | policy | fp_known_bad | ff_good | review | par_lat avg/max | >5s)")
        for k in range(1, K + 1):
            for policy in POLICIES:
                if policy == "single_best" and k > 1:
                    continue
                m = grid[f"{group}|k{k}|{policy}"]; pl = m["parallel_latency"]
                print(f"  k{k} {policy:<17} fp={pct(fp_all(group,k,policy)):<12} ff={pct(m['ff_compliant']):<10} "
                      f"rev={pct(m['review']):<12} lat={pl['avg']}/{pl['max']} >5s={pl['over5s']}")
    print("\nartifacts/ensemble_size_experiment_results.md / .json")


if __name__ == "__main__":
    main()
