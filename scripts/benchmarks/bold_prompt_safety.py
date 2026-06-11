"""Benchmark-ONLY: safest simple prompt/model for judging government-warning BOLDNESS from labels.

Find the simplest, least-priming prompt + model that gives trustworthy VISUAL evidence of
header/body stroke weight, prioritizing SAFETY (no high-confidence false-passes) over fewer
reviews. Visual evidence only -- the prompts never judge legal compliance.

Touches NO production code. extraction.py is imported READ-ONLY for the client + per-model params
(like the other benchmarks). No flags are enabled; no model voting is used; each (model, prompt)
is scored on its own.

Design:
- ONE normalized strict Structured-Outputs schema for EVERY prompt (framing varies, output shape
  does not), so scoring is uniform. Prompts F (avoids "bold") and H (200-800 weights) fill the
  relationship/weight fields; the effective header/body bold is DERIVED in Python for scoring.
- 15 prompt variants A,D,E,F,G,H,I,J,K,L,M,N,O,Q,R. P (crop_minimal) is SKIPPED -- this run crops
  nothing, so a crop prompt would mix input types; crop+VariantA is addressed from prior data.
- I (self_consistency) is ADAPTED: the array-of-5 cannot fit the single normalized schema, so it
  asks for 5 INTERNAL looks and reports the consensus with confidence = agreement (flagged below).
- Ground truth from bold_safety/manifest.json. 429 retry/backoff; resumable JSONL checkpoint.

Stages: 0 smoke (availability) | 1 scored bold_safety (3x, all prompts) | 2 winners on baselines+real.
Usage:
  python scripts/benchmarks/bold_prompt_safety.py --stage smoke
  python scripts/benchmarks/bold_prompt_safety.py --stage 1               # [--prompts A,H,..]
  python scripts/benchmarks/bold_prompt_safety.py --stage 2 --combos "gpt-5.4-mini:H,gpt-4.1-mini:E"
Writes output/warning_bold_prompt_model_comparison_<ts>.{txt,json} (+ stage2 variant).
"""
import base64
import json
import os
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

from extraction import _get_client, _model_params, _create_with_fallbacks   # read-only reuse


def _ensure_openai_key():
    """The lazy client reads OPENAI_API_KEY from the env; a bare script must load it from
    .streamlit/secrets.toml itself (same as the other benchmarks)."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    import re
    secrets = os.path.join(ROOT, ".streamlit", "secrets.toml")
    if os.path.exists(secrets):
        with open(secrets, encoding="utf-8") as fh:
            m = re.search(r'OPENAI_API_KEY\s*=\s*"([^"]+)"', fh.read())
        if m and m.group(1) and not m.group(1).startswith(("sk-...", "...")):
            os.environ["OPENAI_API_KEY"] = m.group(1)


_ensure_openai_key()

MODELS = ["gpt-5.4-mini", "gpt-4o", "gpt-4.1", "gpt-4o-mini", "gpt-4.1-mini"]
BS = os.path.join(ROOT, "bold_safety")
BASELINE = os.path.join(ROOT, "test_labels", "clearer_baseline_labels")
REAL = os.path.join(ROOT, "test_labels", "real_labels")

# Prior-production reference (gathered under the then-default header_body_gate) on THIS SAME
# bold_safety set (prior data, body-bold decider, production full prompt +
# verification._check_warning). Cited in the final report, not re-run.
PROD_REFERENCE = {
    "model_prompt": "gpt-5.4-mini | full extract + header_body_gate (prior production default)",
    "boldbody_caught": "2/9 (rubber-stamps body_bold=False/high ~7/9 -> false-pass risk)",
    "compliant_correct": "1/3 (unstable; false-failed compliant 2/3)",
    "notbold_caught": "3/3",
    "note": "reads the header rule, NOT the body rule; this is the bar to beat on bold_safety.",
}
CROP_A_REFERENCE = ("Crop + Variant A (USE_PARALLEL_WARNING_CHECK): wired & validated, kept OFF by "
                    "default (2026-06-09). CROP input, not full-image -- reported separately, not "
                    "mixed into the full-image ranking. Revisit lever = crop read confidence.")


# --- normalized output schema (identical for every prompt) -------------------
def _nenum(*vals):
    return {"type": ["string", "null"], "enum": list(vals) + [None]}


_CONF = {"type": "string", "enum": ["high", "medium", "low"]}
_NORM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["warning_present", "header_text_seen", "header_body_relationship", "header_weight",
                 "body_weight", "header_bold", "header_bold_confidence", "body_bold",
                 "body_bold_confidence", "legibility", "short_basis", "image_quality_notes"],
    "properties": {
        "warning_present": {"type": "boolean"},
        "header_text_seen": {"type": ["string", "null"]},
        "header_body_relationship": _nenum("header_heavier", "same", "body_heavier", "unclear"),
        "header_weight": _nenum("thin", "regular", "semibold_heavy", "unclear"),
        "body_weight": _nenum("thin", "regular", "semibold_heavy", "unclear"),
        "header_bold": {"type": ["boolean", "null"]},
        "header_bold_confidence": _CONF,
        "body_bold": {"type": ["boolean", "null"]},
        "body_bold_confidence": _CONF,
        "legibility": {"type": "string", "enum": ["good", "limited", "poor"]},
        "short_basis": {"type": "string"},
        "image_quality_notes": {"type": ["string", "null"]},
    },
}
_RF = {"type": "json_schema", "json_schema": {"name": "bold_visual", "strict": True, "schema": _NORM_SCHEMA}}

_PRE = ("You are a VISION assistant. Look ONLY at the U.S. government health warning on this alcohol "
        "label and report what you VISUALLY SEE about letter STROKE WEIGHTS. You are NOT judging legal "
        "compliance and there is no expected answer.\n\n")
_FIELDS = (
    "\n\nThe warning has a HEADER (the words 'GOVERNMENT WARNING') and a BODY (the sentences after it). "
    "Return EVERY field as pure visual observation:\n"
    "- warning_present; header_text_seen (or null)\n"
    "- header_body_relationship: 'header_heavier' | 'same' | 'body_heavier' | 'unclear' | null\n"
    "- header_weight / body_weight: 'thin' | 'regular' | 'semibold_heavy' | 'unclear'\n"
    "- header_bold / body_bold: true if visibly bold/heavy, false if regular/thin, null if you cannot tell\n"
    "- header_bold_confidence / body_bold_confidence: high|medium|low\n"
    "- legibility: 'good'|'limited'|'poor'; short_basis (one short phrase of what you saw); "
    "image_quality_notes (or null)")

# id -> (label, framing). Framing guides the LOOK; _FIELDS fixes the OUTPUT. P skipped.
PROMPTS = {
    "A": ("baseline_is_it_bold",
          "Is the 'GOVERNMENT WARNING' header bold? Is the warning body bold? Judge visible stroke weight."),
    "D": ("bold_legibility_gated",
          "First decide whether you can resolve the stroke thickness; if small/blurry set legibility "
          "'limited'/'poor' and lower confidence. Then judge whether the header and the body are bold."),
    "E": ("multi_property",
          "For BOTH the header and the body, observe these separately: is it bold (thick strokes), italic "
          "(slanted), or underlined? Only thick STROKES count as bold; all-caps and large size are NOT bold."),
    "F": ("relative_scale_no_bold_word",
          "Do not use the word 'bold'. Rate the RELATIVE stroke weight only: the header_body_relationship, "
          "and each region's weight (thin/regular/semibold_heavy). Leave header_bold/body_bold null if "
          "unsure -- the relationship and weights are what matter."),
    "G": ("describe_first",
          "FIRST describe in short_basis the actual stroke thickness you see for the header and the body, "
          "THEN fill the weight and bold fields to match that description -- do not decide bold before describing."),
    "H": ("weight_gap_200_800",
          "Estimate each region's font-weight like a numeric class from 200 (very thin) to 800 (very heavy): "
          "map <400 -> 'thin', ~400-500 -> 'regular', >=600 -> 'semibold_heavy'. Set header_weight and "
          "body_weight from that estimate, and header_bold/body_bold true only for semibold_heavy."),
    "I": ("self_consistency_internal",
          "Make FIVE independent silent looks at the strokes. Report the CONSENSUS in the fields below, and "
          "set confidence to 'high' only if all five looks agreed, 'medium' if most agreed, 'low' if they "
          "split. (Report the consensus values; do not output the five separately.)"),
    "J": ("relation_only",
          "Look at the government warning text only. Compare the letter strokes of the header text and the "
          "body text. Choose the relationship: header heavier, same weight, body heavier, or unclear. Also "
          "say whether the body text itself appears light/regular/heavy. Base this only on visible stroke thickness."),
    "K": ("body_first",
          "Look at the warning body sentences first. Are the body sentence strokes regular or heavy? Then "
          "compare the header strokes to the body strokes. Answer from visible stroke weight only. If the body "
          "and header look similar, say same/unclear."),
    "L": ("regular_text_anchor",
          "Use the warning body as the reference text. Decide whether the header letters are heavier than that "
          "reference, and whether the body reference text itself is regular or heavy. Ignore size, capitalization, "
          "color, and position."),
    "M": ("no_expected_answer",
          "Inspect the government warning typography. Describe the header stroke weight and the body stroke weight "
          "separately using one word each: light, regular, heavy, or unclear. Do not decide what the label should "
          "look like."),
    "N": ("same_or_different",
          "Compare the strokes in 'GOVERNMENT WARNING' to the strokes in the warning body. Are they visually "
          "different in thickness, visually the same, or unclear? Then say whether the body strokes are heavy on "
          "their own."),
    "O": ("confidence_by_visibility",
          "Look only at visible letter stroke thickness in the government warning. If the strokes are sharp enough "
          "to compare, report the header/body stroke relationship. If blur, small text, distortion, or compression "
          "makes the comparison unreliable, return unclear."),
    "Q": ("body_violation_detector",
          "Ignore the header at first. Look only at the warning body sentences. Are the body sentence strokes heavy "
          "like bold text, or regular? Then compare the header to the body. Use unclear if not visually obvious."),
    "R": ("four_bucket_visual",
          "Classify the visual weight of the warning header and warning body separately: thin, regular, "
          "semibold/heavy, or unclear. Use only the letter stroke thickness you can see."),
}
PROMPT_ORDER = ["A", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "Q", "R"]
SKIPPED = {"P": "crop_minimal -- skipped: no crop is performed in this run (would mix input types)."}


def _prompt(v):
    return _PRE + PROMPTS[v][1] + _FIELDS


# --- ground truth ------------------------------------------------------------
def _bs_images():
    with open(os.path.join(BS, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    return sorted(({"path": os.path.join(BS, n), "name": n, "variant": g["variant"],
                    "header_bold_font": g["header_bold_font"], "body_bold_font": g["body_bold_font"]}
                   for n, g in man.items()), key=lambda d: d["name"])


def _mt(p):
    return "image/jpeg" if p.lower().endswith((".jpg", ".jpeg")) else "image/png"


def _block(path):
    with open(path, "rb") as fh:
        b = fh.read()
    return {"type": "image_url",
            "image_url": {"url": f"data:{_mt(path)};base64,{base64.b64encode(b).decode()}", "detail": "high"}}


# --- one call with 429 retry/backoff (latency = the successful model call only) ----
def _call(model, prompt, image_paths, max_retries=5):
    content = [{"type": "text", "text": prompt}] + [_block(p) for p in image_paths]
    delay, retries = 2.0, 0
    for attempt in range(max_retries + 1):
        try:
            params = _model_params(model, response_format=_RF)
            t = time.perf_counter()
            resp = _create_with_fallbacks(_get_client(), content, params)
            dt = round(time.perf_counter() - t, 2)
            return json.loads(resp.choices[0].message.content), dt, retries, None
        except Exception as exc:
            msg = str(exc)
            rl = "429" in msg or "rate limit" in msg.lower()
            if rl and attempt < max_retries:
                retries += 1
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            return None, None, retries, ("RATELIMIT:" if rl else "ERROR:") + msg[:140]


# --- checkpoint (resumable JSONL) --------------------------------------------
def _ckpt_path(stage):
    return os.path.join(OUT_DIR, f"warning_bold_stage{stage}.jsonl")


def _load_ckpt(stage):
    path, done = _ckpt_path(stage), {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done[(r["model"], r["prompt"], r["image"], r["rep"])] = r
                except Exception:
                    pass
    return done


def _append_ckpt(stage, rec):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(_ckpt_path(stage), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --- effective bold derivation (uniform scoring across all prompts) ----------
def _eff_body_bold(f):
    if f.get("body_bold") is not None:
        return f["body_bold"]
    bw = f.get("body_weight")
    if bw == "semibold_heavy":
        return True
    if bw in ("thin", "regular"):
        return False
    return None


def _eff_header_bold(f):
    if f.get("header_bold") is not None:
        return f["header_bold"]
    hw, rel = f.get("header_weight"), f.get("header_body_relationship")
    if hw == "semibold_heavy" and rel == "header_heavier":
        return True
    if hw in ("thin", "regular") or rel in ("same", "body_heavier"):
        return False
    return None


# --- stage 0: smoke ----------------------------------------------------------
def stage_smoke():
    img = os.path.join(BS, "bold_compliant__clean.png")
    print("Stage 0 smoke -- one call per model:")
    avail = []
    for m in MODELS:
        fields, dt, _r, err = _call(m, _prompt("A"), [img])
        ok = fields is not None
        if ok:
            avail.append(m)
        print(f"  {m:14s} -> {'OK' if ok else 'UNAVAILABLE'}  ({dt}s)" + (f"  {err}" if err else ""))
    return avail


# --- stage 1 -----------------------------------------------------------------
def stage1(prompts, models):
    images = _bs_images()
    done = _load_ckpt(1)
    todo, counts = [], defaultdict(int)
    for m in models:
        for v in prompts:
            for im in images:
                for _ in range(3):
                    rep = counts[(m, v, im["name"])]
                    counts[(m, v, im["name"])] += 1
                    if (m, v, im["name"], rep) not in done:
                        todo.append((m, v, im, rep))
    print(f"Stage 1: {len(models)} models x {len(prompts)} prompts x {len(images)} images x 3 = "
          f"{sum(counts.values())} calls ({len(done)} cached, {len(todo)} to run)\n")

    def _job(m, v, im, rep):
        fields, dt, retries, err = _call(m, _prompt(v), [im["path"]])
        return {"model": m, "prompt": v, "image": im["name"], "variant": im["variant"],
                "rep": rep, "fields": fields, "seconds": dt, "retries": retries, "error": err}
    n = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_job, m, v, im, rep) for (m, v, im, rep) in todo]
        for fut in as_completed(futs):
            _append_ckpt(1, fut.result())
            n += 1
            if n % 50 == 0 or n == len(todo):
                print(f"  [{n}/{len(todo)}] ...", flush=True)
    return _score(_load_ckpt(1), prompts, models)


def _fp(variant, f):
    """false-pass risk: a read that would let a NON-compliant warning PASS."""
    if variant == "boldbody":
        return _eff_body_bold(f) is False
    if variant == "notbold":
        return _eff_header_bold(f) is True
    return False


def _fp_conf(variant, f):
    return f.get("body_bold_confidence") if variant == "boldbody" else f.get("header_bold_confidence")


def _correct(variant, f):
    hb, bb = _eff_header_bold(f), _eff_body_bold(f)
    if variant == "bold_compliant":
        return hb is True and bb is False
    if variant == "boldbody":
        return bb is True
    if variant == "notbold":
        return hb is False
    return None


def _review(variant, f):
    if f.get("legibility") == "poor":
        return True
    if variant == "boldbody":
        return _eff_body_bold(f) is None or f.get("body_bold_confidence") == "low"
    if variant == "notbold":
        return _eff_header_bold(f) is None or f.get("header_bold_confidence") == "low"
    if variant == "bold_compliant":
        return _eff_header_bold(f) is None or _eff_body_bold(f) is None
    return _eff_header_bold(f) is None


def _decisive(variant, f):
    return _eff_body_bold(f) if variant == "boldbody" else _eff_header_bold(f)


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def _score(done, prompts, models):
    by = defaultdict(list)
    for r in done.values():
        if r["prompt"] in prompts and r["model"] in models:
            by[(r["model"], r["prompt"])].append(r)
    summary = {}
    for (m, v), recs in by.items():
        ok = [r for r in recs if r["fields"]]
        errs = [r for r in recs if not r["fields"]]
        rl = sum(1 for r in errs if (r.get("error") or "").startswith("RATELIMIT"))
        lat = [r["seconds"] for r in ok if r["seconds"] is not None]
        cc = bbfp = nbfp = hifp = rev = medc = 0
        per_img, worst, best = defaultdict(list), None, None
        for r in ok:
            f, var = r["fields"], r["variant"]
            if var == "bold_compliant" and _correct(var, f):
                cc += 1
            if _fp(var, f):
                if var == "boldbody":
                    bbfp += 1
                else:
                    nbfp += 1
                if _fp_conf(var, f) == "high":
                    hifp += 1
                    worst = worst or {"image": r["image"], "variant": var, "read": _decisive(var, f),
                                      "conf": _fp_conf(var, f), "basis": f.get("short_basis")}
            if _review(var, f):
                rev += 1
            if _correct(var, f) and (f.get("body_bold_confidence") == "medium"
                                     or f.get("header_bold_confidence") == "medium"):
                medc += 1
            if best is None and var in ("boldbody", "notbold") and _correct(var, f) \
                    and _fp_conf(var, f) == "high":
                best = {"image": r["image"], "variant": var, "read": _decisive(var, f),
                        "conf": _fp_conf(var, f), "basis": f.get("short_basis")}
            per_img[r["image"]].append(str(_decisive(var, f)))
        stable = sum(1 for vs in per_img.values() if len(vs) >= 2 and len(set(vs)) == 1)
        summary[f"{m}|{v}"] = {
            "model": m, "prompt": v, "label": PROMPTS[v][0], "n_ok": len(ok), "n_err": len(errs),
            "ratelimit": rl, "retries": sum(r.get("retries", 0) for r in recs),
            "compliant_correct": cc, "boldbody_false_pass": bbfp, "notbold_false_pass": nbfp,
            "high_conf_false_pass": hifp, "review": rev, "medium_conf_correct": medc,
            "stable_images": stable, "n_images": len(per_img),
            "lat_avg": round(sum(lat) / len(lat), 2) if lat else None, "lat_p50": _pct(lat, 50),
            "lat_p90": _pct(lat, 90), "lat_p95": _pct(lat, 95), "lat_max": max(lat) if lat else None,
            "over_5s": sum(1 for x in lat if x > 5), "worst_example": worst, "best_example": best,
        }
    return summary


def _rank(summary):
    rows = list(summary.values())
    safe = [r for r in rows if r["high_conf_false_pass"] == 0]
    rejected = [r for r in rows if r["high_conf_false_pass"] > 0]
    safe.sort(key=lambda r: (r["boldbody_false_pass"] + r["notbold_false_pass"],
                             -r["compliant_correct"], r["review"], r["lat_p50"] or 99))
    return safe, rejected


# --- stage 2 -----------------------------------------------------------------
def _stage2_images():
    imgs = [{"path": os.path.join(BASELINE, f"{s}_{side}.png"), "name": f"{s}_{side}"}
            for s in ("clear_baseline_1", "clear_baseline_2", "clear_baseline_3") for side in ("Front", "Other")]
    imgs += [{"path": os.path.join(REAL, f"test_1_{side}.jpeg"), "name": f"test_1_{side}"} for side in ("Front", "Other")]
    return imgs


def stage2(combos):
    images, done, todo, counts = _stage2_images(), _load_ckpt(2), [], defaultdict(int)
    for (m, v) in combos:
        for im in images:
            for _ in range(3):
                rep = counts[(m, v, im["name"])]
                counts[(m, v, im["name"])] += 1
                if (m, v, im["name"], rep) not in done:
                    todo.append((m, v, im, rep))
    print(f"Stage 2: {len(combos)} combos x {len(images)} images x 3 = {sum(counts.values())} calls "
          f"({len(todo)} to run)\n")

    def _job(m, v, im, rep):
        fields, dt, retries, err = _call(m, _prompt(v), [im["path"]])
        return {"model": m, "prompt": v, "image": im["name"], "rep": rep,
                "fields": fields, "seconds": dt, "retries": retries, "error": err}
    n = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_job, m, v, im, rep) for (m, v, im, rep) in todo]
        for fut in as_completed(futs):
            _append_ckpt(2, fut.result())
            n += 1
            if n % 20 == 0 or n == len(todo):
                print(f"  [{n}/{len(todo)}] ...", flush=True)
    return _load_ckpt(2)


# --- report ------------------------------------------------------------------
def _write(summary, safe, rejected, prompts, models, stage_tag="stage1"):
    L = ["", "=" * 110, "WARNING-BOLD PROMPT/MODEL COMPARISON -- bold_safety, full-image, 3x", "=" * 110,
         f"models={models}",
         f"prompts={[f'{p}:{PROMPTS[p][0]}' for p in prompts]}",
         f"SKIPPED: {SKIPPED}",
         "false-pass risk = a bold read that would let a NON-compliant warning PASS "
         "(boldbody read body_bold=False, or notbold read header_bold=True).",
         "DECISION: reject ANY combo with a HIGH-CONFIDENCE false-pass; safety beats fewer reviews;",
         "          must do well on bold_safety VIOLATIONS, not only clean labels.", ""]
    hdr = (f"{'model|prompt':24s} {'cmpOK':6s} {'bbFP':5s} {'nbFP':5s} {'HIfp':5s} {'rev':4s} {'medOK':6s} "
           f"{'stable':7s} {'avg':5s} {'p50':5s} {'p90':5s} {'p95':5s} {'max':5s} {'>5s':4s} {'err/rl':7s}")
    L.append(hdr); L.append("-" * len(hdr))
    for r in sorted(summary.values(), key=lambda r: (r["model"], PROMPT_ORDER.index(r["prompt"]))):
        stable_str = f"{r['stable_images']}/{r['n_images']}"
        errrl = f"{r['n_err']}/{r['ratelimit']}"
        L.append(f"{r['model']+'|'+r['prompt']:24s} {str(r['compliant_correct']):6s} "
                 f"{str(r['boldbody_false_pass']):5s} {str(r['notbold_false_pass']):5s} "
                 f"{str(r['high_conf_false_pass']):5s} {str(r['review']):4s} {str(r['medium_conf_correct']):6s} "
                 f"{stable_str:7s} {str(r['lat_avg']):5s} {str(r['lat_p50']):5s} {str(r['lat_p90']):5s} "
                 f"{str(r['lat_p95']):5s} {str(r['lat_max']):5s} {str(r['over_5s']):4s} {errrl:7s}")
    L.append("")
    L.append("--- SAFE combos (no high-conf false-pass), ranked: lowest false-pass risk, most correct ---")
    for r in safe[:12]:
        ex = r.get("best_example")
        L.append(f"   {r['model']+'|'+r['prompt']:24s} ({r['label']})  fp(bb/nb)={r['boldbody_false_pass']}/"
                 f"{r['notbold_false_pass']}  cmpOK={r['compliant_correct']}  rev={r['review']}  "
                 f"p50={r['lat_p50']}s p95={r['lat_p95']}s"
                 + (f"  best: {ex['variant']} read={ex['read']}/{ex['conf']}" if ex else ""))
    L.append("")
    L.append("--- REJECTED (high-confidence false-pass > 0) -- the dangerous ones ---")
    for r in sorted(rejected, key=lambda r: -r["high_conf_false_pass"]):
        w = r.get("worst_example") or {}
        L.append(f"   {r['model']+'|'+r['prompt']:24s} HIfp={r['high_conf_false_pass']} "
                 f"(bb={r['boldbody_false_pass']}, nb={r['notbold_false_pass']})"
                 + (f"  worst: {w.get('variant')} {w.get('image')} read={w.get('read')}/{w.get('conf')} "
                    f"basis={w.get('basis')!r}" if w else ""))
    top = [(r["model"], r["prompt"]) for r in safe[:5]]
    L.append("")
    L.append(f"PRIOR-PRODUCTION REFERENCE (same set, prior data): {PROD_REFERENCE}")
    L.append(f"CROP+A: {CROP_A_REFERENCE}")
    L.append("")
    L.append(f"SUGGESTED Stage-2 combos (top {len(top)} safe): " + ", ".join(f"{m}:{v}" for m, v in top))
    text = "\n".join(L)
    print(text)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"warning_bold_prompt_model_comparison_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "safe": safe, "rejected": rejected,
                   "production_reference": PROD_REFERENCE, "crop_a_reference": CROP_A_REFERENCE,
                   "suggested_stage2": [f"{m}:{v}" for m, v in top]}, fh, indent=2, ensure_ascii=False)
    print(f"\nWritten {os.path.relpath(base, ROOT)}.txt / .json")
    return top


def _arg(args, flag, default=None):
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else default


def main():
    args = sys.argv[1:]
    os.makedirs(OUT_DIR, exist_ok=True)
    stage = _arg(args, "--stage", "1")
    if stage == "smoke":
        print("\nAvailable:", stage_smoke())
        return
    if stage == "1":
        prompts = [p for p in PROMPT_ORDER if p in (_arg(args, "--prompts") or ",".join(PROMPT_ORDER)).split(",")]
        avail = stage_smoke()
        print()
        summary = stage1(prompts, avail)
        safe, rejected = _rank(summary)
        _write(summary, safe, rejected, prompts, avail)
        return
    if stage == "2":
        combos = [(c.split(":")[0], c.split(":")[1]) for c in (_arg(args, "--combos") or "").split(",") if ":" in c]
        if not combos:
            sys.exit("--combos required, e.g. --combos gpt-5.4-mini:H,gpt-4.1-mini:E")
        stage2(combos)
        print(f"Stage 2 complete: checkpoint {_ckpt_path(2)}")
        return
    sys.exit(f"unknown --stage {stage}")


if __name__ == "__main__":
    main()
