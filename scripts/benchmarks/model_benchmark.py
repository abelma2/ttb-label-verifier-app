"""Benchmark vision models on the government-warning checks, against known ground truth.

This answers a single question: WHICH model can correctly report the signals the
government-warning rule depends on -- and in particular, can a stronger model reliably
read BOLD, the one signal we currently hand to a human?

For each (model, test case) it runs the SAME extraction prompt the app uses
(extraction._build_content / _model_params / _parse_response), captures the model's
warning observations -- verbatim `text`, `header_all_caps`, `header_bold`, `confidence`
-- runs the deterministic `_check_warning` verdict, and scores the observations against
ground truth.

Test cases (ground truth in parentheses):
  adversarial/  -- synthetic, generated with CONTROLLED fonts, so caps & bold are known:
    01_compliant  (caps=YES bold=YES wording=exact)  the clean positive
    02_titlecase  (caps=NO  bold=YES wording=exact)  can the model catch title case?
    03_notbold    (caps=YES bold=NO  wording=exact)  the BOLD discrimination test
    04_reworded   (caps=YES bold=YES wording=WRONG)  does the model hallucinate canon text?
  test_labels/baseline_labels/ -- realistic front+other pairs (caps=YES bold=YES exact)

Models: a curated OpenAI list. A model the key can't access is reported as "unavailable"
rather than crashing the run.

Key (env or .streamlit/secrets.toml): OPENAI_API_KEY.

Usage (from the project root):
    python scripts/benchmarks/model_benchmark.py                      # all default models
    python scripts/benchmarks/model_benchmark.py gpt-4o gpt-5         # only these
    python scripts/benchmarks/model_benchmark.py --cases adv          # only the adversarial cases
Writes a report to output/model_benchmark_<ts>.{txt,json}.
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime

import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT
OUT_DIR = os.path.join(ROOT, "output")

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console mangles em-dashes otherwise
except Exception:
    pass

from openai import OpenAI

from config import GOVERNMENT_WARNING, REQUEST_TIMEOUT_SECONDS
from extraction import _build_content, _model_params, _parse_response
from verification import (
    _check_warning, _warning_body, _normalize, _CANONICAL_WARNING_BODY_NORM,
    PASS, REVIEW, FAIL,
)

# --- models ------------------------------------------------------------------
# Unavailable ones are skipped at run time.
OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano"]

# --- test cases with ground truth -------------------------------------------
ADV = os.path.join(ROOT, "adversarial")
BASE = os.path.join(ROOT, "test_labels", "baseline_labels")


def _case(cid, images, caps, bold, wording):
    return {"id": cid, "images": images, "gt": {"caps": caps, "bold": bold, "wording": wording}}


CASES = [
    _case("adv_01_compliant", [os.path.join(ADV, "01_compliant.png")], True, True, "exact"),
    _case("adv_02_titlecase", [os.path.join(ADV, "02_titlecase.png")], False, True, "exact"),
    _case("adv_03_notbold",   [os.path.join(ADV, "03_notbold.png")],   True, False, "exact"),
    _case("adv_04_reworded",  [os.path.join(ADV, "04_reworded.png")],  True, True, "wrong"),
    # baselines carry caps/wording truth, but bold is UNKNOWN (None): these are realistic
    # renders, not font-controlled, so eyeballing the header weight isn't valid ground truth.
    # Only the adversarial set (above) provides certifiable bold truth.
    _case("baseline_1", [os.path.join(BASE, "baseline_1_Front.png"), os.path.join(BASE, "baseline_1_Other.png")], True, None, "exact"),
    _case("baseline_2", [os.path.join(BASE, "baseline_2_Front.png"), os.path.join(BASE, "baseline_2_Other.png")], True, None, "exact"),
    _case("baseline_3", [os.path.join(BASE, "baseline_3_Front.png"), os.path.join(BASE, "baseline_3_Other.png")], True, None, "exact"),
]


# --- key loading -------------------------------------------------------------

def _secrets_value(key_name):
    secrets = os.path.join(ROOT, ".streamlit", "secrets.toml")
    if os.path.exists(secrets):
        with open(secrets, encoding="utf-8") as fh:
            m = re.search(rf'{key_name}\s*=\s*"([^"]+)"', fh.read())
        if m and m.group(1) and not m.group(1).startswith(("sk-...", "...")):
            return m.group(1)
    return None


def _load_keys():
    return os.environ.get("OPENAI_API_KEY") or _secrets_value("OPENAI_API_KEY")


# --- one extraction call -----------------------------------------------------

def _extract(client, model, images):
    """Run the app's extraction prompt on one model; return the coerced schema dict.

    Falls back to dropping response_format if a model rejects it (Structured Outputs is not
    universally supported), since the prompt already demands JSON-only output."""
    content = _build_content(images, "image/png")
    params = _model_params(model)
    last = None
    for _ in range(3):   # retry once per known param rejection, then give up
        try:
            resp = client.chat.completions.create(messages=[{"role": "user", "content": content}], **params)
            return _parse_response(resp)
        except Exception as exc:
            last = exc
            msg = str(exc)
            if ("response_format" in msg or "json_schema" in msg) and "response_format" in params:
                params["response_format"] = {"type": "json_object"}   # SO unsupported -> JSON mode
            elif "reasoning_effort" in msg and params.get("reasoning_effort") == "minimal":
                params["reasoning_effort"] = "low"        # some reasoning models reject 'minimal'
            else:
                raise
    raise last


# --- scoring -----------------------------------------------------------------

def _wording_matches_canon(text):
    return _normalize(_warning_body(text or "")) == _CANONICAL_WARNING_BODY_NORM


def _effective_caps(gw):
    """Mirror _check_warning: caps come from the literal text when the header is in it,
    else from the model's header_all_caps flag."""
    text = gw.get("text") or ""
    m = re.search(r"government\s+warning", text, re.IGNORECASE)
    if m:
        return text[m.start():m.end()].isupper()
    return gw.get("header_all_caps")


def _score(gw, gt):
    """Compare one model's warning observations to ground truth. Returns a dict of bools
    (None where the model couldn't determine a signal)."""
    eff_caps = _effective_caps(gw)
    bold = gw.get("header_bold")
    wording_match = _wording_matches_canon(gw.get("text"))
    want_match = gt["wording"] == "exact"
    gt_bold = gt["bold"]   # None => bold truth unknown for this case (realistic baselines)
    return {
        "wording_ok": wording_match == want_match,      # exact->should match; wrong->should NOT match
        "caps_ok": (eff_caps == gt["caps"]) if eff_caps is not None else None,
        "bold_ok": None if (gt_bold is None or bold is None) else (bold == gt_bold),
        "obs": {"text_matches_canon": wording_match, "eff_caps": eff_caps,
                "header_all_caps": gw.get("header_all_caps"), "header_bold": bold,
                "header_bold_confidence": gw.get("header_bold_confidence"),
                "header_bold_basis": gw.get("header_bold_basis"),
                "confidence": gw.get("confidence")},
    }


def _tri(ok):
    return "—" if ok is None else ("OK" if ok else "XX")


# --- run ---------------------------------------------------------------------

def _build_model_list(restrict, openai_key):
    models, seen = [], set()
    for m in OPENAI_MODELS:
        if not restrict or m in restrict:
            models.append({"name": m, "key": openai_key})
            seen.add(m)
    # allow any explicitly-requested model not in the predefined list
    for m in (restrict or []):
        if m in seen:
            continue
        models.append({"name": m, "key": openai_key})
    return models


def _client_for(cfg):
    return OpenAI(api_key=cfg["key"], timeout=REQUEST_TIMEOUT_SECONDS)


def main():
    args = sys.argv[1:]
    only_cases = None
    if "--cases" in args:
        i = args.index("--cases")
        only_cases = args[i + 1] if i + 1 < len(args) else "all"
        del args[i:i + 2]
    restrict = set(args) if args else None

    cases = CASES
    if only_cases == "adv":
        cases = [c for c in CASES if c["id"].startswith("adv_")]
    elif only_cases == "baseline":
        cases = [c for c in CASES if c["id"].startswith("baseline")]

    openai_key = _load_keys()
    if not openai_key:
        sys.exit("ERROR: no OpenAI key (env OPENAI_API_KEY or .streamlit/secrets.toml).")
    models = _build_model_list(restrict, openai_key)
    if not models:
        sys.exit("No models selected.")

    # preload images once per case
    for c in cases:
        c["loaded"] = [(open(p, "rb").read(), "image/png") for p in c["images"]]

    # one client per model, reused across its cases (OpenAI client is thread-safe)
    for m in models:
        m["client"] = _client_for(m)

    results = {}   # model -> {case_id -> cell}
    avail = {}     # model -> bool/str
    jobs = [(m, c) for m in models for c in cases]
    total, done = len(jobs), 0
    print(f"Running {len(models)} model(s) x {len(cases)} case(s) = {total} calls...\n")

    def _run(m, c):
        start = time.perf_counter()
        try:
            extracted = _extract(m["client"], m["name"], c["loaded"])
        except Exception as exc:
            return m["name"], c["id"], {"error": str(exc)[:200]}
        elapsed = time.perf_counter() - start
        gw = extracted.get("government_warning", {})
        verdict = _check_warning(gw)
        sc = _score(gw, c["gt"])
        return m["name"], c["id"], {
            "elapsed": round(elapsed, 2), "verdict": verdict.status,
            "score": {k: sc[k] for k in ("wording_ok", "caps_ok", "bold_ok")},
            "obs": sc["obs"], "gt": c["gt"],
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_run, m, c) for m, c in jobs]
        for fut in as_completed(futs):
            name, cid, cell = fut.result()
            done += 1
            results.setdefault(name, {})[cid] = cell
            tag = "ERR" if "error" in cell else cell["verdict"]
            print(f"  [{done}/{total}] {name:16s} {cid:18s} -> {tag}")

    # availability: a model whose every cell errored is "unavailable"
    for m in models:
        cells = results.get(m["name"], {})
        avail[m["name"]] = not all("error" in c for c in cells.values()) if cells else False

    lines = ["", "=" * 78, "GOVERNMENT-WARNING MODEL BENCHMARK", "=" * 78]
    lines.append(f"cases: {', '.join(c['id'] for c in cases)}")
    lines.append("ground truth per case (caps / bold / wording):")
    for c in cases:
        g = c["gt"]
        lines.append(f"   {c['id']:18s} caps={str(g['caps']):5s} bold={str(g['bold']):5s} wording={g['wording']}")
    lines.append("")

    # per-model detail
    for m in models:
        name = m["name"]
        cells = results.get(name, {})
        if not avail.get(name):
            err = next((c["error"] for c in cells.values() if "error" in c), "no cells")
            lines.append(f"--- {name} : UNAVAILABLE ({err}) ---\n")
            continue
        lines.append(f"--- {name} ---")
        lines.append(f"   {'case':18s} {'wording':8s} {'caps':5s} {'bold':5s} {'verdict':13s} obs(caps/bold)")
        w_ok = ca_ok = b_ok = 0
        w_n = ca_n = b_n = 0
        for c in cases:
            cell = cells.get(c["id"], {})
            if "error" in cell:
                lines.append(f"   {c['id']:18s} ERROR: {cell['error'][:60]}")
                continue
            s = cell["score"]
            o = cell["obs"]
            lines.append(f"   {c['id']:18s} {_tri(s['wording_ok']):8s} {_tri(s['caps_ok']):5s} "
                         f"{_tri(s['bold_ok']):5s} {cell['verdict']:13s} "
                         f"caps={str(o['eff_caps']):5s} bold={str(o['header_bold'])}")
            if s["wording_ok"] is not None:
                w_n += 1; w_ok += bool(s["wording_ok"])
            if s["caps_ok"] is not None:
                ca_n += 1; ca_ok += bool(s["caps_ok"])
            if s["bold_ok"] is not None:
                b_n += 1; b_ok += bool(s["bold_ok"])
        # headline discrimination checks
        c01 = cells.get("adv_01_compliant", {}).get("obs", {})
        c03 = cells.get("adv_03_notbold", {}).get("obs", {})
        c02 = cells.get("adv_02_titlecase", {}).get("obs", {})
        bold_disc = (c01.get("header_bold") is True and c03.get("header_bold") is False)
        caps_disc = (c01.get("eff_caps") is True and c02.get("eff_caps") is False)
        lines.append(f"   SCORE  wording {w_ok}/{w_n}  caps {ca_ok}/{ca_n}  bold {b_ok}/{b_n}")
        lines.append(f"   BOLD discriminates compliant vs not-bold? {'YES' if bold_disc else 'NO'}  "
                     f"(01 bold={c01.get('header_bold')}, 03 bold={c03.get('header_bold')})")
        lines.append(f"   CAPS catches title case?                  {'YES' if caps_disc else 'NO'}  "
                     f"(01 caps={c01.get('eff_caps')}, 02 caps={c02.get('eff_caps')})")
        lines.append("   bold reasoning (case: bold [conf] basis):")
        for _cid in ("adv_01_compliant", "adv_03_notbold", "baseline_1", "baseline_2", "baseline_3"):
            ob = cells.get(_cid, {}).get("obs", {})
            if ob:
                lines.append(f"      {_cid:18s} {str(ob.get('header_bold')):5s} "
                             f"[{ob.get('header_bold_confidence')}] {ob.get('header_bold_basis')!r}")
        et = sorted(c["elapsed"] for c in cells.values() if "elapsed" in c)
        if et:
            lines.append(f"   TIME   mean {sum(et) / len(et):.2f}s  median {et[len(et) // 2]:.2f}s  "
                         f"(min {et[0]:.2f}s, max {et[-1]:.2f}s)")
        lines.append("")

    # read-time ranking across models (fastest first)
    timing = []
    for m in models:
        ts = [c["elapsed"] for c in results.get(m["name"], {}).values() if "elapsed" in c]
        if ts:
            timing.append((m["name"], sum(ts) / len(ts), sorted(ts)[len(ts) // 2]))
    if timing:
        timing.sort(key=lambda t: t[1])
        lines.append("--- read time per model (fastest first) ---")
        for name, mean_t, med_t in timing:
            lines.append(f"   {name:18s} mean {mean_t:5.2f}s   median {med_t:5.2f}s")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt = os.path.join(OUT_DIR, f"model_benchmark_{stamp}.txt")
    js = os.path.join(OUT_DIR, f"model_benchmark_{stamp}.json")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump({"models": [m["name"] for m in models], "available": avail,
                   "results": results}, fh, indent=2, ensure_ascii=False)
    print(f"\nWritten to:\n  {os.path.relpath(txt, ROOT)}\n  {os.path.relpath(js, ROOT)}")


if __name__ == "__main__":
    main()
