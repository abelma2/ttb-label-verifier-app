"""Image-downscale sweep: the latency lever that actually works without losing accuracy.

The input labels are ~6 MB / ~1632px -- far more resolution than the model needs to read the
text. detail=high tiles the image into 512px tiles, so a smaller image = fewer tiles = less
vision encoding = faster, with the SAME accuracy-safe single combined call (no risky split/merge).

This sweeps image WIDTH x product on the production combined read, measuring wall-clock AND
accuracy (non-warning false-fails + warning verdict), plus the MIN transcribed warning-text
length per width -- the safety signal that the small warning text is still being read (a short
read means the downscale went too far).

Usage:  python scripts/benchmarks/image_scale.py [model ...]
Writes output/image_scale_<ts>.{txt,json}.
"""
import io
import json
import os
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

from PIL import Image
import model_image_experiment as M
from extraction import _model_params  # noqa: F401  (kept for parity; read goes via M)
from verification import verify, PASS, REVIEW, FAIL

WIDTHS = [None, 1280, 1024, 896, 800, 700]   # None = original
REPEATS = 2
MODELS = sys.argv[1:] or ["gpt-5.4-mini", "gpt-5.5"]
# the full canonical warning is ~280 chars; flag a width as DEGRADED if any read drops well below
_WARN_MIN_OK = 240


def _resize(loaded, width):
    out = []
    for b, _mt in loaded:
        im = Image.open(io.BytesIO(b)).convert("RGB")
        if width and im.width > width:
            im = im.resize((width, int(im.height * width / im.width)))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        out.append((buf.getvalue(), "image/png"))
    return out


def main():
    print(f"models: {MODELS}   widths: {WIDTHS}   products: {len(M.PRODUCTS)}   repeats: {REPEATS}\n")
    # pre-resize once per (product, width)
    resized = {(p["label"], w): _resize(p["loaded"], w) for p in M.PRODUCTS for w in WIDTHS}
    kb = {(p["label"], w): len(resized[(p["label"], w)][0][0]) // 1024
          for p in M.PRODUCTS for w in WIDTHS}

    report = {}
    for model in MODELS:
        print(f"=== {model} ===")
        for w in WIDTHS:
            walls, nonwarn_fail, warn_dist = [], 0, Counter()
            warn_text_lens = []
            for p in M.PRODUCTS:
                for rep in range(REPEATS):
                    try:
                        t0 = time.perf_counter()
                        ex = M._extract_with_model(model, resized[(p["label"], w)])
                        dt = round(time.perf_counter() - t0, 2)
                        r = verify(ex, p["app"])
                        statuses = {f.field: f.status for f in r["fields"]}
                        gw = next((f for f in r["fields"] if f.field == "government_warning"), None)
                        walls.append(dt)
                        nonwarn_fail += sum(1 for f, s in statuses.items()
                                            if s == FAIL and f != "government_warning")
                        warn_dist[getattr(gw, "status", None)] += 1
                        warn_text_lens.append(len((ex.get("government_warning") or {}).get("text") or ""))
                    except Exception as exc:
                        print(f"  w={w} {p['label']} rep{rep}: ERROR {str(exc)[:90]}")
            if walls:
                ws = sorted(walls)
                report.setdefault(model, {})[str(w)] = {
                    "width": w or "orig",
                    "img_kb": kb[(M.PRODUCTS[0]["label"], w)],
                    "wall_mean": round(sum(walls) / len(walls), 2), "wall_median": ws[len(ws) // 2],
                    "wall_min": ws[0], "wall_max": ws[-1],
                    "nonwarning_false_fail": nonwarn_fail,
                    "warning_verdicts": dict(warn_dist),
                    "min_warn_text_len": min(warn_text_lens) if warn_text_lens else 0,
                }
                r = report[model][str(w)]
                flag = "  <-- WARNING TEXT DEGRADED" if r["min_warn_text_len"] < _WARN_MIN_OK else ""
                wv = r["warning_verdicts"]
                print(f"  width={str(w or 'orig'):5s} ({r['img_kb']:4d}KB front): mean {r['wall_mean']}s "
                      f"(med {r['wall_median']}s)  NW-fail {nonwarn_fail}  "
                      f"warn P/R/F {wv.get(PASS,0)}/{wv.get(REVIEW,0)}/{wv.get(FAIL,0)}  "
                      f"min_warntext {r['min_warn_text_len']}{flag}")
        print()
    _write(report)


def _write(report):
    L = ["", "=" * 92, "IMAGE-DOWNSCALE SWEEP  (combined read; latency vs accuracy)  clearer_baseline_labels",
         "=" * 92,
         "fewer pixels -> fewer detail=high tiles -> faster, same single accuracy-safe call.",
         "min_warntext = shortest transcribed warning over all runs at that width (canonical ~280; "
         "a low value = the small warning text stopped reading -> downscaled too far).", ""]
    for model, widths in report.items():
        L.append(f"--- {model} ---")
        L.append(f"   {'width':6s} {'KB':6s} {'mean':7s} {'median':7s} {'NW-fail':8s} "
                 f"{'warn(P/R/F)':13s} {'min_warntext':12s}")
        for wkey, r in widths.items():
            wv = r["warning_verdicts"]
            prf = f"{wv.get(PASS,0)}/{wv.get(REVIEW,0)}/{wv.get(FAIL,0)}"
            flag = "  DEGRADED" if r["min_warn_text_len"] < _WARN_MIN_OK else ""
            L.append(f"   {str(r['width']):6s} {str(r['img_kb']):6s} {str(r['wall_mean'])+'s':7s} "
                     f"{str(r['wall_median'])+'s':7s} {str(r['nonwarning_false_fail']):8s} "
                     f"{prf:13s} {str(r['min_warn_text_len']):12s}{flag}")
        L.append("")
    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt = os.path.join(OUT_DIR, f"image_scale_{stamp}.txt")
    js = os.path.join(OUT_DIR, f"image_scale_{stamp}.json")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Written to:\n  {os.path.relpath(txt, ROOT)}\n  {os.path.relpath(js, ROOT)}")


if __name__ == "__main__":
    main()
