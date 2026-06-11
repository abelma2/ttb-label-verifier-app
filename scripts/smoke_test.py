"""Smoke test: run the REAL extraction + verification pipeline on local image(s).

This is the end-to-end check the unit tests can't do — it calls the vision model, so
you need an OpenAI API key. The key is read from OPENAI_API_KEY, or from
.streamlit/secrets.toml (the same file the app uses).

  INPUT:   image files in  test_labels/   (or any path you pass)
  OUTPUT:  printed to the console AND written to  output/result_<timestamp>.txt / .json
  RUNTIME: the vision-read time per product is reported (~7s/bottle budget for an
           accurate front+back detail=high read; see BENCHMARK_NOTES.md).

Usage (from the project root):

    # all images passed are treated as ONE product (e.g. a front + back label):
    python scripts/smoke_test.py test_labels/front.png test_labels/back.png

    # point at a folder to read every image in it as one product:
    python scripts/smoke_test.py test_labels

    # test each image as its own separate product:
    python scripts/smoke_test.py --each test_labels

    # group front/back by filename (test_1_Front + test_1_Other -> one product):
    python scripts/smoke_test.py --group test_labels

It reports the extracted JSON, the per-label rules screening (verify_label_only — the
government warning + mandatory-field presence), and the read time. For full
label-vs-application matching, use the app UI (streamlit run app.py).
"""
import json
import os
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output")
sys.path.insert(0, ROOT)

# Windows consoles default to cp1252 and mangle em-dashes; print as UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _load_key():
    """Return the OpenAI key from the env or .streamlit/secrets.toml; set it in the env."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    secrets = os.path.join(ROOT, ".streamlit", "secrets.toml")
    if os.path.exists(secrets):
        with open(secrets, encoding="utf-8") as fh:
            m = re.search(r'OPENAI_API_KEY\s*=\s*"([^"]+)"', fh.read())
        if m and m.group(1) and m.group(1) != "sk-...":
            os.environ["OPENAI_API_KEY"] = m.group(1)
            return m.group(1)
    return None


def _media_type(path):
    return "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"


def _is_image(name):
    return name.lower().endswith((".png", ".jpg", ".jpeg"))


def _gather(paths):
    """Expand any folders into the image files they contain."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(os.path.join(p, n) for n in sorted(os.listdir(p)) if _is_image(n))
        elif _is_image(p):
            out.append(p)
    return out


def _group_by_product(files):
    """Group files by product, stripping a trailing _Front/_Other/_Back/_Label suffix from
    the filename (so test_1_Front.jpg + test_1_Other.jpg -> one 'test_1' product)."""
    groups = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        key = re.sub(r"[ _\-]*(front|other|back|label).*$", "", stem, flags=re.IGNORECASE) or stem
        groups.setdefault(key, []).append(f)
    return groups


def _run_one(extract_fields, verify_label_only, label, files):
    """Run the pipeline on one product (one or more images). Returns (lines, record)."""
    images = [(open(f, "rb").read(), _media_type(f)) for f in files]
    start = time.perf_counter()
    extracted = extract_fields(images)            # the vision-model call (the slow part)
    elapsed = time.perf_counter() - start
    result = verify_label_only(extracted)

    record = {
        "label": label,
        "images": [os.path.relpath(f, ROOT) for f in files],
        "runtime_seconds": round(elapsed, 2),
        "extraction": extracted,
        "screening": {
            "overall": result["overall"],
            "beverage_type": result.get("beverage_type"),
            "fields": [asdict(f) for f in result["fields"]],
            "additional_statements": result.get("additional_statements", []),
            "image_quality_notes": result.get("image_quality_notes"),
        },
    }

    icon = {"pass": "OK ", "needs_review": "?? ", "fail": "XX "}
    lines = [f"=== {label} ==="]
    lines += [f"  - {os.path.relpath(f, ROOT)}" for f in files]
    budget = "  <-- over the ~7s budget!" if elapsed > 7 else ""
    lines.append(f"  RUNTIME (vision read): {elapsed:.2f}s{budget}")
    lines.append("  --- extraction ---")
    lines.append("  " + json.dumps(extracted, indent=2, ensure_ascii=False).replace("\n", "\n  "))
    lines.append(f"  --- screening (rules only) -- OVERALL: {result['overall'].upper()} ---")
    for fr in result["fields"]:
        lines.append(f"  {icon.get(fr.status, '   ')}{fr.field}: {fr.reason}")
    for s in result.get("additional_statements") or []:
        lines.append(f"  +  other statement: {s.get('value')}")
    if result.get("image_quality_notes"):
        lines.append(f"  !  image note: {result['image_quality_notes']}")
    return lines, record


def main():
    args = sys.argv[1:]
    each = "--each" in args
    group = "--group" in args
    paths = [a for a in args if a not in ("--each", "--group")] or ["test_labels"]

    if not _load_key():
        sys.exit("ERROR: no OpenAI key. Put it in .streamlit/secrets.toml (replace the "
                 "sk-... placeholder), or set OPENAI_API_KEY in your environment.")

    files = _gather(paths)
    if not files:
        sys.exit("No images found. Put .png/.jpg files in test_labels/ or pass paths.")

    if group:
        gmap = _group_by_product(files)
        items = [(k, gmap[k]) for k in sorted(gmap)]
    elif each:
        items = [(os.path.basename(f), [f]) for f in files]
    else:
        items = [(f"{len(files)} image(s) as one product", files)]

    from extraction import extract_fields
    from verification import verify_label_only
    extract = extract_fields

    all_lines, records = [], []
    total_start = time.perf_counter()
    for label, group_files in items:
        try:
            lines, rec = _run_one(extract, verify_label_only, label, group_files)
        except Exception as exc:  # one bad product must not sink the whole run
            lines = [f"=== {label} ===", f"  ERROR: {exc}"]
            rec = {"label": label, "error": str(exc)}
        all_lines += lines + [""]
        records.append(rec)
    total = time.perf_counter() - total_start

    all_lines.append(f"TOTAL RUNTIME: {total:.2f}s for {len(records)} product(s)")
    report = "\n".join(all_lines)
    print(report)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(OUT_DIR, f"result_{stamp}.txt")
    json_path = os.path.join(OUT_DIR, f"result_{stamp}.json")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"runtime_seconds": round(total, 2), "products": records},
                  fh, indent=2, ensure_ascii=False)
    print(f"\nResults written to:\n  {os.path.relpath(txt_path, ROOT)}\n  {os.path.relpath(json_path, ROOT)}")


if __name__ == "__main__":
    main()
