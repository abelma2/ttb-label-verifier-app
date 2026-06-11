"""Diagnostic: does a cropped/zoomed close-up recover a field the full-image read missed?

For test_2 (Shiner Orange Wit) the full-label read missed the name/address that IS printed in
the top front medallion. This crops that region, upscales it, and re-runs extract_fields on
just the crop to show whether a clearer close-up recovers the read.

Run:  python scripts/benchmarks/crop_diag.py
"""
import io
import os
import re
import sys

import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from PIL import Image


def _load_key():
    if os.environ.get("OPENAI_API_KEY"):
        return
    secrets = os.path.join(ROOT, ".streamlit", "secrets.toml")
    if os.path.exists(secrets):
        with open(secrets, encoding="utf-8") as fh:
            m = re.search(r'OPENAI_API_KEY\s*=\s*"([^"]+)"', fh.read())
        if m and m.group(1) and m.group(1) != "sk-...":
            os.environ["OPENAI_API_KEY"] = m.group(1)


def _crop(path, box, scale=3):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    c = img.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
    return c.resize((c.width * scale, c.height * scale))


def main():
    _load_key()
    from extraction import extract_fields

    front = os.path.join(ROOT, "test_labels", "real_labels", "test_2_Front.jpeg")
    # the top medallion (name/address) sits in the upper-middle of the front label
    crop = _crop(front, (0.25, 0.15, 0.78, 0.40), scale=3)
    out = os.path.join(ROOT, "output", "test_2_namecrop.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    crop.save(out)

    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    ext = extract_fields([(buf.getvalue(), "image/png")])
    print(f"cropped close-up of the test_2 front medallion (saved {os.path.relpath(out, ROOT)}):")
    for fld in ("brand_name", "name_and_address", "net_contents"):
        v = ext.get(fld, {})
        print(f"  {fld:16s} present={v.get('present')}  value={v.get('value')!r}  conf={v.get('confidence')}")


if __name__ == "__main__":
    main()
