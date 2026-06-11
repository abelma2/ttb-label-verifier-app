"""Generate a controlled SAFETY benchmark of warning-block images for Experiment B3.

Bold ground truth is set by the actual font weight (Arial vs Arial Bold), and header/body are
rendered at the SAME size so only WEIGHT varies (isolates "bold" from "bigger"). For B3's question
-- "are the header strokes visibly thicker than the BODY text?" -- the ground truth is:

  bold_compliant : header BOLD,    body regular -> bold_gt True   (compliant header)
  notbold        : header REGULAR, body regular -> bold_gt False  (NOT-bold header -> false-PASS test)
  titlecase      : header BOLD title-case, body regular -> bold_gt True (caps differs; header still bold)
  boldbody       : header BOLD,    body BOLD    -> bold_gt False  (all-bold = remainder-bold violation;
                                                                    header is NOT thicker than the body)

Each variant is rendered clean and under realistic degradations: lowres, low-quality JPEG,
rotate+blur, and a curved/cylindrical warp (simulated can). Writes images + manifest.json to
bold_safety/. Requires Pillow + the Windows Arial fonts. BENCHMARK ONLY.

Run:  python scripts/benchmarks/generate_bold_safety.py
"""
import json
import math
import os
import textwrap

from PIL import Image, ImageDraw, ImageFilter, ImageFont

import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT
OUT = os.path.join(ROOT, "bold_safety")
os.makedirs(OUT, exist_ok=True)

_FONTS = r"C:\Windows\Fonts"
def _reg(s): return ImageFont.truetype(os.path.join(_FONTS, "arial.ttf"), s)
def _bold(s): return ImageFont.truetype(os.path.join(_FONTS, "arialbd.ttf"), s)

HEADER_CAPS = "GOVERNMENT WARNING:"
HEADER_TITLE = "Government Warning:"
BODY = ("(1) According to the Surgeon General, women should not drink alcoholic beverages during "
        "pregnancy because of the risk of birth defects. (2) Consumption of alcoholic beverages "
        "impairs your ability to drive a car or operate machinery, and may cause health problems.")

# variant -> (header_text, header_bold, body_bold, bold_gt for B3's "header thicker than body?")
VARIANTS = {
    "bold_compliant": (HEADER_CAPS,  True,  False, True),
    "notbold":        (HEADER_CAPS,  False, False, False),
    "titlecase":      (HEADER_TITLE, True,  False, True),
    "boldbody":       (HEADER_CAPS,  True,  True,  False),
}
SIZE = 16  # header AND body the same point size -> only weight differs


def _render(header_text, header_bold, body_bold):
    W = 760
    hf = _bold(SIZE) if header_bold else _reg(SIZE)
    bf = _bold(SIZE) if body_bold else _reg(SIZE)
    lines = textwrap.wrap(BODY, width=78)
    H = 18 + 28 + len(lines) * 21 + 18
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((24, 18), header_text, font=hf, fill="black")
    y = 18 + 28
    for ln in lines:
        d.text((24, y), ln, font=bf, fill="black"); y += 21
    return img


def _curve(img, amp=0.07):
    """Simulate text curving around a cylinder: bow each column vertically by a sine of x."""
    w, h = img.size
    A = int(h * amp) + 6
    out = Image.new("RGB", (w, h + 2 * A), "white")
    step = 4
    for x in range(0, w, step):
        strip = img.crop((x, 0, min(x + step, w), h))
        dy = int(A * math.sin(math.pi * (x + step / 2) / w))
        out.paste(strip, (x, A - dy))
    return out


def _distortions(img):
    yield "clean", img, "png"
    yield "lowres", img.resize((int(img.width * 0.42), int(img.height * 0.42))), "png"
    yield "jpeg", img, "jpeg"  # heavy JPEG compression at save time
    yield "rotblur", img.rotate(4, expand=True, fillcolor="white").filter(ImageFilter.GaussianBlur(1.3)), "png"
    yield "curved", _curve(img), "png"


def main():
    manifest = {}
    for vname, (htext, hbold, bbold, gt) in VARIANTS.items():
        base = _render(htext, hbold, bbold)
        for dname, im, fmt in _distortions(base):
            fn = f"{vname}__{dname}.{'jpg' if fmt == 'jpeg' else 'png'}"
            path = os.path.join(OUT, fn)
            if fmt == "jpeg":
                im.save(path, "JPEG", quality=20)
            else:
                im.save(path, "PNG")
            manifest[fn] = {"variant": vname, "distortion": dname, "bold_gt": gt,
                            "header_bold_font": hbold, "body_bold_font": bbold}
            print("wrote", fn)
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n{len(manifest)} images + manifest -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
