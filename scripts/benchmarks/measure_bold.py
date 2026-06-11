"""Deterministic stroke-weight measurement: is the 'GOVERNMENT WARNING' header actually
bolder than the body text? Answers whether the baseline labels are valid BOLD ground truth.

Method (self-calibrating, no absolute thresholds):
  For a cropped warning block, binarize, split into text line-bands by horizontal
  projection, treat the first band as the header and the rest as body, and for each band
  estimate a size-normalized stroke weight:
      stroke_width  = median horizontal run-length of ink pixels
      weight ratio  = stroke_width / band_height        (bold has thicker strokes per height)
  The discriminator is HEADER weight / BODY weight WITHIN each image (cancels font size).

Calibration anchors (font-controlled by scripts/generate_adversarial.py):
  adv_01_compliant -> header Arial BOLD,    body Arial regular  => ratio should be HIGH
  adv_03_notbold   -> header Arial regular, body Arial regular  => ratio should be ~1
Then we see whether each baseline's ratio looks like the BOLD anchor or the REGULAR anchor.

Run:  python scripts/benchmarks/measure_bold.py     (needs Pillow)
"""
import os
import sys

import _paths
_paths.ensure_paths()
ROOT = _paths.ROOT
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from PIL import Image
except ImportError:
    sys.exit("ERROR: Pillow not installed.  pip install pillow")

ADV = os.path.join(ROOT, "adversarial")
BASE = os.path.join(ROOT, "test_labels", "baseline_labels")
INK = 160   # grayscale <= INK counts as ink (catches dark navy headers, ignores light bg)

# (id, path, crop as fractions x0,y0,x1,y1, note)
TARGETS = [
    ("adv_01_compliant  [BOLD anchor]",   os.path.join(ADV, "01_compliant.png"), (0.03, 0.40, 0.95, 0.62)),
    ("adv_03_notbold    [REG  anchor]",   os.path.join(ADV, "03_notbold.png"),   (0.03, 0.40, 0.95, 0.62)),
    ("baseline_1_Other",  os.path.join(BASE, "baseline_1_Other.png"), (0.04, 0.64, 0.80, 0.95)),
    ("baseline_2_Other",  os.path.join(BASE, "baseline_2_Other.png"), (0.04, 0.77, 0.82, 0.99)),
    ("baseline_3_Other",  os.path.join(BASE, "baseline_3_Other.png"), (0.06, 0.59, 0.66, 0.94)),
]


def _ink_rows(px, w, h):
    """ink-pixel count per row."""
    return [sum(1 for x in range(w) if px[x, y] <= INK) for y in range(h)]


def _bands(rows, w):
    """Group contiguous text rows (allowing 1px gaps) into (y0,y1) bands. A text row has
    more than a trace of ink."""
    thresh = max(2, int(0.01 * w))
    bands, start = [], None
    gap = 0
    for y, c in enumerate(rows):
        if c > thresh:
            if start is None:
                start = y
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap > 2:
                    bands.append((start, y - gap + 1))
                    start = None
    if start is not None:
        bands.append((start, len(rows)))
    return [(a, b) for a, b in bands if b - a >= 3]   # drop 1-2px specks


def _weight(px, w, band):
    """stroke_width (median ink run length) and height-normalized weight for a band."""
    y0, y1 = band
    height = y1 - y0
    runs = []
    ink_cols = set()
    for y in range(y0, y1):
        run = 0
        for x in range(w):
            if px[x, y] <= INK:
                run += 1
                ink_cols.add(x)
            else:
                if run:
                    runs.append(run)
                run = 0
        if run:
            runs.append(run)
    # ignore very long runs (underlines / solid graphics), keep glyph strokes
    runs = sorted(r for r in runs if r <= max(3, height))
    if not runs:
        return 0.0, 0.0, height
    stroke = runs[len(runs) // 2]              # median run length ~ stroke thickness
    return stroke, stroke / height, height


def main():
    print(f"ink threshold = {INK} (grayscale <= this is ink)\n")
    anchors = {}
    for tid, path, crop in TARGETS:
        if not os.path.exists(path):
            print(f"{tid}: MISSING {path}\n")
            continue
        img = Image.open(path).convert("L")
        W, H = img.size
        x0, y0, x1, y1 = (int(crop[0] * W), int(crop[1] * H), int(crop[2] * W), int(crop[3] * H))
        sub = img.crop((x0, y0, x1, y1))
        w, h = sub.size
        px = sub.load()
        rows = _ink_rows(px, w, h)
        bands = _bands(rows, w)
        print(f"=== {tid} ===  ({os.path.basename(path)} {W}x{H}, crop {x0},{y0}-{x1},{y1})")
        if len(bands) < 2:
            print(f"   only {len(bands)} text band(s) found in crop — adjust crop. bands={bands}\n")
            continue
        header = bands[0]
        # body = the union span of the remaining bands; measure them pooled
        body = (bands[1][0], bands[-1][1])
        hs, hr, hh = _weight(px, w, header)
        bs, br, bh = _weight(px, w, body)
        ratio = hr / br if br else 0.0
        print(f"   header band y={header} h={hh}px  stroke={hs}px  weight(s/h)={hr:.3f}")
        print(f"   body   band y={body} h={bh}px  stroke={bs}px  weight(s/h)={br:.3f}")
        print(f"   --> HEADER/BODY weight ratio = {ratio:.2f}\n")
        if "BOLD anchor" in tid:
            anchors["bold"] = ratio
        elif "REG  anchor" in tid:
            anchors["reg"] = ratio
        else:
            anchors.setdefault("baselines", []).append((tid, ratio))

    if "bold" in anchors and "reg" in anchors:
        bold, reg = anchors["bold"], anchors["reg"]
        mid = (bold + reg) / 2
        print("-" * 60)
        print(f"CALIBRATION:  BOLD anchor ratio = {bold:.2f}   REGULAR anchor ratio = {reg:.2f}")
        print(f"midpoint = {mid:.2f}  (ratio above midpoint => bold-like, below => regular-like)\n")
        for tid, r in anchors.get("baselines", []):
            verdict = "BOLD-like" if r > mid else "REGULAR-like (NOT clearly bold)"
            print(f"   {tid:18s} ratio={r:.2f}  ->  {verdict}")


if __name__ == "__main__":
    main()
