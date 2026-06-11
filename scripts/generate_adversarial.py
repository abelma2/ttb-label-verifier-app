"""Generate synthetic adversarial label images to VALIDATE the government-warning checks.

Each image is a complete single-image label with all mandatory fields; only the warning
varies, so the verdict isolates the warning rule. Bold is controlled by the actual font
(Arial vs Arial Bold) and caps by the literal text, giving us ground truth:

  01_compliant  -> ALL-CAPS + BOLD header, exact body            -> expect PASS
  02_titlecase  -> 'Government Warning:' (title case), BOLD       -> expect FAIL (caps)
  03_notbold    -> ALL-CAPS but REGULAR (not bold) header         -> expect FAIL/REVIEW (bold)
  04_reworded   -> ALL-CAPS + BOLD header, WRONG wording          -> expect FAIL (wording)

Requires Pillow and the Windows Arial fonts. Run:  python scripts/generate_adversarial.py
"""
import os
import textwrap

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "adversarial")
os.makedirs(OUT, exist_ok=True)

_FONTS = r"C:\Windows\Fonts"
def _reg(size):
    return ImageFont.truetype(os.path.join(_FONTS, "arial.ttf"), size)
def _bold(size):
    return ImageFont.truetype(os.path.join(_FONTS, "arialbd.ttf"), size)

WARNING_BODY = (
    "(1) According to the Surgeon General, women should not drink alcoholic beverages "
    "during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic "
    "beverages impairs your ability to drive a car or operate machinery, and may cause "
    "health problems.")
REWORDED_BODY = (
    "(1) Pregnant women should avoid alcohol because of possible birth defects. "
    "(2) Alcohol can impair your driving and may cause health issues.")


def make(filename, header_text, header_font, body_text):
    img = Image.new("RGB", (720, 600), "white")
    d = ImageDraw.Draw(img)
    d.text((30, 25), "OLD TOM DISTILLERY", font=_bold(30), fill="black")
    d.text((30, 72), "Kentucky Straight Bourbon Whiskey", font=_reg(18), fill="black")
    d.text((30, 104), "45% ALC./VOL. (90 PROOF)", font=_reg(16), fill="black")
    d.text((30, 134), "750 mL", font=_reg(16), fill="black")
    d.text((30, 164), "DISTILLED & BOTTLED BY OLD TOM DISTILLERY, BARDSTOWN, KY",
           font=_reg(14), fill="black")
    y = 250
    d.text((30, y), header_text, font=header_font, fill="black")   # the controlled header
    y += 28
    for line in textwrap.wrap(body_text, width=74):
        d.text((30, y), line, font=_reg(14), fill="black")          # body always regular 14
        y += 21
    img.save(os.path.join(OUT, filename))
    print("wrote", os.path.relpath(os.path.join(OUT, filename), ROOT))


if __name__ == "__main__":
    make("01_compliant.png", "GOVERNMENT WARNING:", _bold(15), WARNING_BODY)
    make("02_titlecase.png", "Government Warning:", _bold(15), WARNING_BODY)
    make("03_notbold.png",   "GOVERNMENT WARNING:", _reg(15),  WARNING_BODY)
    make("04_reworded.png",  "GOVERNMENT WARNING:", _bold(15), REWORDED_BODY)
    print("done ->", os.path.relpath(OUT, ROOT))
