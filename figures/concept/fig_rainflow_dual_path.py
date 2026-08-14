"""
fig_rainflow_dual_path.py
-------------------------
Two-path degradation architecture (thesis label fig:dual_phi_arch).
ReportLab only, no matplotlib.

Run: python fig_rainflow_dual_path.py
Output: set OUTPUT below to "png", "pdf" or "both". Files are written next to
        this script. The PNG is rasterised from the same PDF the vector output
        uses, so the two cannot differ.

PNG export needs PyMuPDF (pip install pymupdf) or Poppler's pdftoppm on PATH.
"""

from __future__ import annotations
from pathlib import Path
import io
import shutil
import subprocess
import tempfile

from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth

# -- Output ------------------------------------------------------------------ #
OUTPUT = "png"                      # "png", "pdf" or "both"
STEM = "fig_rainflow_dual_path"      # output filename without extension

# ── Palette ───────────────────────────────────────────────────────────────────
TEAL_FILL   = HexColor("#E1F5EE")
TEAL_STR    = HexColor("#0F6E56")
TEAL_TITLE  = HexColor("#085041")
TEAL_SUB    = HexColor("#0F6E56")
TEAL_ARR    = HexColor("#0F6E56")

CORAL_FILL  = HexColor("#FAECE7")
CORAL_STR   = HexColor("#993C1D")
CORAL_TITLE = HexColor("#712B13")
CORAL_SUB   = HexColor("#993C1D")
CORAL_ARR   = HexColor("#993C1D")

GRAY_FILL   = HexColor("#F1EFE8")
GRAY_STR    = HexColor("#5F5E5A")
GRAY_TITLE  = HexColor("#444441")
GRAY_SUB    = HexColor("#5F5E5A")
GRAY_ARR    = HexColor("#5F5E5A")

MUTED       = HexColor("#5F5E5A")

# ── Canvas ────────────────────────────────────────────────────────────────────
FIG_W = 160 * mm
FIG_H =  72 * mm
DPI   = 300

# ── Layout (all in mm, converted inline) ──────────────────────────────────────
TOP_CY  = 56 * mm
BOT_CY  = 20 * mm
BOX_H   = 12 * mm
BOX_R   =  1.5 * mm

# Box widths are set from the widest string each box carries, measured with
# check_fit() below, plus 2 mm of padding. Margins are 9 mm on both sides.
IN_X    =  9 * mm
IN_W    = 32 * mm
IN_CY   = (TOP_CY + BOT_CY) / 2
IN_H    = 14 * mm

# Content boxes, 7 mm between columns for the arrows
B1_X    =  48 * mm; B1_W = 30 * mm
B2_X    =  85 * mm; B2_W = 35 * mm
B3_X    = 127 * mm; B3_W = 24 * mm

# Labels carrying sub- or superscripts, as (string, style) segments.
# The dispatch sub-gradient. Three points are easy to get wrong here:
#   f, not f_d.    f is the dispatch degradation cost of Equation 2.18; f_d is
#                  the annual fractional capacity loss, a different object, which
#                  belongs on the top row of this figure.
#   p_t, not e_t.  The LP consumes the derivative with respect to the dispatch
#                  decision, not the energy state.
#   No superscript. The box stands for the whole sub-gradient vector, which has a
#                  charge and a discharge component (Equations 2.22 and 2.23).
#                  Writing p_t^ch would name only half of what the box outputs.
#                  The caption declares the bare form and cites both equations.
GRADIENT_LABEL = [("\u2202f/\u2202p", ""), ("t", "sub")]
FD_LABEL       = [("f", ""), ("d", "sub"), (" accumulation", "")]
XU_STRESS_SUB  = [("S", ""), ("\u03b4", "sub"), ("(\u03b4) \u00b7 S", ""),
                  ("\u03c3", "sub"), ("(\u03c3) per cycle", "")]
SHI_SUB        = [("\u03a6 = k", ""), ("3", "sub"), ("\u03b4", ""),
                  ("k", "sup"), ("4", "sup"), (",  k", ""), ("4", "sub"),
                  (" > 1", "")]

FS_TITLE = 9
FS_SUB   = 7.5
FS_LABEL = 7
FS_LEG   = 7.5
LEG_Y    = 3.5 * mm


# ── Drawing helpers ───────────────────────────────────────────────────────────
def rounded_rect(c, x, y, w, h, r, fill, stroke, sw=0.4):
    p = c.beginPath()
    p.moveTo(x + r, y)
    p.lineTo(x + w - r, y)
    p.arcTo(x + w - 2*r, y,           x + w,     y + 2*r, startAng=-90, extent=90)
    p.lineTo(x + w, y + h - r)
    p.arcTo(x + w - 2*r, y + h - 2*r, x + w,     y + h,   startAng=0,   extent=90)
    p.lineTo(x + r, y + h)
    p.arcTo(x,       y + h - 2*r,     x + 2*r,   y + h,   startAng=90,  extent=90)
    p.lineTo(x, y + r)
    p.arcTo(x,       y,               x + 2*r,   y + 2*r, startAng=180, extent=90)
    p.close()
    c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(sw)
    c.drawPath(p, fill=1, stroke=1)


def seg_width(segs, font, size):
    """Width of a segment list, sub- and superscripts included."""
    total = 0.0
    for s, st in segs:
        if st == "subsup":                       # stacked, so they share an x
            total += max(stringWidth(s[0], font, size * 0.72),
                         stringWidth(s[1], font, size * 0.72))
        else:
            total += stringWidth(s, font, size if st == "" else size * 0.72)
    return total


def draw_centred_segs(c, cx, y, segs, font, size, color):
    """Centred text from (string, style) pairs.

    style is "", "sub", "sup", or "subsup". For "subsup" the string is a
    (subscript, superscript) pair drawn one above the other at the same x, as
    in p_t^ch.
    """
    x = cx - seg_width(segs, font, size) / 2
    c.setFillColor(color)
    small = size * 0.72
    for s, st in segs:
        if st == "subsup":
            c.setFont(font, small)
            c.drawString(x, y - 0.22 * size, s[0])
            c.drawString(x, y + 0.38 * size, s[1])
            x += max(stringWidth(s[0], font, small),
                     stringWidth(s[1], font, small))
        else:
            fs = size if st == "" else small
            dy = {"": 0.0, "sub": -0.22 * size, "sup": 0.38 * size}[st]
            c.setFont(font, fs)
            c.drawString(x, y + dy, s)
            x += stringWidth(s, font, fs)


def as_segs(text):
    """Accept a plain string or an existing segment list."""
    return text if isinstance(text, list) else [(text, "")]


def draw_box(c, x, cy, w, h, r, fill, stroke,
             title_col, sub_col, title, subtitle):
    y = cy - h / 2
    rounded_rect(c, x, y, w, h, r, fill, stroke)
    cx = x + w / 2
    draw_centred_segs(c, cx, cy + 1.6 * mm, as_segs(title),
                      "Helvetica-Bold", FS_TITLE, title_col)
    draw_centred_segs(c, cx, cy - 2.4 * mm, as_segs(subtitle),
                      "Helvetica", FS_SUB, sub_col)


def arrow_h(c, x1, x2, y, color, lw=0.8):
    """Horizontal arrow from x1 to x2 at height y."""
    head = 1.8 * mm
    hw   = 0.45 * head
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(lw)
    c.line(x1, y, x2 - head * 0.7, y)
    p = c.beginPath()
    p.moveTo(x2, y)
    p.lineTo(x2 - head, y + hw)
    p.lineTo(x2 - head, y - hw)
    p.close()
    c.drawPath(p, fill=1, stroke=0)


def swatch(c, x, y, color, label):
    sw, sh = 3.5 * mm, 2.5 * mm
    c.setFillColor(color); c.setLineWidth(0)
    c.roundRect(x, y, sw, sh, 1 * mm, fill=1, stroke=0)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", FS_LEG)
    c.drawString(x + sw + 2 * mm, y + 0.7 * mm, label)


# ── Build ─────────────────────────────────────────────────────────────────────
def build(buf):
    c = rl_canvas.Canvas(buf, pagesize=(FIG_W, FIG_H))
    c.setFillColor(white)
    c.rect(0, 0, FIG_W, FIG_H, fill=1, stroke=0)

    # Input box
    draw_box(c, IN_X, IN_CY, IN_W, IN_H, BOX_R,
             GRAY_FILL, GRAY_STR, GRAY_TITLE, GRAY_SUB,
             "Rainflow extraction", "cycles (\u03b4, \u03c3)")

    # Fork geometry
    FORK_X = IN_X + IN_W + 3 * mm
    c.setStrokeColor(GRAY_ARR); c.setLineWidth(0.8)
    c.line(IN_X + IN_W, IN_CY, FORK_X, IN_CY)   # horizontal stem
    c.line(FORK_X, BOT_CY, FORK_X, TOP_CY)       # vertical spine

    # Arms with arrowheads into first boxes
    arrow_h(c, FORK_X, B1_X, TOP_CY, TEAL_ARR)
    arrow_h(c, FORK_X, B1_X, BOT_CY, CORAL_ARR)

    # ── Top row (teal) ───────────────────────────────────────────────────────
    draw_box(c, B1_X, TOP_CY, B1_W, BOX_H, BOX_R,
             TEAL_FILL, TEAL_STR, TEAL_TITLE, TEAL_SUB,
             "Xu stress function", XU_STRESS_SUB)

    arrow_h(c, B1_X + B1_W, B2_X, TOP_CY, TEAL_ARR)

    draw_box(c, B2_X, TOP_CY, B2_W, BOX_H, BOX_R,
             TEAL_FILL, TEAL_STR, TEAL_TITLE, TEAL_SUB,
             FD_LABEL, "cyclic + calendar aging")

    arrow_h(c, B2_X + B2_W, B3_X, TOP_CY, TEAL_ARR)

    draw_box(c, B3_X, TOP_CY, B3_W, BOX_H, BOX_R,
             TEAL_FILL, TEAL_STR, TEAL_TITLE, TEAL_SUB,
             "SoH \u2192 EoL", "prediction")

    c.setFillColor(MUTED); c.setFont("Helvetica", FS_LABEL)
    c.drawCentredString(B3_X + B3_W / 2,
                        TOP_CY - BOX_H / 2 - 3.5 * mm,
                        "Simulation output")

    # ── Bottom row (coral) ───────────────────────────────────────────────────
    # Use ASCII-safe subtitle: "Phi = k3*delta^k4, k4 > 1"
    draw_box(c, B1_X, BOT_CY, B1_W, BOX_H, BOX_R,
             CORAL_FILL, CORAL_STR, CORAL_TITLE, CORAL_SUB,
             "Shi polynomial", SHI_SUB)

    arrow_h(c, B1_X + B1_W, B2_X, BOT_CY, CORAL_ARR)

    draw_box(c, B2_X, BOT_CY, B2_W, BOX_H, BOX_R,
             CORAL_FILL, CORAL_STR, CORAL_TITLE, CORAL_SUB,
             "Convex cost function", "convexity guaranteed")

    arrow_h(c, B2_X + B2_W, B3_X, BOT_CY, CORAL_ARR)

    draw_box(c, B3_X, BOT_CY, B3_W, BOX_H, BOX_R,
             CORAL_FILL, CORAL_STR, CORAL_TITLE, CORAL_SUB,
             GRADIENT_LABEL, "gradient")

    c.setFillColor(MUTED); c.setFont("Helvetica", FS_LABEL)
    c.drawCentredString(B3_X + B3_W / 2,
                        BOT_CY + BOX_H / 2 + 1.5 * mm,
                        "Optimization input")

    # ── Legend ───────────────────────────────────────────────────────────────
    swatch(c, IN_X,       LEG_Y, HexColor("#1D9E75"),
           "Physical reference path (Xu)")
    swatch(c, 62 * mm,    LEG_Y, HexColor("#D85A30"),
           "Gradient computation path (Shi)")
    swatch(c, 120 * mm,   LEG_Y, HexColor("#888780"),
           "Shared input")

    c.save()


# -- Fit check ---------------------------------------------------------------- #
BOXES = [
    ("input",     IN_W, "Rainflow extraction",  "cycles (\u03b4, \u03c3)"),
    ("xu",        B1_W, "Xu stress function",  XU_STRESS_SUB),
    ("fd",        B2_W, FD_LABEL,              "cyclic + calendar aging"),
    ("soh",       B3_W, "SoH \u2192 EoL",           "prediction"),
    ("shi",       B1_W, "Shi polynomial",      SHI_SUB),
    ("convex",    B2_W, "Convex cost function", "convexity guaranteed"),
    ("gradient",  B3_W, GRADIENT_LABEL,         "gradient"),
]


def check_fit(pad=2 * mm):
    """Warn if any label is wider than the box that holds it."""
    bad = []
    for tag, w, title, sub in BOXES:
        tw = seg_width(as_segs(title), "Helvetica-Bold", FS_TITLE)
        sw = seg_width(as_segs(sub), "Helvetica", FS_SUB)
        need = max(tw, sw)
        if need > w - pad:
            bad.append(f"  {tag}: needs {need / mm:.1f} mm in a "
                       f"{(w - pad) / mm:.1f} mm box")
    if bad:
        print("WARNING: text wider than its box")
        print("\n".join(bad))
    else:
        print("fit check: all labels inside their boxes")


# -- Output ------------------------------------------------------------------ #
def write_png(pdf_bytes: bytes, png_path: Path, dpi: int = DPI) -> bool:
    """Rasterise an in-memory PDF to PNG. PyMuPDF first, then Poppler's pdftoppm."""
    try:
        try:
            import pymupdf              # PyMuPDF 1.24.3 and later
        except ImportError:
            import fitz as pymupdf      # older releases expose it as fitz
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        doc.load_page(0).get_pixmap(dpi=dpi, alpha=False).save(str(png_path))
        doc.close()
        return True
    except ImportError:
        pass
    exe = shutil.which("pdftoppm")
    if exe is None:
        print("PNG not written. Install PyMuPDF with: pip install pymupdf")
        return False
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pdf = Path(tmp) / "page.pdf"
        tmp_pdf.write_bytes(pdf_bytes)
        subprocess.run([exe, "-png", "-r", str(dpi), "-singlefile",
                        str(tmp_pdf), str(png_path.with_suffix(""))], check=True)
    return True


if OUTPUT not in ("png", "pdf", "both"):
    raise ValueError(f'OUTPUT must be "png", "pdf" or "both", not {OUTPUT!r}')

check_fit()

OUT = Path(__file__).parent

pdf_buf = io.BytesIO()
build(pdf_buf)
pdf_bytes = pdf_buf.getvalue()

if OUTPUT in ("pdf", "both"):
    pdf_path = OUT / f"{STEM}.pdf"
    pdf_path.write_bytes(pdf_bytes)
    print(f"Saved {pdf_path.name}  ({len(pdf_bytes) // 1024} KB)")

if OUTPUT in ("png", "both"):
    png_path = OUT / f"{STEM}.png"
    if write_png(pdf_bytes, png_path):
        print(f"Saved {png_path.name}  ({DPI} dpi)")