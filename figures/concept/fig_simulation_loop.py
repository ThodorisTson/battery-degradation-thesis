"""
fig_simulation_loop.py
----------------------
Multi-year simulation loop structure (thesis label fig:simulation_loop).

ReportLab only, no matplotlib.

Coordinate system: ReportLab y=0 is BOTTOM-LEFT.
Design coordinates below are in SVG px, converted with s() and sy().

Run: python fig_simulation_loop.py
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
OUTPUT = "png"                  # "png", "pdf" or "both"
STEM = "fig_simulation_loop"     # output filename without extension

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE    = HexColor("#0076C2")   # LP dispatch (TU Delft blue)
GRAY_F  = HexColor("#F1EFE8")   # gray box fill
GRAY_S  = HexColor("#5F5E5A")   # gray box stroke / arrow
GRAY_T  = HexColor("#444441")   # gray box title text
GRAY_ST = HexColor("#5F5E5A")   # gray box subtitle text
TEAL_F  = HexColor("#E1F5EE")   # two-path fill
TEAL_S  = HexColor("#0F6E56")   # two-path stroke / arrow
TEAL_T  = HexColor("#085041")   # two-path title
CORAL   = HexColor("#D85A30")   # battery replacement / yes arrow
MUTED   = HexColor("#5F5E5A")   # legend + floating labels
DASHED  = HexColor("#888780")   # outer rect

# ── Canvas ────────────────────────────────────────────────────────────────────
# SVG viewBox 680×530 → scale to 160mm wide → scale = 160/680
SCALE   = 160 / 680             # mm per SVG px
FIG_W   = 160 * mm
FIG_H   = 530 * SCALE * mm
DPI     = 300

def s(px):
    """Convert SVG px coordinate to ReportLab points (via mm)."""
    return px * SCALE * mm

def sy(px):
    """Convert SVG y (top-left origin) to ReportLab y (bottom-left origin)."""
    return FIG_H - s(px)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def rr(c, x, y_svg, w, h, r, fill, stroke, sw=0.4):
    """Rounded rect. x,y_svg,w,h,r all in SVG px; converted internally."""
    X, W, H, R = s(x), s(w), s(h), s(r)
    Y = sy(y_svg + h)           # ReportLab y is bottom of box
    p = c.beginPath()
    p.moveTo(X + R, Y)
    p.lineTo(X + W - R, Y)
    p.arcTo(X + W - 2*R, Y,           X + W,     Y + 2*R, startAng=-90, extent=90)
    p.lineTo(X + W, Y + H - R)
    p.arcTo(X + W - 2*R, Y + H - 2*R, X + W,     Y + H,   startAng=0,   extent=90)
    p.lineTo(X + R, Y + H)
    p.arcTo(X,       Y + H - 2*R,     X + 2*R,   Y + H,   startAng=90,  extent=90)
    p.lineTo(X, Y + R)
    p.arcTo(X,       Y,               X + 2*R,   Y + 2*R, startAng=180, extent=90)
    p.close()
    c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(sw)
    c.drawPath(p, fill=1, stroke=1)


def txt(c, x_svg, y_svg, text, font, size, color, anchor="centre"):
    X, Y = s(x_svg), sy(y_svg)
    c.setFillColor(color); c.setFont(font, size)
    if anchor == "centre":
        c.drawCentredString(X, Y, text)
    elif anchor == "left":
        c.drawString(X, Y, text)


def box(c, x, y_top, w, h, r, fill, stroke, sw,
        title, title_col, title_font, title_size,
        sub, sub_col, sub_size):
    """Draw rounded box with title and subtitle centred inside."""
    rr(c, x, y_top, w, h, r, fill, stroke, sw)
    cx = x + w / 2
    cy_title = y_top + h * 0.38   # title sits at ~38% from top
    cy_sub   = y_top + h * 0.68   # subtitle at ~68% from top
    txt(c, cx, cy_title, title, title_font, title_size, title_col)
    txt(c, cx, cy_sub,   sub,   "Helvetica",  sub_size, sub_col)


def arrow_h(c, x1, x2, y_svg, color, lw=1.0):
    """Horizontal arrow in SVG coordinates."""
    head = s(6); hw = head * 0.4
    X1, X2, Y = s(x1), s(x2), sy(y_svg)
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(lw)
    c.line(X1, Y, X2 - head * 0.7, Y)
    p = c.beginPath()
    p.moveTo(X2, Y); p.lineTo(X2 - head, Y + hw); p.lineTo(X2 - head, Y - hw); p.close()
    c.drawPath(p, fill=1, stroke=0)


def arrow_v(c, x_svg, y1, y2, color, lw=1.0):
    """Vertical arrow (pointing down if y2 > y1 in SVG coords)."""
    head = s(6); hw = head * 0.4
    X = s(x_svg)
    Y1, Y2 = sy(y1), sy(y2)
    # Y2 < Y1 in RL coords when pointing down in SVG
    tip_y = min(Y1, Y2)
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(lw)
    c.line(X, max(Y1, Y2), X, tip_y + head * 0.7)
    p = c.beginPath()
    p.moveTo(X, tip_y); p.lineTo(X - hw, tip_y + head); p.lineTo(X + hw, tip_y + head); p.close()
    c.drawPath(p, fill=1, stroke=0)


def arrow_up(c, x_svg, y1_svg, y2_svg, color, lw=1.0):
    """Vertical arrow pointing UP (y2_svg < y1_svg)."""
    head = s(6); hw = head * 0.4
    X = s(x_svg)
    Y1, Y2 = sy(y1_svg), sy(y2_svg)   # Y2 > Y1 in RL (higher on page)
    tip_y = Y2
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(lw)
    c.line(X, Y1, X, tip_y - head * 0.7)
    p = c.beginPath()
    p.moveTo(X, tip_y); p.lineTo(X - hw, tip_y - head); p.lineTo(X + hw, tip_y - head); p.close()
    c.drawPath(p, fill=1, stroke=0)


def line(c, x1, y1, x2, y2, color, lw=1.0, dash=None):
    """Plain line in SVG coordinates."""
    c.setStrokeColor(color); c.setLineWidth(lw)
    if dash:
        c.setDash(*dash)
    else:
        c.setDash()
    c.line(s(x1), sy(y1), s(x2), sy(y2))
    c.setDash()


def diamond(c, cx, cy, hw, hh, fill, stroke, sw=0.9):
    """Diamond shape centred at (cx,cy) SVG coords."""
    CX, CY = s(cx), sy(cy)
    HW, HH = s(hw), s(hh)
    p = c.beginPath()
    p.moveTo(CX,      CY + HH)   # top tip
    p.lineTo(CX + HW, CY)        # right tip
    p.lineTo(CX,      CY - HH)   # bottom tip
    p.lineTo(CX - HW, CY)        # left tip
    p.close()
    c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(sw)
    c.drawPath(p, fill=1, stroke=1)


def swatch(c, x, y_svg, color, label):
    SW, SH = s(12), s(12)
    X, Y = s(x), sy(y_svg + 12)
    c.setFillColor(color); c.roundRect(X, Y, SW, SH, s(2), fill=1, stroke=0)
    c.setFillColor(MUTED); c.setFont("Helvetica", 7)
    c.drawString(X + SW + s(4), Y + s(2), label)



def rich_left(c, x_svg, y_svg, segs):
    """Left-anchored rich text at SVG coords.
    segs: list of (text, font, size_pt, color, style); style in {'', 'sub', 'bar'}.
    'bar' overlines the text (E-bar, P-bar); 'sub' draws it as a subscript.
    Sizes are absolute points, like the box text."""
    x = s(x_svg); Y = sy(y_svg)
    for text, font, size, color, style in segs:
        c.setFillColor(color)
        if style == "sub":
            ss = size * 0.72
            c.setFont(font, ss)
            c.drawString(x, Y - size * 0.20, text)
            x += stringWidth(text, font, ss)
        elif style == "bar":
            c.setFont(font, size)
            c.drawString(x, Y, text)
            w = stringWidth(text, font, size)
            inset = size * 0.06
            c.setStrokeColor(color); c.setLineWidth(size * 0.11); c.setLineCap(1)
            yb = Y + size * 0.84
            c.line(x + inset, yb, x + w - inset, yb)
            c.setLineCap(0)
            x += w
        else:
            c.setFont(font, size)
            c.drawString(x, Y, text)
            x += stringWidth(text, font, size)
    return x


def rich_width(segs):
    """Width in points of a rich_left segment list."""
    return sum(stringWidth(text, font, size * 0.72 if style == "sub" else size)
               for text, font, size, _col, style in segs)


def rich_centre(c, cx_svg, y_svg, segs):
    """Centred version of rich_left. cx_svg is an SVG x coordinate."""
    x_svg = cx_svg - rich_width(segs) / (SCALE * mm) / 2
    return rich_left(c, x_svg, y_svg, segs)


# ── Build PDF ─────────────────────────────────────────────────────────────────
def build(buf):
    c = rl_canvas.Canvas(buf, pagesize=(FIG_W, FIG_H))
    c.setFillColor(white)
    c.rect(0, 0, FIG_W, FIG_H, fill=1, stroke=0)

    LW = 1.0   # standard line width

    # ── Outer dashed rectangle ────────────────────────────────────────────
    c.setStrokeColor(DASHED); c.setLineWidth(0.6); c.setDash(4, 3)
    c.roundRect(s(22), sy(22+476), s(636), s(476), s(8), fill=0, stroke=1)
    c.setDash()

    # "repeat for 20 years" label on top edge
    c.setFillColor(white)
    c.roundRect(s(256), sy(14+18), s(142), s(18), s(3), fill=1, stroke=0)
    txt(c, 327, 23, "repeat for 20 years", "Helvetica", 7.5, MUTED)

    # ── Row 1: LP dispatch → SoC → Rainflow ──────────────────────────────
    # LP dispatch (x=52, y=54, w=130, h=52)
    rr(c, 52, 54, 130, 52, 7, BLUE, BLUE, 0.5)
    txt(c, 117, 75, "LP dispatch",     "Helvetica-Bold", 9,   white)
    txt(c, 117, 94, "maximize revenue", "Helvetica",      7.5, HexColor("#cce4f7"))

    arrow_h(c, 182, 210, 80, GRAY_S, LW)

    # SoC trajectory (x=210, y=54, w=130, h=52)
    box(c, 210, 54, 130, 52, 7, GRAY_F, GRAY_S, 0.5,
        "SoC trajectory", GRAY_T, "Helvetica-Bold", 9,
        "8,760 hourly steps", GRAY_ST, 7.5)

    arrow_h(c, 340, 368, 80, GRAY_S, LW)

    # Rainflow counting (x=368, y=54, w=150, h=52)
    box(c, 368, 54, 150, 52, 7, GRAY_F, GRAY_S, 0.5,
        "Rainflow counting", GRAY_T, "Helvetica-Bold", 9,
        "extract cycles (\u03b4, \u03c3)", GRAY_ST, 7.5)

    # Rainflow right → step down to Two-path top
    line(c, 518, 80, 550, 80,  GRAY_S, LW)
    arrow_v(c, 550, 80, 170, GRAY_S, LW)

    # ── Right column: Two-path → SoH update ──────────────────────────────
    # Two-path (x=454, y=170, w=196, h=52)
    rr(c, 454, 170, 196, 52, 7, TEAL_F, TEAL_S, 0.7)
    txt(c, 552, 191, "Two-path architecture", "Helvetica-Bold", 9,   TEAL_T)
    txt(c, 552, 209, "Xu reports \u00b7 Shi gradient", "Helvetica",      7.5, TEAL_S)

    arrow_v(c, 552, 222, 298, TEAL_S, LW)

    # SoH update (x=454, y=298, w=196, h=52). Drawn as box plus rich subtitle so
    # that f_d carries its subscript, as in Chapters 2 and 3.
    rr(c, 454, 298, 196, 52, 7, GRAY_F, GRAY_S, 0.5)
    txt(c, 552, 318, "SoH update", "Helvetica-Bold", 9, GRAY_T)
    rich_centre(c, 552, 337, [
        ("accumulate f", "Helvetica", 7.5, GRAY_ST, ""),
        ("d",            "Helvetica", 7.5, GRAY_ST, "sub"),
        (" \u2192 SoH",  "Helvetica", 7.5, GRAY_ST, ""),
    ])

    # SoH update left edge (454,324) → horizontal to diamond right tip (390,324)
    arrow_h(c, 454, 390, 324, GRAY_S, LW)

    # ── Diamond: center=(322,324), hw=66, hh=38 ───────────────────────────
    # right-tip=(388,324) left-tip=(256,324) top=(322,286) bottom=(322,362)
    diamond(c, 322, 324, 66, 38, white, GRAY_S, 0.9)
    txt(c, 322, 321, "SoH <70%", "Helvetica-Bold", 9, GRAY_T)

    # ── YES branch: bottom-tip (322,362) → down → battery replacement ────
    arrow_v(c, 322, 362, 410, CORAL, LW)
    txt(c, 332, 388, "yes", "Helvetica", 7.5, CORAL, anchor="left")

    # Battery replacement (x=164, y=410, w=316, h=52)
    rr(c, 164, 410, 316, 52, 7, CORAL, CORAL, 0.5)
    txt(c, 322, 428, "Battery replacement", "Helvetica-Bold", 9, white)
    txt(c, 322, 447, "SoH resets to 1.0  \u00b7  replacement cost subtracted from NPV",
        "Helvetica", 7.5, HexColor("#fde8e0"))

    # YES loop back: battery left (164,436) → left to x=36 → up to LP left-mid (36,80) → right into LP
    line(c, 164, 436,  36, 436, BLUE, LW)
    line(c,  36, 436,  36,  80, BLUE, LW)
    arrow_h(c, 36, 52, 80, BLUE, LW)

    # ── NO branch: left-tip (256,324) → left to x=117 → up to LP bottom-mid (117,106)
    line(c, 256, 324, 117, 324, GRAY_S, LW)
    arrow_up(c, 117, 324, 106, GRAY_S, LW)
    txt(c, 185, 316, "no", "Helvetica", 7.5, GRAY_ST, anchor="left")

    # ── Annotation: which LP inputs change between years (Jenna comment 2) ─
    BORD = HexColor("#C9C8C2"); LEAD = HexColor("#B4B3AC")
    INKA = HexColor("#444441"); GRYA = HexColor("#5F5E5A")
    AB, AR, ASZ = "Helvetica-Bold", "Helvetica", 8.0

    # callout box in the empty interior of the loop
    rr(c, 124, 150, 262, 80, 6, white, BORD, 0.7)
    # dotted leader from LP-dispatch bottom down to the callout
    c.setStrokeColor(LEAD); c.setLineWidth(0.6); c.setDash(2, 3)
    c.line(s(150), sy(106), s(150), sy(150)); c.setDash()
    c.setFillColor(LEAD); c.circle(s(150), sy(150), s(1.6), fill=1, stroke=0)
    # heading + the fixed / varying contrast
    txt(c, 137, 171, "LP inputs, year k", AB, 8.5, INKA, anchor="left")
    rich_left(c, 137, 193, [
        ("same every year:  ", AR, ASZ, GRYA, ""),
        ("wind, price, ",      AR, ASZ, INKA, ""),
        ("P",                  AR, ASZ, INKA, "bar"),
    ])
    rich_left(c, 137, 214, [
        ("changes every year:  ", AR, ASZ, GRYA, ""),
        ("E",           AB, ASZ, BLUE, "bar"),
        ("k",           AB, ASZ, BLUE, "sub"),
        (" = ",         AB, ASZ, BLUE, ""),
        ("E",           AB, ASZ, BLUE, "bar"),
        (" \u00b7 SoH", AB, ASZ, BLUE, ""),
        ("k\u22121",    AB, ASZ, BLUE, "sub"),
    ])


    # ── Legend ────────────────────────────────────────────────────────────
    swatch(c,  52, 506, BLUE,   "LP optimization")
    swatch(c, 196, 506, DASHED, "Simulation step")
    swatch(c, 340, 506, TEAL_S, "Degradation model")
    swatch(c, 494, 506, CORAL,  "Replacement event")

    c.save()


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