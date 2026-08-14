"""
fig_coupling_tension.py
-----------------------
Produces Figure 3.1: Three-way coupling between battery size, operational
strategy, and degradation rate.

Uses ReportLab for direct PDF/PNG output — no matplotlib, clean vector output.
Fonts: Helvetica (built-in, visually equivalent to DejaVu Sans for diagrams).
Colors: TU Delft thesis palette.

Run: python fig_coupling_tension.py
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

from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.units import mm

# -- Output ------------------------------------------------------------------ #
OUTPUT = "png"                   # "png", "pdf" or "both"
STEM = "fig_coupling_tension"     # output filename without extension

# ── Palette (TU Delft) ────────────────────────────────────────────────────────
NAVY    = HexColor("#0C2340")
BLUE    = HexColor("#0076C2")
DARKRED = HexColor("#A50034")
NEUTRAL = HexColor("#404040")
WHITE   = white

# ── Page / figure dimensions ──────────────────────────────────────────────────
# Full A4 textwidth ≈ 160 mm; height just enough for the diagram.
FIG_W   = 160 * mm          # 160 mm wide
FIG_H   =  42 * mm          # 42 mm tall
DPI     = 300

# ── Box geometry (all in points; 1 mm = 2.835 pt) ────────────────────────────
# ReportLab y=0 is BOTTOM-LEFT. We work top-down by subtracting from FIG_H.
BOX_W   = 44 * mm
BOX_H   = 18 * mm
BOX_R   =  2 * mm           # corner radius
BOX_Y   = (FIG_H - BOX_H) / 2   # vertically centred

# Three box x-left edges, evenly distributed with gaps
GAP        = (FIG_W - 3 * BOX_W) / 4   # equal margin/gap
BOX_X      = [GAP, GAP * 2 + BOX_W, GAP * 3 + BOX_W * 2]
COLORS_BOX = [NAVY, BLUE, DARKRED]
TITLES     = ["Battery size",       "Operation strategy",   "Degradation rate"]
SUBS       = ["E, P capacity",      "Dispatch schedule",    "Capacity fade, fd"]

# Arrow geometry
FB_Y       = BOX_Y - 7 * mm    # how far below boxes the feedback arc runs
ARROW_HEAD = 2.5 * mm          # arrowhead size

# ── Helper: rounded rect (ReportLab has no built-in roundRect fill+stroke) ────
def rounded_rect(c: canvas.Canvas, x, y, w, h, r, fill_color):
    p = c.beginPath()
    p.moveTo(x + r, y)
    p.lineTo(x + w - r, y)
    p.arcTo(x + w - 2*r, y, x + w, y + 2*r, startAng=-90, extent=90)
    p.lineTo(x + w, y + h - r)
    p.arcTo(x + w - 2*r, y + h - 2*r, x + w, y + h, startAng=0, extent=90)
    p.lineTo(x + r, y + h)
    p.arcTo(x, y + h - 2*r, x + 2*r, y + h, startAng=90, extent=90)
    p.lineTo(x, y + r)
    p.arcTo(x, y, x + 2*r, y + 2*r, startAng=180, extent=90)
    p.close()
    c.setFillColor(fill_color)
    c.setStrokeColor(fill_color)
    c.drawPath(p, fill=1, stroke=0)


def arrowhead(c: canvas.Canvas, tip_x, tip_y, direction="right", size=ARROW_HEAD):
    """Filled triangle arrowhead."""
    if direction == "right":
        pts = [(tip_x, tip_y),
               (tip_x - size, tip_y + size * 0.55),
               (tip_x - size, tip_y - size * 0.55)]
    elif direction == "up":
        pts = [(tip_x, tip_y),
               (tip_x - size * 0.55, tip_y - size),
               (tip_x + size * 0.55, tip_y - size)]
    p = c.beginPath()
    p.moveTo(*pts[0])
    for pt in pts[1:]:
        p.lineTo(*pt)
    p.close()
    c.setFillColor(NEUTRAL)
    c.setStrokeColor(NEUTRAL)
    c.drawPath(p, fill=1, stroke=0)


def draw_dashed_line(c: canvas.Canvas, x1, y1, x2, y2,
                     dash=(3*mm, 2*mm), lw=0.8):
    c.setStrokeColor(NEUTRAL)
    c.setLineWidth(lw)
    c.setDash(*dash)
    c.line(x1, y1, x2, y2)
    c.setDash()   # reset


def draw_dashed_path(c: canvas.Canvas, pts, dash=(3*mm, 2*mm), lw=0.8):
    """Dashed polyline. One path, so the dash pattern runs on across corners."""
    c.setStrokeColor(NEUTRAL)
    c.setLineWidth(lw)
    c.setDash(*dash)
    p = c.beginPath()
    p.moveTo(*pts[0])
    for pt in pts[1:]:
        p.lineTo(*pt)
    c.drawPath(p, stroke=1, fill=0)
    c.setDash()


def draw_solid_line(c: canvas.Canvas, x1, y1, x2, y2, lw=1.0):
    c.setStrokeColor(NEUTRAL)
    c.setLineWidth(lw)
    c.setDash()
    c.line(x1, y1, x2, y2)


# ── Draw to a PDF buffer ──────────────────────────────────────────────────────
def build(buf):
    c = canvas.Canvas(buf, pagesize=(FIG_W, FIG_H))

    # White background
    c.setFillColor(white)
    c.rect(0, 0, FIG_W, FIG_H, fill=1, stroke=0)

    # ── Boxes ────────────────────────────────────────────────────────────────
    for i, (bx, col, title, sub) in enumerate(
            zip(BOX_X, COLORS_BOX, TITLES, SUBS)):
        rounded_rect(c, bx, BOX_Y, BOX_W, BOX_H, BOX_R, col)

        cx = bx + BOX_W / 2
        # Title
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(cx, BOX_Y + BOX_H * 0.60, title)
        # Subtitle
        c.setFillColor(HexColor("#c8d8e8") if col == NAVY else
                       HexColor("#cce4f7") if col == BLUE else
                       HexColor("#f0c0cc"))
        c.setFont("Helvetica", 8.5)
        c.drawCentredString(cx, BOX_Y + BOX_H * 0.28, sub)

    # ── Forward arrows ───────────────────────────────────────────────────────
    arrow_y = BOX_Y + BOX_H / 2
    for i in range(2):
        x_start = BOX_X[i] + BOX_W
        x_end   = BOX_X[i + 1] - ARROW_HEAD
        draw_solid_line(c, x_start, arrow_y, x_end, arrow_y, lw=1.2)
        arrowhead(c, BOX_X[i + 1], arrow_y, direction="right")

    # ── Economic feedback arc (dashed, below boxes) ──────────────────────────
    fb_y      = FB_Y
    left_x    = BOX_X[0] + 3 * mm
    right_x   = BOX_X[2] + BOX_W - 3 * mm
    box_bot   = BOX_Y

    # One continuous dashed path, so the dash pattern does not restart at each
    # corner and leave a gap where the arrowhead meets the line. The label box
    # below is painted over the centre of the bottom run.
    mid = FIG_W / 2
    draw_dashed_path(c, [(right_x, box_bot),
                         (right_x, fb_y),
                         (left_x,  fb_y),
                         (left_x,  box_bot - ARROW_HEAD)])
    arrowhead(c, left_x, box_bot, direction="up")

    # Feedback label
    c.setFont("Helvetica", 8.5)
    c.setFillColor(NEUTRAL)
    label_text = "Economic feedback"
    tw = c.stringWidth(label_text, "Helvetica", 8.5)
    pad_x, pad_y = 3 * mm, 1.5 * mm
    lx = mid - tw / 2 - pad_x
    ly = fb_y - 3 * mm
    c.setFillColor(white)
    c.setStrokeColor(white)
    c.rect(lx, ly, tw + 2 * pad_x, 6 * mm, fill=1, stroke=0)
    c.setFillColor(NEUTRAL)
    c.drawString(lx + pad_x, ly + pad_y, label_text)

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