"""
fig_coupling_tension.py
-----------------------
Produces Figure 3.1: Three-way coupling between battery size, operational
strategy, and degradation rate.

Uses ReportLab for direct PDF/PNG output — no matplotlib, clean vector output.
Fonts: Helvetica (built-in, visually equivalent to DejaVu Sans for diagrams).
Colors: TU Delft thesis palette.

Run: python fig_coupling_tension.py
Output: fig_coupling_tension.pdf  +  fig_coupling_tension.png  (300 dpi)
        saved next to this script.
"""

from __future__ import annotations
from pathlib import Path
import io

from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.units import mm, inch
from reportlab.lib import colors
from PIL import Image as PILImage

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

    # Right drop
    draw_dashed_line(c, right_x, box_bot, right_x, fb_y)
    # Bottom run (leave gap for label in centre)
    label_w = 32 * mm
    mid     = FIG_W / 2
    draw_dashed_line(c, right_x, fb_y, mid + label_w / 2, fb_y)
    draw_dashed_line(c, mid - label_w / 2, fb_y, left_x, fb_y)
    # Left rise (stop short for arrowhead)
    draw_dashed_line(c, left_x, fb_y, left_x, box_bot + ARROW_HEAD)
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


# ── Output ────────────────────────────────────────────────────────────────────
OUT = Path(__file__).parent

# PDF
pdf_buf = io.BytesIO()
build(pdf_buf)
pdf_bytes = pdf_buf.getvalue()
(OUT / "fig_coupling_tension.pdf").write_bytes(pdf_bytes)
print(f"Saved fig_coupling_tension.pdf  ({len(pdf_bytes)//1024} KB)")

# PNG via Pillow + pdf2image (poppler) if available, else via reportlab rasterise
try:
    from pdf2image import convert_from_bytes
    pages = convert_from_bytes(pdf_bytes, dpi=DPI)
    pages[0].save(str(OUT / "fig_coupling_tension.png"), "PNG")
    print(f"Saved fig_coupling_tension.png  (pdf2image, {DPI} dpi)")
except Exception:
    # Fallback: render to PNG directly with reportlab's renderPM
    try:
        from reportlab.graphics import renderPM
        from reportlab.graphics.shapes import Drawing
        # Re-render as PNG using ImageMagick via subprocess if available
        import subprocess, shutil
        if shutil.which("gs"):   # Ghostscript
            subprocess.run(
                ["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pngalpha",
                 f"-r{DPI}", f"-sOutputFile={OUT}/fig_coupling_tension.png",
                 str(OUT / "fig_coupling_tension.pdf")],
                check=True, capture_output=True
            )
            print(f"Saved fig_coupling_tension.png  (ghostscript, {DPI} dpi)")
        else:
            print("PNG skipped — install pdf2image+poppler or ghostscript for PNG export.")
            print("PDF is ready and can be included directly in LaTeX.")
    except Exception as e:
        print(f"PNG skipped ({e}). PDF is ready.")
