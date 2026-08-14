"""
fig_rainflow_dual_path.py
-------------------------
Figure 3.5: Two-path degradation architecture.
ReportLab only — no matplotlib.

Run: python fig_rainflow_dual_path.py
Output: fig_rainflow_dual_path.pdf + fig_rainflow_dual_path.png
"""

from __future__ import annotations
from pathlib import Path
import io

from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.units import mm

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

# Input box — enough left margin so text doesn't clip
IN_X    =  6 * mm
IN_W    = 26 * mm
IN_CY   = (TOP_CY + BOT_CY) / 2
IN_H    = 14 * mm

# Content boxes: start after input box + fork gap
# B1 starts at 38mm, B2 at 70mm (wider), B3 at 105mm
B1_X    = 38 * mm;  B1_W = 28 * mm
B2_X    = 70 * mm;  B2_W = 31 * mm
B3_X    = 105 * mm; B3_W = 28 * mm

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


def draw_box(c, x, cy, w, h, r, fill, stroke,
             title_col, sub_col, title, subtitle):
    y = cy - h / 2
    rounded_rect(c, x, y, w, h, r, fill, stroke)
    cx = x + w / 2
    c.setFillColor(title_col)
    c.setFont("Helvetica-Bold", FS_TITLE)
    c.drawCentredString(cx, cy + 1.6 * mm, title)
    c.setFillColor(sub_col)
    c.setFont("Helvetica", FS_SUB)
    c.drawCentredString(cx, cy - 2.4 * mm, subtitle)


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
             "Xu stress function", "S\u03b4(\u03b4) \u00b7 S\u03c3(\u03c3) per cycle")

    arrow_h(c, B1_X + B1_W, B2_X, TOP_CY, TEAL_ARR)

    draw_box(c, B2_X, TOP_CY, B2_W, BOX_H, BOX_R,
             TEAL_FILL, TEAL_STR, TEAL_TITLE, TEAL_SUB,
             "fd accumulation", "cyclic + calendar aging")

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
             "Shi polynomial", "\u03a6 = k3\u00b7\u03b4^k4, k4 > 1")

    arrow_h(c, B1_X + B1_W, B2_X, BOT_CY, CORAL_ARR)

    draw_box(c, B2_X, BOT_CY, B2_W, BOX_H, BOX_R,
             CORAL_FILL, CORAL_STR, CORAL_TITLE, CORAL_SUB,
             "Convex cost function", "convexity guaranteed")

    arrow_h(c, B2_X + B2_W, B3_X, BOT_CY, CORAL_ARR)

    draw_box(c, B3_X, BOT_CY, B3_W, BOX_H, BOX_R,
             CORAL_FILL, CORAL_STR, CORAL_TITLE, CORAL_SUB,
             "\u2202Deg/\u2202DoD", "gradient")

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


# ── Save ──────────────────────────────────────────────────────────────────────
OUT = Path(__file__).parent

pdf_buf = io.BytesIO()
build(pdf_buf)
pdf_bytes = pdf_buf.getvalue()
(OUT / "fig_rainflow_dual_path.pdf").write_bytes(pdf_bytes)
print(f"Saved fig_rainflow_dual_path.pdf  ({len(pdf_bytes)//1024} KB)")

try:
    from pdf2image import convert_from_bytes
    pages = convert_from_bytes(pdf_bytes, dpi=DPI)
    pages[0].save(str(OUT / "fig_rainflow_dual_path.png"), "PNG")
    print(f"Saved fig_rainflow_dual_path.png  ({DPI} dpi)")
except Exception as e:
    print(f"PNG skipped ({e}). PDF is ready for LaTeX.")
