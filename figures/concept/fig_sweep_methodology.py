"""
fig_sweep_methodology.py
------------------------
Figure 3.X: Two-stage parameter sweep methodology diagram.

Uses ReportLab only — no matplotlib.
All coordinates transcribed exactly from parameter_sweep_methodology_v2.svg
(viewBox 680x600). Scale = 160mm / 680px.

Run: python fig_sweep_methodology.py
Output: fig_sweep_methodology.pdf  +  fig_sweep_methodology.png
"""

from __future__ import annotations
from pathlib import Path
import io

from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.units import mm

# ── Palette (from SVG computed styles) ────────────────────────────────────────
GRAY_F   = HexColor("#F1EFE8")   # gray box fill
GRAY_S   = HexColor("#5F5E5A")   # gray stroke
GRAY_T   = HexColor("#444441")   # gray title
GRAY_ST  = HexColor("#5F5E5A")   # gray subtitle

PURP_F   = HexColor("#EEEDFE")   # purple box fill  (simulation step)
PURP_S   = HexColor("#534AB7")   # purple stroke
PURP_T   = HexColor("#3C3489")   # purple title
PURP_ST  = HexColor("#534AB7")   # purple subtitle

TEAL_F   = HexColor("#E1F5EE")   # teal box fill  (output/optimum)
TEAL_S   = HexColor("#0F6E56")   # teal stroke
TEAL_T   = HexColor("#085041")   # teal title
TEAL_ST  = HexColor("#0F6E56")   # teal subtitle

CORAL_F  = HexColor("#FAECE7")   # coral box fill (sweep series)
CORAL_S  = HexColor("#993C1D")   # coral stroke
CORAL_T  = HexColor("#712B13")   # coral title
CORAL_ST = HexColor("#993C1D")   # coral subtitle

MUTED    = HexColor("#5F5E5A")   # floating labels, legend text
DIVIDER  = HexColor("#888780")   # centre divider line

# ── Canvas ────────────────────────────────────────────────────────────────────
SCALE  = 160 / 680               # mm per SVG px
FIG_W  = 160 * mm
FIG_H  = 600 * SCALE * mm
DPI    = 300


def s(px):  return px * SCALE * mm
def sy(px): return FIG_H - s(px)


# ── Helpers ───────────────────────────────────────────────────────────────────
def rr(c, x, y_top, w, h, r, fill, stroke, sw=0.4):
    X, W, H, R = s(x), s(w), s(h), s(r)
    Y = sy(y_top + h)
    p = c.beginPath()
    p.moveTo(X+R, Y); p.lineTo(X+W-R, Y)
    p.arcTo(X+W-2*R, Y,       X+W,   Y+2*R, startAng=-90, extent=90)
    p.lineTo(X+W, Y+H-R)
    p.arcTo(X+W-2*R, Y+H-2*R, X+W,   Y+H,   startAng=0,   extent=90)
    p.lineTo(X+R, Y+H)
    p.arcTo(X,    Y+H-2*R,    X+2*R, Y+H,   startAng=90,  extent=90)
    p.lineTo(X, Y+R)
    p.arcTo(X,    Y,           X+2*R, Y+2*R, startAng=180, extent=90)
    p.close()
    c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(sw)
    c.drawPath(p, fill=1, stroke=1)


def two_line_box(c, x, y_top, w, h, r, fill, stroke, sw,
                 title, tc, tf, ts_size,
                 sub,   sc, sub_size):
    rr(c, x, y_top, w, h, r, fill, stroke, sw)
    cx, cy = x + w/2, y_top + h/2
    # title at 38% from top, sub at 68%
    c.setFillColor(tc); c.setFont(tf, ts_size)
    c.drawCentredString(s(cx), sy(y_top + h*0.38), title)
    c.setFillColor(sc); c.setFont("Helvetica", sub_size)
    c.drawCentredString(s(cx), sy(y_top + h*0.68), sub)


def three_line_box(c, x, y_top, w, h, r, fill, stroke, sw,
                   title, tc, tf, ts_size,
                   sub1, sub2, sc, sub_size):
    """Box with title + two subtitle lines (coral sweep boxes)."""
    rr(c, x, y_top, w, h, r, fill, stroke, sw)
    cx = x + w/2
    c.setFillColor(tc); c.setFont(tf, ts_size)
    c.drawCentredString(s(cx), sy(y_top + h*0.30), title)
    c.setFillColor(sc); c.setFont("Helvetica", sub_size)
    c.drawCentredString(s(cx), sy(y_top + h*0.56), sub1)
    c.drawCentredString(s(cx), sy(y_top + h*0.78), sub2)


def arrow_v_down(c, x, y1, y2, color, lw=1.0):
    """Arrow pointing DOWN (y2 > y1 in SVG coords)."""
    head = s(5); hw = head * 0.42
    X, Y1, Y2 = s(x), sy(y1), sy(y2)
    tip = Y2                            # lower in RL = lower on page
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(lw)
    c.line(X, Y1, X, tip + head*0.7)
    p = c.beginPath()
    p.moveTo(X, tip); p.lineTo(X-hw, tip+head); p.lineTo(X+hw, tip+head); p.close()
    c.drawPath(p, fill=1, stroke=0)


def arrow_v_up(c, x, y1, y2, color, lw=1.0):
    """Arrow pointing UP (y2 < y1 in SVG coords)."""
    head = s(5); hw = head * 0.42
    X, Y1, Y2 = s(x), sy(y1), sy(y2)
    tip = Y2                            # higher in RL = higher on page
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(lw)
    c.line(X, Y1, X, tip - head*0.7)
    p = c.beginPath()
    p.moveTo(X, tip); p.lineTo(X-hw, tip-head); p.lineTo(X+hw, tip-head); p.close()
    c.drawPath(p, fill=1, stroke=0)


def arrow_h_right(c, x1, x2, y, color, lw=1.0):
    """Arrow pointing RIGHT."""
    head = s(5); hw = head * 0.42
    X1, X2, Y = s(x1), s(x2), sy(y)
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(lw)
    c.line(X1, Y, X2 - head*0.7, Y)
    p = c.beginPath()
    p.moveTo(X2, Y); p.lineTo(X2-head, Y+hw); p.lineTo(X2-head, Y-hw); p.close()
    c.drawPath(p, fill=1, stroke=0)


def plain_line(c, x1, y1, x2, y2, color, lw=0.9):
    c.setStrokeColor(color); c.setLineWidth(lw); c.setDash()
    c.line(s(x1), sy(y1), s(x2), sy(y2))


def txt(c, x, y_svg, text, font, size, color, anchor="centre"):
    c.setFillColor(color); c.setFont(font, size)
    if anchor == "centre":
        c.drawCentredString(s(x), sy(y_svg), text)
    else:
        c.drawString(s(x), sy(y_svg), text)


def swatch(c, x, y_svg, color, label):
    SW, SH = s(12), s(12)
    X, Y = s(x), sy(y_svg + 12)
    c.setFillColor(color); c.roundRect(X, Y, SW, SH, s(2), fill=1, stroke=0)
    c.setFillColor(MUTED); c.setFont("Helvetica", 7.5)
    c.drawString(X + SW + s(4), Y + s(2), label)


# ── Build ─────────────────────────────────────────────────────────────────────
def build(buf):
    c = rl_canvas.Canvas(buf, pagesize=(FIG_W, FIG_H))
    c.setFillColor(white); c.rect(0, 0, FIG_W, FIG_H, fill=1, stroke=0)

    LW = 0.9

    # ── Stage headings ────────────────────────────────────────────────────
    txt(c, 160, 28, "Stage 1: battery sizing",   "Helvetica-Bold", 9,  MUTED)
    txt(c, 504, 28, "Stage 2: operating window", "Helvetica-Bold", 9,  MUTED)

    # ── Centre divider (dashed) ───────────────────────────────────────────
    c.setStrokeColor(HexColor("#CCCCCC")); c.setLineWidth(0.4); c.setDash(4, 3)
    c.line(s(338), sy(14), s(338), sy(588))
    c.setDash()

    # ════════════════════════════════════════════════════════════════════
    # STAGE 1 — LEFT COLUMN  (x center=160, box w=232, x=44..276)
    # ════════════════════════════════════════════════════════════════════

    # Wind + price (gray, h=44, y=44..88)
    two_line_box(c, 44, 44, 232, 44, 6, GRAY_F, GRAY_S, 0.4,
                 "Wind + price time series", GRAY_T, "Helvetica-Bold", 9,
                 "ERA5 90 m, DK1 2022",      GRAY_ST, 7.5)
    arrow_v_down(c, 160, 88, 108, MUTED, LW)

    # Coarse E×P grid (purple, h=56, y=108..164)
    two_line_box(c, 44, 108, 232, 56, 6, PURP_F, PURP_S, 0.4,
                 "Coarse E \u00d7 P grid",      PURP_T, "Helvetica-Bold", 9,
                 "11 \u00d7 8 pts, E: 150\u20131500 MWh", PURP_ST, 7.5)
    arrow_v_down(c, 160, 164, 184, MUTED, LW)

    # 20-yr multi-year loop (purple, h=56, y=184..240)
    two_line_box(c, 44, 184, 232, 56, 6, PURP_F, PURP_S, 0.4,
                 "20-yr multi-year loop",          PURP_T, "Helvetica-Bold", 9,
                 "LP + rainflow + deg., 3 scenarios", PURP_ST, 7.5)
    arrow_v_down(c, 160, 240, 260, MUTED, LW)

    # Locate degraded region (teal, h=44, y=260..304)
    two_line_box(c, 44, 260, 232, 44, 6, TEAL_F, TEAL_S, 0.4,
                 "Locate degraded region", TEAL_T, "Helvetica-Bold", 9,
                 "Coarse grid maximum",    TEAL_ST, 7.5)
    arrow_v_down(c, 160, 304, 324, MUTED, LW)

    # Refined E×P grid (purple, h=56, y=324..380)
    two_line_box(c, 44, 324, 232, 56, 6, PURP_F, PURP_S, 0.4,
                 "Refined E \u00d7 P grid",     PURP_T, "Helvetica-Bold", 9,
                 "10 \u00d7 6 pts, E: 300\u2013600 MWh", PURP_ST, 7.5)
    arrow_v_down(c, 160, 380, 400, MUTED, LW)

    # Grid optimum + quadratic fit (teal, h=56, y=400..456)
    two_line_box(c, 44, 400, 232, 56, 6, TEAL_F, TEAL_S, 0.4,
                 "Grid optimum + quadratic fit", TEAL_T, "Helvetica-Bold", 9,
                 "Xu optimum \u2192 Stage 2 anchor", TEAL_ST, 7.5)

    # ── Handoff arrow (276,428) → (354,428) ──────────────────────────────
    arrow_h_right(c, 276, 354, 428, MUTED, LW)
    txt(c, 315, 418, "fixed",   "Helvetica", 7.5, MUTED)
    txt(c, 315, 442, "(E*, P*)", "Helvetica", 7.5, MUTED)

    # ════════════════════════════════════════════════════════════════════
    # STAGE 2 — RIGHT COLUMN  (x center=492, box w=268, x=358..626)
    # ════════════════════════════════════════════════════════════════════

    # Fixed battery size (gray, h=56, y=400..456)
    two_line_box(c, 358, 400, 268, 56, 6, GRAY_F, GRAY_S, 0.4,
                 "Fixed battery size",    GRAY_T, "Helvetica-Bold", 9,
                 "E = 475 MWh, P = 150 MW", GRAY_ST, 7.5)

    # Fork from fixed box top (492,400) up to y=376, then split to 420 and 564
    plain_line(c, 492, 400, 492, 376, MUTED, LW)  # vertical stem up
    plain_line(c, 420, 376, 564, 376, MUTED, LW)  # horizontal fork
    arrow_v_up(c, 420, 376, 340, MUTED, LW)        # left arm → width series
    arrow_v_up(c, 564, 376, 340, MUTED, LW)        # right arm → center series

    # Width series (coral, h=72, y=268..340, x=358..482)
    three_line_box(c, 358, 268, 124, 72, 6, CORAL_F, CORAL_S, 0.4,
                   "Width series", CORAL_T, "Helvetica-Bold", 9,
                   "Center = 0.50", "d \u2208 {0.4, 0.6, 0.8, 1.0}",
                   CORAL_ST, 7.5)

    # Center series (coral, h=72, y=268..340, x=502..626)
    three_line_box(c, 502, 268, 124, 72, 6, CORAL_F, CORAL_S, 0.4,
                   "Center series", CORAL_T, "Helvetica-Bold", 9,
                   "Width = 0.80", "\u03c3\u0304 \u2208 {0.40 \u2026 0.60}",
                   CORAL_ST, 7.5)

    # Merge arms: both series tops (420,268) and (564,268) → up to y=244 → merge → arrow up to sim loop
    plain_line(c, 420, 268, 420, 244, MUTED, LW)
    plain_line(c, 564, 268, 564, 244, MUTED, LW)
    plain_line(c, 420, 244, 564, 244, MUTED, LW)
    arrow_v_up(c, 492, 244, 224, MUTED, LW)

    # 20-yr simulation loop Stage 2 (purple, h=56, y=168..224)
    two_line_box(c, 358, 168, 268, 56, 6, PURP_F, PURP_S, 0.4,
                 "20-yr simulation loop",          PURP_T, "Helvetica-Bold", 9,
                 "Fixed (E, P), per window, Xu + Shi", PURP_ST, 7.5)
    arrow_v_up(c, 492, 168, 148, MUTED, LW)

    # Window sensitivity results (teal, h=56, y=92..148)
    two_line_box(c, 358, 92, 268, 56, 6, TEAL_F, TEAL_S, 0.4,
                 "Window sensitivity results",       TEAL_T, "Helvetica-Bold", 9,
                 "Width vs center effects on NPV and fd", TEAL_ST, 7.5)

    # ── Legend ────────────────────────────────────────────────────────────
    swatch(c,  44, 488, HexColor("#AFA9EC"), "Simulation step")
    swatch(c, 168, 488, HexColor("#5DCAA5"), "Output / optimum")
    swatch(c, 308, 488, HexColor("#F0997B"), "Window sweep series")
    swatch(c, 494, 488, HexColor("#B4B2A9"), "Fixed input")

    c.save()


# ── Save ──────────────────────────────────────────────────────────────────────
OUT = Path(__file__).parent

pdf_buf = io.BytesIO()
build(pdf_buf)
pdf_bytes = pdf_buf.getvalue()
(OUT / "fig_sweep_methodology.pdf").write_bytes(pdf_bytes)
print(f"Saved fig_sweep_methodology.pdf  ({len(pdf_bytes)//1024} KB)")

try:
    from pdf2image import convert_from_bytes
    pages = convert_from_bytes(pdf_bytes, dpi=DPI)
    pages[0].save(str(OUT / "fig_sweep_methodology.png"), "PNG")
    print(f"Saved fig_sweep_methodology.png  ({DPI} dpi)")
except Exception as e:
    print(f"PNG skipped ({e}). PDF is ready for LaTeX.")
