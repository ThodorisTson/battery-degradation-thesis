"""
fig_sweep_methodology.py
------------------------
Two-stage parameter sweep structure (thesis label fig:sweep_methodology).

ReportLab only, no matplotlib.
Coordinates follow the source SVG frame (viewBox 680 wide). Scale = 160mm/680px.

Every number in the boxes is taken from Sections 3.7.1 and 3.7.2 and from the
Stage 1 result in Section 4.4.1. If the sweep configuration changes, change it
here as well; a methodology diagram that disagrees with its own method section
is worse than no diagram.

Run: python fig_sweep_methodology.py
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
OUTPUT = "png"                   # "png", "pdf" or "both"
STEM = "fig_sweep_methodology"    # output filename without extension

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

# Labels carrying subscripts, as (string, style) segments.
# delta_max is the window width of Section 3.7.2, not the DoD parameter d and
# not the per-cycle amplitude delta_i. The window center is the parameter
# (sigma_max + sigma_min)/2, which is set before the year is solved; sigma-bar
# is the resulting profile mean, a dispatch output, so it is not used here.
# The full width set is {0.40, 0.60, 0.80, 1.00}; the box shows the range in the
# same form as the center series, because the full list does not fit at 7.5 pt.
# Section 3.7.2 lists all four, and notes that the 0.40 width is skipped because
# its polynomial fit returns k4 = 0.908, which is not convex.
WIDTH_SET     = [("\u03b4", ""), ("max", "sub"), (" \u2208 {0.40 \u2026 1.00}", "")]
WIDTH_FIXED   = [("\u03b4", ""), ("max", "sub"), (" = 0.80", "")]
WINDOW_RESULT = [("Width vs center effects on NPV and f", ""), ("d", "sub")]
DIVIDER  = HexColor("#888780")   # centre divider line

# ── Canvas ────────────────────────────────────────────────────────────────────
SCALE  = 160 / 680               # mm per SVG px
FIG_W  = 160 * mm
# Legend sits at y=488..500, so 514 leaves a 14 px bottom margin. The former
# 600 left about 24 mm of blank paper below the legend.
FIG_H  = 514 * SCALE * mm
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


def seg_width(segs, font, size):
    """Width of a segment list, sub- and superscripts included."""
    return sum(stringWidth(s, font, size if st == "" else size * 0.72)
               for s, st in segs)


def as_segs(text):
    """Accept a plain string or an existing segment list."""
    return text if isinstance(text, list) else [(text, "")]


def centred_segs(c, cx_svg, y_svg, segs, font, size, color):
    """Centred text from (string, style) pairs; style in {'', 'sub', 'sup'}."""
    x = s(cx_svg) - seg_width(segs, font, size) / 2
    c.setFillColor(color)
    for txt_, st in segs:
        fs = size if st == "" else size * 0.72
        dy = {"": 0.0, "sub": -0.22 * size, "sup": 0.38 * size}[st]
        c.setFont(font, fs)
        c.drawString(x, sy(y_svg) + dy, txt_)
        x += stringWidth(txt_, font, fs)


def two_line_box(c, x, y_top, w, h, r, fill, stroke, sw,
                 title, tc, tf, ts_size,
                 sub,   sc, sub_size):
    rr(c, x, y_top, w, h, r, fill, stroke, sw)
    cx = x + w/2
    # title at 38% from top, sub at 68%
    centred_segs(c, cx, y_top + h*0.38, as_segs(title), tf, ts_size, tc)
    centred_segs(c, cx, y_top + h*0.68, as_segs(sub), "Helvetica", sub_size, sc)


def three_line_box(c, x, y_top, w, h, r, fill, stroke, sw,
                   title, tc, tf, ts_size,
                   sub1, sub2, sc, sub_size):
    """Box with title + two subtitle lines (coral sweep boxes)."""
    rr(c, x, y_top, w, h, r, fill, stroke, sw)
    cx = x + w/2
    centred_segs(c, cx, y_top + h*0.30, as_segs(title), tf, ts_size, tc)
    centred_segs(c, cx, y_top + h*0.56, as_segs(sub1), "Helvetica", sub_size, sc)
    centred_segs(c, cx, y_top + h*0.78, as_segs(sub2), "Helvetica", sub_size, sc)


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


# -- Fit check ---------------------------------------------------------------- #
# (tag, box width in SVG px, [(text, font, size), ...])
BOXES = [
    ("wind",     232, [("Wind + price time series", "Helvetica-Bold", 9),
                       ("ERA5 90 m, DK1 2022", "Helvetica", 7.5)]),
    ("coarse",   232, [("Coarse E \u00d7 P grid", "Helvetica-Bold", 9),
                       ("11 \u00d7 8 pts, E: 150\u20131500 MWh", "Helvetica", 7.5)]),
    ("loop1",    232, [("20-yr multi-year loop", "Helvetica-Bold", 9),
                       ("LP + rainflow + deg., 3 scenarios", "Helvetica", 7.5)]),
    ("region",   232, [("Locate degraded region", "Helvetica-Bold", 9),
                       ("Coarse grid maximum", "Helvetica", 7.5)]),
    ("refined",  232, [("Refined E \u00d7 P grid", "Helvetica-Bold", 9),
                       ("7 \u00d7 6 pts, E: 450\u2013750 MWh", "Helvetica", 7.5)]),
    ("optimum",  232, [("Grid optimum + quadratic fit", "Helvetica-Bold", 9),
                       ("Xu optimum \u2192 Stage 2 anchor", "Helvetica", 7.5)]),
    ("size",     268, [("Fixed battery size", "Helvetica-Bold", 9),
                       ("E = 550 MWh, P = 175 MW", "Helvetica", 7.5)]),
    ("width",    124, [("Width series", "Helvetica-Bold", 9),
                       ("Center = 0.50", "Helvetica", 7.5),
                       (WIDTH_SET, "Helvetica", 7.5)]),
    ("center",   124, [("Center series", "Helvetica-Bold", 9),
                       (WIDTH_FIXED, "Helvetica", 7.5),
                       ("Center \u2208 {0.40 \u2026 0.60}", "Helvetica", 7.5)]),
    ("loop2",    268, [("20-yr simulation loop", "Helvetica-Bold", 9),
                       ("Fixed (E, P), per window, Xu + Shi", "Helvetica", 7.5)]),
    ("results",  268, [("Window sensitivity results", "Helvetica-Bold", 9),
                       (WINDOW_RESULT, "Helvetica", 7.5)]),
]


def check_fit(pad=6):
    """Warn if any label is wider than the box that holds it. Widths in SVG px."""
    bad = []
    for tag, w, items in BOXES:
        for text, font, size in items:
            need = seg_width(as_segs(text), font, size) / (SCALE * mm)
            if need > w - pad:
                bad.append(f"  {tag}: needs {need:.0f} px in a {w - pad} px box")
    if bad:
        print("WARNING: text wider than its box")
        print("\n".join(bad))
    else:
        print("fit check: all labels inside their boxes")


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
    c.line(s(338), sy(14), s(338), sy(470))
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
                 "7 \u00d7 6 pts, E: 450\u2013750 MWh", PURP_ST, 7.5)
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
                 "E = 550 MWh, P = 175 MW", GRAY_ST, 7.5)

    # Fork from fixed box top (492,400) up to y=376, then split to 420 and 564
    plain_line(c, 492, 400, 492, 376, MUTED, LW)  # vertical stem up
    plain_line(c, 420, 376, 564, 376, MUTED, LW)  # horizontal fork
    arrow_v_up(c, 420, 376, 340, MUTED, LW)        # left arm → width series
    arrow_v_up(c, 564, 376, 340, MUTED, LW)        # right arm → center series

    # Width series (coral, h=72, y=268..340, x=358..482)
    three_line_box(c, 358, 268, 124, 72, 6, CORAL_F, CORAL_S, 0.4,
                   "Width series", CORAL_T, "Helvetica-Bold", 9,
                   "Center = 0.50", WIDTH_SET,
                   CORAL_ST, 7.5)

    # Center series (coral, h=72, y=268..340, x=502..626)
    three_line_box(c, 502, 268, 124, 72, 6, CORAL_F, CORAL_S, 0.4,
                   "Center series", CORAL_T, "Helvetica-Bold", 9,
                   WIDTH_FIXED, "Center \u2208 {0.40 \u2026 0.60}",
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
                 WINDOW_RESULT, TEAL_ST, 7.5)

    # ── Legend ────────────────────────────────────────────────────────────
    swatch(c,  44, 488, HexColor("#AFA9EC"), "Simulation step")
    swatch(c, 168, 488, HexColor("#5DCAA5"), "Output / optimum")
    swatch(c, 308, 488, HexColor("#F0997B"), "Window sweep series")
    swatch(c, 494, 488, HexColor("#B4B2A9"), "Fixed input")

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