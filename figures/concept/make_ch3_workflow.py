"""
Chapter 3 figure: the three investigative approaches.

  (a) Baseline evaluation   LP -> e* -> two degradation models -> NPV, EoL
  (b) Parameter sweep       the same pipeline repeated over a grid of (E, P)
  (c) Monolithic NLP        degradation cost inside the objective

The figure is authored once as a list of drawing operations in SVG coordinates
(origin top left, y increasing downward) and then written out by two backends:

  render_svg  -> .svg   plain text, no dependencies
  render_pdf  -> .pdf   ReportLab, vector, this is the file LaTeX should use
  the .png is rasterised from the .pdf

Both backends read the same geometry, so the three files cannot drift apart.

LAYOUT
  All positions come from the LAYOUT block below. Vertical placement is driven
  by three numbers: MARGIN_TOP, TITLE_DROP and PANEL_GAP. The panels were
  previously separated by about 116 pt of empty space; PANEL_GAP now sets that
  distance directly.

  The panel (b) title is placed a fixed TITLE_LIFT above the top edge of the
  dashed repeat box. In the earlier version the title baseline sat below that
  edge, on the same line as the "repeat for each (E, P) on the grid" label, and
  the two texts overlapped.

Reproducing on Windows in VS Code:
    pip install reportlab pymupdf
    python make_ch3_workflow.py
PyMuPDF is only needed for the PNG. Without it the script falls back to
poppler's pdftoppm, and if that is absent it writes the SVG and PDF and skips
the PNG with a message.
"""
import os
import subprocess
import shutil
from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

# -- Palette (taken from the original figure, unchanged) --------------------
INK        = "#0B0B0B"   # panel titles
NEUTRAL_BG = "#F1EFE8"   # Inputs / Outputs fill
NEUTRAL_ST = "#5F5E5A"   # Inputs / Outputs stroke, and sub-label text
NEUTRAL_TX = "#444441"   # Inputs / Outputs title text
BLUE_BG    = "#E6F1FB"   # LP / NLP / e* fill
BLUE_ST    = "#185FA5"   # LP / NLP stroke, sub-label text, thick arrow
BLUE_TX    = "#0C447C"   # LP / NLP title text
RED_BG     = "#FCEBEB"   # degradation model fill
RED_ST     = "#A32D2D"   # degradation model stroke and text
RED_ARROW  = "#E24B4A"   # arrow into the NLP objective
CONNECT    = "#898781"   # plain connectors
DASH_ST    = "#888780"   # dashed repeat box
NOTE_TX    = "#52514E"   # side note and repeat-box label

# -- LAYOUT -----------------------------------------------------------------
WIDTH        = 680   # figure width in points

MARGIN_TOP   = 16    # top of the figure to the first panel title baseline
MARGIN_BOT   = 14    # last content to the bottom edge
TITLE_DROP   = 14    # panel title baseline to the top of that panel's content
PANEL_GAP    = 32    # bottom of one panel's content to the next title baseline
TITLE_LIFT   = 8     # panel (b) title baseline above the dashed box top edge

COL_IN       = 24    # left edge of the Inputs box
W_IO         = 92    # Inputs box width
H_BOX        = 46    # standard box height
COL_LP       = 150   # left edge of the LP box
W_LP         = 92
CX_ESTAR     = 300   # centre of the e* circle
R_ESTAR      = 15
COL_MODEL    = 360   # left edge of the degradation model boxes
W_MODEL      = 150
H_MODEL      = 28
DY_MODEL     = 33    # row centre to the outer edge of the model stack
COL_OUT      = 556   # left edge of the Outputs box
W_OUT        = 100

DASH_PAD_T   = 12    # dashed box top edge to the top of the model stack
DASH_PAD_B   = 12    # bottom of the model stack to the dashed box bottom edge
DASH_X0      = 140   # dashed box left edge
DASH_X1      = 528   # dashed box right edge

C_COL_NLP    = 188   # panel (c): left edge of the NLP box
C_W_NLP      = 150
C_COL_OUT    = 410
C_DEG_DY     = 90    # NLP box top to the degradation box top
C_DEG_W      = 116
C_DEG_H      = 40
C_NOTE_X     = 524   # left edge of the side note
C_NOTE_DY    = 67    # NLP box top to the first note baseline
C_NOTE_LEAD  = 16

FS_TITLE     = 14    # panel titles and box titles
FS_LABEL     = 12    # sub-labels, model boxes, notes
RADIUS       = 6     # box corner radius
LW_THIN      = 0.5   # box outline
LW_LINE      = 1.2   # plain connector
LW_THICK     = 3.5   # LP -> e* emphasis
LW_RED       = 2.5   # degradation cost -> NLP


# -- Scene: backend-independent drawing operations --------------------------
class Scene:
    """Collects drawing operations in SVG coordinates (y increases downward)."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.ops = []

    def rrect(self, x, y, w, h, fill, stroke, lw=LW_THIN, r=RADIUS, dash=None):
        self.ops.append(("rrect", dict(x=x, y=y, w=w, h=h, fill=fill,
                                       stroke=stroke, lw=lw, r=r, dash=dash)))

    def circle(self, cx, cy, r, fill, stroke, lw=LW_THIN):
        self.ops.append(("circle", dict(cx=cx, cy=cy, r=r, fill=fill,
                                        stroke=stroke, lw=lw)))

    def poly(self, pts, stroke, lw=LW_LINE, arrow=True):
        """Polyline through pts, optionally with an arrowhead at the last point."""
        self.ops.append(("poly", dict(pts=list(pts), stroke=stroke, lw=lw,
                                      arrow=arrow)))

    def text(self, x, y, s, size=FS_LABEL, color=INK, bold=False,
             anchor="middle", central=False):
        """`central` centres the glyphs vertically on y instead of using the
        alphabetic baseline."""
        self.ops.append(("text", dict(x=x, y=y, s=s, size=size, color=color,
                                      bold=bold, anchor=anchor, central=central)))


# -- Reusable figure pieces -------------------------------------------------
def two_line_box(sc, x, y, w, title, sub, bg, st, tx_title, tx_sub, h=H_BOX):
    sc.rrect(x, y, w, h, bg, st)
    sc.text(x + w / 2, y + 19, title, FS_TITLE, tx_title, bold=True, central=True)
    sc.text(x + w / 2, y + 33, sub, FS_LABEL, tx_sub, central=True)


def pipeline(sc, yc, in_sub, out_sub, out_x=COL_OUT, join_x=532):
    """The LP -> e* -> two models -> Outputs chain, centred on row yc.

    Returns nothing; draws in place. `join_x` is where the two model outputs
    merge before the arrow into the Outputs box.
    """
    top = yc - H_BOX / 2
    two_line_box(sc, COL_IN, top, W_IO, "Inputs", in_sub,
                 NEUTRAL_BG, NEUTRAL_ST, NEUTRAL_TX, NEUTRAL_ST)
    sc.poly([(COL_IN + W_IO, yc), (COL_LP, yc)], CONNECT)
    two_line_box(sc, COL_LP, top, W_LP, "LP", "max revenue",
                 BLUE_BG, BLUE_ST, BLUE_TX, BLUE_ST)

    # emphasised LP -> e* arrow
    sc.poly([(COL_LP + W_LP, yc), (CX_ESTAR - R_ESTAR - 1, yc)],
            BLUE_ST, LW_THICK)
    sc.circle(CX_ESTAR, yc, R_ESTAR, BLUE_BG, BLUE_ST)
    sc.text(CX_ESTAR, yc, "e*", FS_LABEL, BLUE_ST, central=True)

    # fan out to the two degradation models
    y_xu = yc - DY_MODEL + H_MODEL / 2
    y_shi = yc + DY_MODEL - H_MODEL / 2
    sc.poly([(CX_ESTAR + R_ESTAR, yc), (338, yc), (338, y_xu), (COL_MODEL, y_xu)],
            CONNECT)
    sc.poly([(CX_ESTAR + R_ESTAR, yc), (338, yc), (338, y_shi), (COL_MODEL, y_shi)],
            CONNECT)
    sc.rrect(COL_MODEL, yc - DY_MODEL, W_MODEL, H_MODEL, RED_BG, RED_ST)
    sc.text(COL_MODEL + W_MODEL / 2, y_xu, "Physical (Xu) model",
            FS_LABEL, RED_ST, central=True)
    sc.rrect(COL_MODEL, yc + DY_MODEL - H_MODEL, W_MODEL, H_MODEL, RED_BG, RED_ST)
    sc.text(COL_MODEL + W_MODEL / 2, y_shi, "Polynomial (Shi) model",
            FS_LABEL, RED_ST, central=True)

    # merge and hand off to Outputs
    sc.poly([(COL_MODEL + W_MODEL, y_shi), (join_x, y_shi), (join_x, yc)],
            CONNECT, arrow=False)
    sc.poly([(COL_MODEL + W_MODEL, y_xu), (join_x, y_xu), (join_x, yc),
             (out_x, yc)], CONNECT)
    two_line_box(sc, out_x, top, W_OUT, "Outputs", out_sub,
                 NEUTRAL_BG, NEUTRAL_ST, NEUTRAL_TX, NEUTRAL_ST)


def build_scene():
    sc = Scene(WIDTH, 10)  # height patched at the end

    # ---- (a) Baseline evaluation
    y_title_a = MARGIN_TOP
    yc_a = y_title_a + TITLE_DROP + DY_MODEL
    sc.text(COL_IN, y_title_a, "(a) Baseline evaluation", FS_TITLE, INK,
            bold=True, anchor="start")
    pipeline(sc, yc_a, "E, P", "NPV, EoL")
    bottom_a = yc_a + DY_MODEL

    # ---- (b) Parameter sweep
    y_title_b = bottom_a + PANEL_GAP
    dash_top = y_title_b + TITLE_LIFT
    yc_b = dash_top + DASH_PAD_T + DY_MODEL
    dash_bot = yc_b + DY_MODEL + DASH_PAD_B
    sc.text(COL_IN, y_title_b, "(b) Parameter sweep", FS_TITLE, INK,
            bold=True, anchor="start")
    sc.rrect(DASH_X0, dash_top, DASH_X1 - DASH_X0, dash_bot - dash_top,
             None, DASH_ST, LW_THIN, r=10, dash=(4, 3))
    sc.text(DASH_X0 + 8, dash_top + 14, "repeat for each (E, P) on the grid",
            FS_LABEL, NOTE_TX, anchor="start")
    pipeline(sc, yc_b, "grid of E, P", "max NPV point")
    bottom_b = dash_bot

    # ---- (c) Monolithic NLP
    y_title_c = bottom_b + PANEL_GAP
    top_c = y_title_c + TITLE_DROP
    yc_c = top_c + H_BOX / 2
    sc.text(COL_IN, y_title_c, "(c) Monolithic NLP", FS_TITLE, INK,
            bold=True, anchor="start")
    two_line_box(sc, COL_IN, top_c, W_IO, "Inputs", "E, P",
                 NEUTRAL_BG, NEUTRAL_ST, NEUTRAL_TX, NEUTRAL_ST)
    sc.poly([(COL_IN + W_IO, yc_c), (C_COL_NLP, yc_c)], CONNECT)
    two_line_box(sc, C_COL_NLP, top_c, C_W_NLP, "NLP", "single solve",
                 BLUE_BG, BLUE_ST, BLUE_TX, BLUE_ST)
    sc.poly([(C_COL_NLP + C_W_NLP, yc_c), (C_COL_OUT, yc_c)], CONNECT)
    two_line_box(sc, C_COL_OUT, top_c, W_OUT, "Outputs", "NPV",
                 NEUTRAL_BG, NEUTRAL_ST, NEUTRAL_TX, NEUTRAL_ST)

    deg_top = top_c + C_DEG_DY
    deg_cx = C_COL_NLP + C_W_NLP / 2
    sc.poly([(deg_cx, deg_top), (deg_cx, top_c + H_BOX + 2)], RED_ARROW, LW_RED)
    sc.rrect(deg_cx - C_DEG_W / 2, deg_top, C_DEG_W, C_DEG_H, RED_BG, RED_ST, 1.0)
    sc.text(deg_cx, deg_top + 14, "Degradation cost", FS_LABEL, RED_ST, central=True)
    sc.text(deg_cx, deg_top + 28, "in the objective", FS_LABEL, RED_ST, central=True)

    for i, line in enumerate(["Degradation enters", "the problem, not",
                              "computed after it"]):
        sc.text(C_NOTE_X, top_c + C_NOTE_DY + i * C_NOTE_LEAD, line,
                FS_LABEL, NOTE_TX, anchor="start")

    sc.height = deg_top + C_DEG_H + MARGIN_BOT
    return sc


# -- SVG backend ------------------------------------------------------------
SVG_FONT = ("Helvetica, Arial, -apple-system, BlinkMacSystemFont, "
            "&quot;Segoe UI&quot;, sans-serif")


def _svg_text_style(o):
    weight = 700 if o["bold"] else 400
    return (f'font-family:{SVG_FONT};font-size:{o["size"]}px;'
            f'font-weight:{weight};fill:{o["color"]}')


def render_svg(sc, path):
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{sc.width}" '
         f'height="{sc.height}" viewBox="0 0 {sc.width} {sc.height}" role="img">',
         '<title>Three investigative approaches: baseline, parameter sweep, '
         'monolithic NLP</title>',
         '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" '
         'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
         '<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" '
         'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
         '</marker></defs>',
         f'<rect width="{sc.width}" height="{sc.height}" fill="#FFFFFF"/>']
    for kind, o in sc.ops:
        if kind == "rrect":
            fill = o["fill"] or "none"
            dash = f' stroke-dasharray="{o["dash"][0]} {o["dash"][1]}"' if o["dash"] else ""
            L.append(f'<rect x="{o["x"]}" y="{o["y"]}" width="{o["w"]}" '
                     f'height="{o["h"]}" rx="{o["r"]}" fill="{fill}" '
                     f'stroke="{o["stroke"]}" stroke-width="{o["lw"]}"{dash}/>')
        elif kind == "circle":
            L.append(f'<circle cx="{o["cx"]}" cy="{o["cy"]}" r="{o["r"]}" '
                     f'fill="{o["fill"]}" stroke="{o["stroke"]}" '
                     f'stroke-width="{o["lw"]}"/>')
        elif kind == "poly":
            d = "M" + " L".join(f"{x} {y}" for x, y in o["pts"])
            mk = ' marker-end="url(#arrow)"' if o["arrow"] else ""
            L.append(f'<path d="{d}" fill="none" stroke="{o["stroke"]}" '
                     f'stroke-width="{o["lw"]}" stroke-linecap="round" '
                     f'stroke-linejoin="round"{mk}/>')
        elif kind == "text":
            base = ' dominant-baseline="central"' if o["central"] else ""
            esc = (o["s"].replace("&", "&amp;").replace("<", "&lt;")
                        .replace(">", "&gt;"))
            L.append(f'<text x="{o["x"]}" y="{o["y"]}" '
                     f'text-anchor="{o["anchor"]}"{base} '
                     f'style="{_svg_text_style(o)}">{esc}</text>')
    L.append("</svg>")
    Path(path).write_text("\n".join(L), encoding="utf-8")


# -- PDF backend ------------------------------------------------------------
# The SVG marker uses markerUnits="strokeWidth", so its head grows with the
# line. These factors reproduce that: a 6 x 6 marker on a 10-unit viewBox.
ARROW_LEN_PER_LW = 6.0
ARROW_HALF_PER_LW = 3.0


def render_pdf(sc, path):
    c = canvas.Canvas(str(path), pagesize=(sc.width, sc.height))

    def Y(y):                      # SVG y (down) -> PDF y (up)
        return sc.height - y

    for kind, o in sc.ops:
        if kind == "rrect":
            c.setLineWidth(o["lw"]); c.setStrokeColor(HexColor(o["stroke"]))
            c.setDash(*o["dash"]) if o["dash"] else c.setDash()
            if o["fill"]:
                c.setFillColor(HexColor(o["fill"]))
                c.roundRect(o["x"], Y(o["y"] + o["h"]), o["w"], o["h"], o["r"],
                            fill=1, stroke=1)
            else:
                c.roundRect(o["x"], Y(o["y"] + o["h"]), o["w"], o["h"], o["r"],
                            fill=0, stroke=1)
            c.setDash()
        elif kind == "circle":
            c.setLineWidth(o["lw"]); c.setStrokeColor(HexColor(o["stroke"]))
            c.setFillColor(HexColor(o["fill"]))
            c.circle(o["cx"], Y(o["cy"]), o["r"], fill=1, stroke=1)
        elif kind == "poly":
            pts = [(x, Y(y)) for x, y in o["pts"]]
            c.setStrokeColor(HexColor(o["stroke"])); c.setLineWidth(o["lw"])
            c.setLineCap(1); c.setLineJoin(1)
            head_len = ARROW_LEN_PER_LW * o["lw"]
            head_half = ARROW_HALF_PER_LW * o["lw"]
            if o["arrow"]:                     # stop the shaft short of the tip
                (x0, y0), (x1, y1) = pts[-2], pts[-1]
                dx, dy = x1 - x0, y1 - y0
                n = (dx * dx + dy * dy) ** 0.5 or 1.0
                ux, uy = dx / n, dy / n
                pts[-1] = (x1 - ux * head_len, y1 - uy * head_len)
            p = c.beginPath(); p.moveTo(*pts[0])
            for q in pts[1:]:
                p.lineTo(*q)
            c.drawPath(p, stroke=1, fill=0)
            if o["arrow"]:
                bx, by = pts[-1]
                c.setFillColor(HexColor(o["stroke"]))
                h = c.beginPath()
                h.moveTo(bx + ux * head_len, by + uy * head_len)
                h.lineTo(bx - uy * head_half, by + ux * head_half)
                h.lineTo(bx + uy * head_half, by - ux * head_half)
                h.close(); c.drawPath(h, fill=1, stroke=0)
        elif kind == "text":
            font = "Helvetica-Bold" if o["bold"] else "Helvetica"
            c.setFont(font, o["size"]); c.setFillColor(HexColor(o["color"]))
            # SVG dominant-baseline="central" is about 0.36 em above baseline
            y = Y(o["y"]) - (o["size"] * 0.36 if o["central"] else 0.0)
            if o["anchor"] == "middle":
                c.drawCentredString(o["x"], y, o["s"])
            else:
                c.drawString(o["x"], y, o["s"])
    c.showPage(); c.save()


# -- PNG ---------------------------------------------------------------------
def render_png(pdf_path, png_path, dpi=300):
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        doc[0].get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0),
                          alpha=False).save(str(png_path))
        doc.close(); return True
    except ImportError:
        pass
    if shutil.which("pdftoppm") is None:
        print("PNG skipped: install PyMuPDF with  pip install pymupdf")
        return False
    stem = str(png_path)[:-4] if str(png_path).endswith(".png") else str(png_path)
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                    str(pdf_path), stem], check=True)
    return True


if __name__ == "__main__":
    out = Path(__file__).parent
    sc = build_scene()
    render_svg(sc, out / "ch3_three_approaches_workflow.svg")
    render_pdf(sc, out / "ch3_three_approaches_workflow.pdf")
    render_png(out / "ch3_three_approaches_workflow.pdf",
               out / "ch3_three_approaches_workflow.png")
    print(f"figure size {sc.width} x {sc.height:.0f} pt "
          f"(was 680 x 620), {len(sc.ops)} drawing operations")
    print("Wrote ch3_three_approaches_workflow.{svg,pdf,png} in", out)
