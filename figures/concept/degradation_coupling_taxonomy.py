"""
fig_ch1_coupling_taxonomy.py
============================
Figure 1.1 - Three ways of coupling a rainflow degradation model to a dispatch optimization: (a) post-processing, (b) iterative-sequential, (c) embedded.

This script is a direct port of the reference SVG "ch1_degradation_coupling_taxonomy_three_panels.svg". Every coordinate, size and color below is taken from that file, 
so the drawing is reproducible from source instead of from a hand-edited diagram.

Coordinate system
-----------------
All geometry is written in the SVG frame: origin top-left, y increases downward, one unit = one PostScript point. The canvas is 680 x 440 units, so
the output PDF is 680 x 440 pt (9.44 x 6.11 in). The helper "Y()" flips y for ReportLab, whose origin is bottom-left.

Because the figure is included with "width=\\textwidth", LaTeX scales it down by roughly 0.63 on a 430 pt text block. On-page text sizes are therefore about
8.8 pt for node titles and 7.6 pt for subtitles. If that is too small, raise FONT_SCALE (text only) or lower CANVAS_SCALE (geometry and text together).

Outputs
-------
Set OUTPUT below to "png", "pdf" or "both". Files are written next to this script under the name in STEM. The PNG is rasterised from the same PDF the
vector output uses, so the two cannot differ.

The PNG step needs either PyMuPDF ("pip install pymupdf") or Poppler's "pdftoppm" on PATH. If neither is present the script prints a notice and exits
normally.

Run from VS Code on Windows; paths are anchored with Path(__file__).parent.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from reportlab.lib.colors import Color, HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas

HERE = Path(__file__).parent
STEM = "fig_ch1_coupling_taxonomy"

# ----------------------------------------------------------------------------
# 1. Knobs
# ----------------------------------------------------------------------------

OUTPUT = "png"      # "png", "pdf" or "both"
CANVAS_SCALE = 1.0   # scales geometry and text together; 1.0 = same size as the SVG
FONT_SCALE = 1.0     # scales text only, on top of CANVAS_SCALE
PNG_DPI = 300

# Font preference, first match wins.
#   "Carlito" / "Calibri" keeps this figure consistent with the rest of the
#   thesis figures. "Segoe UI" reproduces the browser rendering of the source
#   SVG more literally. Move the entry you want to the front of the list.
FONT_PREFERENCE = ["Carlito", "Calibri", "Segoe UI", "DejaVu Sans", "Liberation Sans"]

# SVG font-weight 500 sits between regular and bold. No metric-compatible
# medium face exists for Carlito or Calibri, so titles use the regular face and
# are separated from subtitles by size and color, as in the source SVG.
# Set to True to use the bold face for node titles and panel labels instead.
BOLD_TITLES = False

# ----------------------------------------------------------------------------
# 2. Palette (RGB values read directly from the source SVG)
# ----------------------------------------------------------------------------

C_GRAY_FILL = HexColor("#F1EFE8")
C_GRAY_STROKE = HexColor("#5F5E5A")
C_GRAY_TITLE = HexColor("#444441")
C_GRAY_SUB = HexColor("#5F5E5A")

C_TEAL_FILL = HexColor("#E1F5EE")
C_TEAL_STROKE = HexColor("#0F6E56")
C_TEAL_TITLE = HexColor("#085041")
C_TEAL_SUB = HexColor("#0F6E56")

C_CORAL_FILL = HexColor("#FAECE7")
C_CORAL_STROKE = HexColor("#993C1D")
C_CORAL_TITLE = HexColor("#712B13")
C_CORAL_SUB = HexColor("#993C1D")

C_ARROW = HexColor("#898781")          # rgb(137,135,129)
C_PANEL_LABEL = HexColor("#0B0B0B")    # rgb(11,11,11)
C_EDGE_LABEL = HexColor("#52514E")     # rgb(82,81,78)
C_BOUNDARY = HexColor("#0B0B0B")       # drawn at 20% alpha, as in the SVG
BOUNDARY_ALPHA = 0.20

# ----------------------------------------------------------------------------
# 3. Layout constants (source SVG values)
# ----------------------------------------------------------------------------

W, H = 680.0, 440.0

BOX_H = 56.0
NODE_STROKE_W = 0.5
ARROW_W = 1.5
# Node geometry shared by all three panels: (x, width, corner radius, style)
NODES = [
    dict(x=40.0, w=78.0, r=4.0, style="gray", title="Inputs", sub=None),
    dict(x=138.0, w=118.0, r=8.0, style="teal", title="Optimization", sub="Max revenue"),
    dict(x=276.0, w=96.0, r=4.0, style="gray", title="SoC", sub="trajectory"),
    dict(x=392.0, w=130.0, r=4.0, style="coral", title="Degradation", sub="Rainflow cycles"),
    dict(x=542.0, w=96.0, r=4.0, style="gray", title="Outputs", sub="EoL, NPV"),
]

STYLES = {
    "gray": dict(fill=C_GRAY_FILL, stroke=C_GRAY_STROKE, title=C_GRAY_TITLE, sub=C_GRAY_SUB),
    "teal": dict(fill=C_TEAL_FILL, stroke=C_TEAL_STROKE, title=C_TEAL_TITLE, sub=C_TEAL_SUB),
    "coral": dict(fill=C_CORAL_FILL, stroke=C_CORAL_STROKE, title=C_CORAL_TITLE, sub=C_CORAL_SUB),
}

FS_PANEL = 14.0     # panel label, e.g. "(a) Post-processing"
FS_TITLE = 14.0     # node title
FS_SUB = 12.0       # node subtitle
FS_EDGE = 12.0      # return-edge label and boundary label

# Per-panel vertical placement. label_y is a text baseline; row_y is the top
# edge of the node rectangles.
PANELS = [
    dict(
        label="(a) Post-processing",
        label_y=40.0,
        row_y=52.0,
        ret=None,
        boundary=None,
    ),
    dict(
        label="(b) Iterative-sequential",
        label_y=150.0,
        row_y=162.0,
        ret=dict(depth=28.0, text="Updated cost or constraint", text_y=240.0),
        boundary=None,
    ),
    dict(
        label="(c) Embedded",
        label_y=280.0,
        row_y=322.0,
        ret=dict(depth=22.0, text="Cost term in the objective", text_y=394.0),
        boundary=dict(x=128.0, y=296.0, w=404.0, h=120.0, r=8.0,
                      text="Single optimization problem", text_x=136.0, text_y=312.0),
    ),
]

RET_FROM = 3        # return edge leaves the degradation node
RET_TO = 1          # and enters the optimization node
RET_LABEL_X = 327.0

# ----------------------------------------------------------------------------
# 4. Font resolution
# ----------------------------------------------------------------------------

FONT_FILES = {
    "Carlito": ("Carlito-Regular.ttf", "Carlito-Bold.ttf"),
    "Calibri": ("calibri.ttf", "calibrib.ttf"),
    "Segoe UI": ("segoeui.ttf", "segoeuib.ttf"),
    "DejaVu Sans": ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    "Liberation Sans": ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
}

SEARCH_DIRS = [
    Path("C:/Windows/Fonts"),
    Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
    Path("/usr/share/fonts/truetype/crosextra"),
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/truetype/liberation"),
    Path("/usr/share/fonts"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts/Supplemental"),
]


def _find_font_file(filename: str) -> Path | None:
    """Locate a font file by name in the usual system directories."""
    for directory in SEARCH_DIRS:
        if not directory.is_dir():
            continue
        candidate = directory / filename
        if candidate.is_file():
            return candidate
        for found in directory.rglob(filename):
            return found
    return None


def resolve_fonts() -> tuple[str, str]:
    """Register the first available preferred family. Falls back to Helvetica."""
    for family in FONT_PREFERENCE:
        regular_name, bold_name = FONT_FILES[family]
        regular = _find_font_file(regular_name)
        bold = _find_font_file(bold_name)
        if regular is None:
            continue
        tag = family.replace(" ", "")
        pdfmetrics.registerFont(TTFont(tag, str(regular)))
        bold_tag = tag
        if bold is not None:
            bold_tag = tag + "-Bold"
            pdfmetrics.registerFont(TTFont(bold_tag, str(bold)))
        print(f"font: {family} ({regular.name})")
        return tag, bold_tag
    print("font: no preferred family found, falling back to Helvetica")
    return "Helvetica", "Helvetica-Bold"


# ----------------------------------------------------------------------------
# 5. Drawing helpers
# ----------------------------------------------------------------------------


def Y(y: float) -> float:
    """Convert an SVG y coordinate (down-positive) to a PDF y coordinate."""
    return H - y


def draw_text(c, x, y, text, font, size, color, anchor="start", central=False):
    """Place text using SVG semantics.

    anchor : 'start' or 'middle'
    central: True reproduces dominant-baseline="central", which centers the
             ascender-to-descender box on y rather than sitting on the baseline.
    """
    size = size * FONT_SCALE
    ascent, descent = pdfmetrics.getAscentDescent(font, size)
    baseline_y = y + (ascent + descent) / 2.0 if central else y
    c.setFont(font, size)
    c.setFillColor(color)
    if anchor == "middle":
        c.drawCentredString(x, Y(baseline_y), text)
    else:
        c.drawString(x, Y(baseline_y), text)


def draw_arrowhead(c, x_tip, y_tip, direction):
    """Reproduce the SVG marker: an open chevron, stroked, no fill.

    The marker is 6 stroke-width units wide with a 0..10 viewBox, so one viewBox
    unit is 0.9 pt at a 1.5 pt line. The chevron spans x 2..8 and y 1..9, and
    refX=8 puts its tip on the line endpoint.
    """
    u = ARROW_W * 6.0 / 10.0          # viewBox unit in points
    back = 6.0 * u                    # 8 - 2 viewBox units
    half = 4.0 * u                    # (9 - 1) / 2 viewBox units

    c.saveState()
    c.setStrokeColor(C_ARROW)
    c.setLineWidth(1.5 * u)
    c.setLineCap(1)
    c.setLineJoin(1)
    path = c.beginPath()
    if direction == "right":
        path.moveTo(x_tip - back, Y(y_tip - half))
        path.lineTo(x_tip, Y(y_tip))
        path.lineTo(x_tip - back, Y(y_tip + half))
    elif direction == "up":
        path.moveTo(x_tip - half, Y(y_tip + back))
        path.lineTo(x_tip, Y(y_tip))
        path.lineTo(x_tip + half, Y(y_tip + back))
    else:
        raise ValueError(f"unsupported arrow direction: {direction}")
    c.drawPath(path, stroke=1, fill=0)
    c.restoreState()


def draw_connector(c, x0, x1, y):
    """Horizontal connector with an arrowhead at x1."""
    c.saveState()
    c.setStrokeColor(C_ARROW)
    c.setLineWidth(ARROW_W)
    c.line(x0, Y(y), x1, Y(y))
    c.restoreState()
    draw_arrowhead(c, x1, y, "right")


def draw_node(c, node, row_y, font_reg, font_title):
    style = STYLES[node["style"]]
    x, w, r = node["x"], node["w"], node["r"]

    c.saveState()
    c.setFillColor(style["fill"])
    c.setStrokeColor(style["stroke"])
    c.setLineWidth(NODE_STROKE_W)
    c.roundRect(x, Y(row_y + BOX_H), w, BOX_H, r, stroke=1, fill=1)
    c.restoreState()

    cx = x + w / 2.0
    if node["sub"] is None:
        draw_text(c, cx, row_y + BOX_H / 2.0, node["title"], font_title, FS_TITLE,
                  style["title"], anchor="middle", central=True)
    else:
        draw_text(c, cx, row_y + 18.0, node["title"], font_title, FS_TITLE,
                  style["title"], anchor="middle", central=True)
        draw_text(c, cx, row_y + 36.0, node["sub"], font_reg, FS_SUB,
                  style["sub"], anchor="middle", central=True)


def draw_return_edge(c, panel, font_reg):
    """Degradation -> optimization edge, routed below the node row."""
    ret = panel["ret"]
    row_y = panel["row_y"]
    src = NODES[RET_FROM]
    dst = NODES[RET_TO]
    x_src = src["x"] + src["w"] / 2.0
    x_dst = dst["x"] + dst["w"] / 2.0
    y_bottom = row_y + BOX_H
    y_route = y_bottom + ret["depth"]
    y_tip = y_bottom + 6.0            # head stops just below the box edge

    c.saveState()
    c.setStrokeColor(C_ARROW)
    c.setLineWidth(ARROW_W)
    c.setLineJoin(0)
    path = c.beginPath()
    path.moveTo(x_src, Y(y_bottom))
    path.lineTo(x_src, Y(y_route))
    path.lineTo(x_dst, Y(y_route))
    path.lineTo(x_dst, Y(y_tip))
    c.drawPath(path, stroke=1, fill=0)
    c.restoreState()
    draw_arrowhead(c, x_dst, y_tip, "up")

    draw_text(c, RET_LABEL_X, ret["text_y"], ret["text"], font_reg, FS_EDGE,
              C_EDGE_LABEL, anchor="middle")


def draw_boundary(c, boundary, font_reg):
    c.saveState()
    c.setStrokeColor(C_BOUNDARY)
    c.setStrokeAlpha(BOUNDARY_ALPHA)
    c.setLineWidth(0.5)
    c.setDash(4, 3)
    c.roundRect(boundary["x"], Y(boundary["y"] + boundary["h"]),
                boundary["w"], boundary["h"], boundary["r"], stroke=1, fill=0)
    c.restoreState()
    draw_text(c, boundary["text_x"], boundary["text_y"], boundary["text"],
              font_reg, FS_EDGE, C_EDGE_LABEL, anchor="start")


# ----------------------------------------------------------------------------
# 6. Fit check
# ----------------------------------------------------------------------------


def check_text_fits(font_reg, font_title) -> None:
    """Warn if any label is wider than the box that holds it.

    Box widths come from a browser rendering of the SVG. A different font family
    has different metrics, so this guards against silent overflow.
    """
    problems = []
    for node in NODES:
        limit = node["w"] - 8.0
        for text, font, size in ((node["title"], font_title, FS_TITLE),
                                 (node["sub"], font_reg, FS_SUB)):
            if text is None:
                continue
            width = pdfmetrics.stringWidth(text, font, size * FONT_SCALE)
            if width > limit:
                problems.append(f"  {text!r}: {width:.1f} pt in a {limit:.1f} pt box")
    if problems:
        print("WARNING: text wider than its box")
        print("\n".join(problems))
    else:
        print("fit check: all labels inside their boxes")


# ----------------------------------------------------------------------------
# 7. Build
# ----------------------------------------------------------------------------


def build_pdf(path: Path, font_reg: str, font_bold: str) -> None:
    font_title = font_bold if BOLD_TITLES else font_reg

    page_w, page_h = W * CANVAS_SCALE, H * CANVAS_SCALE
    c = rl_canvas.Canvas(str(path), pagesize=(page_w, page_h))
    c.setTitle("Three ways of coupling rainflow degradation to a dispatch optimization")
    if CANVAS_SCALE != 1.0:
        c.scale(CANVAS_SCALE, CANVAS_SCALE)

    for panel in PANELS:
        if panel["boundary"] is not None:
            draw_boundary(c, panel["boundary"], font_reg)

        draw_text(c, 40.0, panel["label_y"], panel["label"], font_title, FS_PANEL,
                  C_PANEL_LABEL, anchor="start")

        row_y = panel["row_y"]
        for node in NODES:
            draw_node(c, node, row_y, font_reg, font_title)

        y_mid = row_y + BOX_H / 2.0
        for left, right in zip(NODES[:-1], NODES[1:]):
            x0 = left["x"] + left["w"] + 4.0
            x1 = right["x"] - 4.0
            draw_connector(c, x0, x1, y_mid)

        if panel["ret"] is not None:
            draw_return_edge(c, panel, font_reg)

    c.showPage()
    c.save()


def export_png(pdf_path: Path, png_path: Path, dpi: int = PNG_DPI) -> bool:
    """PDF to PNG via PyMuPDF, then Poppler. Returns True on success."""
    try:
        try:
            import pymupdf              # PyMuPDF 1.24.3 and later
        except ImportError:
            import fitz as pymupdf      # older releases expose it as fitz

        doc = pymupdf.open(str(pdf_path))
        doc.load_page(0).get_pixmap(dpi=dpi, alpha=False).save(str(png_path))
        doc.close()
        return True
    except ImportError:
        pass

    exe = shutil.which("pdftoppm")
    if exe:
        subprocess.run([exe, "-png", "-r", str(dpi), "-singlefile",
                        str(pdf_path), str(png_path.with_suffix(""))], check=True)
        return True

    print("PNG not written. Install PyMuPDF with: pip install pymupdf")
    return False


def main() -> int:
    if OUTPUT not in ("png", "pdf", "both"):
        raise ValueError(f'OUTPUT must be "png", "pdf" or "both", not {OUTPUT!r}')

    font_reg, font_bold = resolve_fonts()
    check_text_fits(font_reg, font_bold if BOLD_TITLES else font_reg)

    keep_pdf = OUTPUT in ("pdf", "both")
    with tempfile.TemporaryDirectory() as tmp:
        pdf_dir = HERE if keep_pdf else Path(tmp)
        pdf_path = pdf_dir / f"{STEM}.pdf"
        build_pdf(pdf_path, font_reg, font_bold)
        if keep_pdf:
            print(f"wrote {pdf_path}  "
                  f"({W * CANVAS_SCALE:.0f} x {H * CANVAS_SCALE:.0f} pt)")

        if OUTPUT in ("png", "both"):
            png_path = HERE / f"{STEM}.png"
            if export_png(pdf_path, png_path):
                print(f"wrote {png_path}  ({PNG_DPI} DPI)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
