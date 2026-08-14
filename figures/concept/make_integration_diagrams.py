"""
Generate the post-processing and iterative-sequential integration diagrams as vector PDFs, using XDSM-style notation (not formal XDSMs).

Shapes:
  rounded rectangle -> optimization block
  rectangle         -> analysis / simulation block
  parallelogram     -> data (inputs and outputs)

The two diagrams share identical block positions so the only visible difference is the presence of a feedback loop in the iterative case.

Requirements: reportlab, and pymupdf for the PNG  (pip install reportlab pymupdf) 
Runs standalone; outputs are written next to this script.

NOTE ON SCOPE. None of the three diagrams below maps to a numbered figure in the thesis. Panels (a) and (b) of Figure 1.1 are drawn by degradation_coupling_taxonomy.py 
and Figure 3.13 by nested_gradient_concept_v4.py. This script is kept as a working alternative rendering, not as a build step for the document.
"""

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white

# -- Output ------------------------------------------------------------------ #
OUTPUT = "png"     # "png", "pdf" or "both"
DPI = 300

# ----------------------------------------------------------------------
# Palette (matches the thesis_style.py TU Delft palette by hex value)
# ----------------------------------------------------------------------
OPT_COLOR      = HexColor("#0076C2")   # TU Delft blue  -> optimization
ANALYSIS_COLOR = HexColor("#7B6FA8")   # muted purple   -> analysis / simulation
DATA_COLOR     = HexColor("#6E7B8B")   # slate gray     -> data nodes
LINE_COLOR     = HexColor("#333333")   # connectors
EDGE_LABEL     = HexColor("#444444")   # small edge labels

FONT      = "Helvetica-Bold"
EDGE_FONT = "Helvetica"

# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------
def _centered_text(c, cx, cy, lines, size=10, color=white):
    leading = size * 1.16
    c.setFont(FONT, size)
    c.setFillColor(color)
    n = len(lines)
    start = cy + (n - 1) * leading / 2.0
    for i, line in enumerate(lines):
        y = start - i * leading - size * 0.35
        c.drawCentredString(cx, y, line)


def rounded_box(c, block, size=10):
    cx, cy = block["c"]; w, h = block["w"], block["h"]
    c.setFillColor(block.get("color", OPT_COLOR))
    c.setStrokeColor(block.get("color", OPT_COLOR))
    c.roundRect(cx - w / 2, cy - h / 2, w, h, radius=h / 2.4, fill=1, stroke=0)
    _centered_text(c, cx, cy, block["label"], size=size)


def rect_box(c, block, size=10):
    cx, cy = block["c"]; w, h = block["w"], block["h"]
    c.setFillColor(block.get("color", ANALYSIS_COLOR))
    c.rect(cx - w / 2, cy - h / 2, w, h, fill=1, stroke=0)
    _centered_text(c, cx, cy, block["label"], size=size)


def parallelogram(c, block, size=10):
    cx, cy = block["c"]; w, h = block["w"], block["h"]
    s = block.get("skew", 12)
    c.setFillColor(block.get("color", DATA_COLOR))
    p = c.beginPath()
    p.moveTo(cx - w / 2 - s / 2, cy - h / 2)   # bottom-left
    p.lineTo(cx + w / 2 - s / 2, cy - h / 2)   # bottom-right
    p.lineTo(cx + w / 2 + s / 2, cy + h / 2)   # top-right
    p.lineTo(cx - w / 2 + s / 2, cy + h / 2)   # top-left
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    _centered_text(c, cx, cy, block["label"], size=size)


def _arrowhead(c, x, y, ang, size=7):
    c.setFillColor(LINE_COLOR)
    left = ang + math.radians(152)
    right = ang - math.radians(152)
    p = c.beginPath()
    p.moveTo(x, y)
    p.lineTo(x + size * math.cos(left),  y + size * math.sin(left))
    p.lineTo(x + size * math.cos(right), y + size * math.sin(right))
    p.close()
    c.drawPath(p, fill=1, stroke=0)


def arrow(c, pts, width=1.4, head=7):
    c.setStrokeColor(LINE_COLOR)
    c.setLineWidth(width)
    c.setLineJoin(1)
    p = c.beginPath()
    p.moveTo(*pts[0])
    for pt in pts[1:]:
        p.lineTo(*pt)
    c.drawPath(p, stroke=1, fill=0)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    _arrowhead(c, x1, y1, math.atan2(y1 - y0, x1 - x0), size=head)


def boundary(c, x0, y0, x1, y1, color, dash=None, label=None, label_size=8):
    c.setStrokeColor(color)
    c.setLineWidth(1.2)
    if dash:
        c.setDash(dash, 0)
    c.roundRect(x0, y0, x1 - x0, y1 - y0, radius=14, stroke=1, fill=0)
    c.setDash([], 0)
    if label:
        c.setFont(EDGE_FONT, label_size)
        c.setFillColor(color)
        c.drawString(x0 + 10, y1 - label_size - 6, label)


def edge_label(c, x, y, text, size=8, anchor="center"):
    c.setFont(EDGE_FONT, size)
    c.setFillColor(EDGE_LABEL)
    if anchor == "center":
        c.drawCentredString(x, y, text)
    elif anchor == "left":
        c.drawString(x, y, text)


# convenience: edge points of a block
def right(b):  return (b["c"][0] + b["w"] / 2, b["c"][1])
def left(b):   return (b["c"][0] - b["w"] / 2, b["c"][1])
def top(b):    return (b["c"][0], b["c"][1] + b["h"] / 2)
def bottom(b): return (b["c"][0], b["c"][1] + -b["h"] / 2)


# ----------------------------------------------------------------------
# Shared block layout (identical positions in both diagrams)
# ----------------------------------------------------------------------
def base_blocks():
    return {
        "fixed":    dict(c=(130, 262), w=120, h=44, skew=12, color=DATA_COLOR,
                         label=["Fixed", "parameters"]),
        "opt":      dict(c=(130, 185), w=120, h=48, color=OPT_COLOR,
                         label=["Optimization"]),
        "x":        dict(c=(305, 185), w=128, h=44, skew=12, color=DATA_COLOR,
                         label=["Operational", "decisions"]),
        "xstar":    dict(c=(478, 185), w=86,  h=44, skew=12, color=DATA_COLOR,
                         label=["Optimum"]),
        "analysis": dict(c=(305, 100), w=150, h=48, color=ANALYSIS_COLOR,
                         label=["Degradation", "analysis"]),
    }


# ----------------------------------------------------------------------
# Diagram 1: post-processing (no feedback)
# ----------------------------------------------------------------------
def build_post_process(path):
    W, H = 560, 300
    c = canvas.Canvas(str(path), pagesize=(W, H))
    b = base_blocks()
    b["metrics"] = dict(c=(305, 30), w=160, h=44, skew=12, color=DATA_COLOR,
                        label=["Lifetime estimate"])

    # arrows first (so blocks sit on top)
    arrow(c, [(130, 262 - 22), (130, 185 + 24)])                 # fixed -> opt
    arrow(c, [right(b["opt"]), (b["x"]["c"][0] - 64, 185)])      # opt -> x
    arrow(c, [right(b["x"]),   (b["xstar"]["c"][0] - 43, 185)])  # x -> x*
    arrow(c, [(305, 185 - 22), (305, 100 + 24)])                 # x -> analysis
    arrow(c, [(305, 100 - 24), (305, 30 + 22)])                  # analysis -> metrics

    edge_label(c, 318, 138, "SoC trajectory", anchor="left")

    # blocks
    parallelogram(c, b["fixed"])
    rounded_box(c,   b["opt"])
    parallelogram(c, b["x"])
    parallelogram(c, b["xstar"])
    rect_box(c,      b["analysis"])
    parallelogram(c, b["metrics"])

    c.showPage(); c.save()


# ----------------------------------------------------------------------
# Diagram 2: iterative-sequential (feedback loop)
# ----------------------------------------------------------------------
def build_iterative(path):
    W, H = 560, 300
    c = canvas.Canvas(str(path), pagesize=(W, H))
    b = base_blocks()

    # forward arrows
    arrow(c, [(130, 262 - 22), (130, 185 + 24)])                 # fixed -> opt
    arrow(c, [right(b["opt"]), (b["x"]["c"][0] - 64, 185)])      # opt -> x
    arrow(c, [right(b["x"]),   (b["xstar"]["c"][0] - 43, 185)])  # x -> x*
    arrow(c, [(305, 185 - 22), (305, 100 + 24)])                 # x -> analysis

    # feedback loop: analysis -> down -> left -> up into optimization
    arrow(c, [(305, 100 - 24), (305, 40), (130, 40), (130, 185 - 24)])

    # labels
    edge_label(c, 318, 138, "SoC trajectory", anchor="left")
    edge_label(c, 402, 197, "on convergence")           # on the x -> x* arrow
    edge_label(c, 217, 50, "updated cost / constraint")
    edge_label(c, 217, 26, "repeat until convergence")

    # blocks
    parallelogram(c, b["fixed"])
    rounded_box(c,   b["opt"])
    parallelogram(c, b["x"])
    parallelogram(c, b["xstar"])
    rect_box(c,      b["analysis"])

    c.showPage(); c.save()


# ----------------------------------------------------------------------
# Diagram 3: nested degradation-aware optimizer (thesis Figure 3.13)
#   Outer loop sets energy capacity -> inner loop solves dispatch, builds
#   SoC trajectory, counts rainflow cycles, accumulates degradation, and
#   returns the gradient dNPV/dE, which updates the capacity. Repeats.
# ----------------------------------------------------------------------
def build_nested(path):
    W, H = 640, 390
    c = canvas.Canvas(str(path), pagesize=(W, H))

    BOUND_OUTER = HexColor("#5B6770")
    BOUND_INNER = HexColor("#9AA7B2")

    # blocks
    sizing   = dict(c=(180, 275), w=116, h=48, color=OPT_COLOR,
                    label=["Sizing", "optimization"])
    estar    = dict(c=(48, 275),  w=80,  h=42, skew=12, color=DATA_COLOR,
                    label=["Optimal", "size"])
    ecap     = dict(c=(180, 200), w=112, h=42, skew=12, color=DATA_COLOR,
                    label=["Energy", "capacity"])
    dispatch = dict(c=(180, 124), w=92,  h=46, color=OPT_COLOR,
                    label=["Dispatch", "LP"])
    soc      = dict(c=(300, 124), w=100, h=42, skew=12, color=DATA_COLOR,
                    label=["SoC", "trajectory"])
    degr     = dict(c=(445, 124), w=132, h=46, color=ANALYSIS_COLOR,
                    label=["Rainflow +", "degradation"])
    fixed    = dict(c=(180, 38),  w=146, h=40, skew=12, color=DATA_COLOR,
                    label=["Prices, wind, costs"])

    # loop boundaries
    boundary(c, 85, 62, 580, 355, BOUND_OUTER, label="Outer loop (sizing)")
    boundary(c, 110, 78, 552, 170, BOUND_INNER, dash=[4, 3],
             label="Inner loop (dispatch + degradation)")

    # arrows
    arrow(c, [(180, 251), (180, 221)])                       # sizing -> capacity
    arrow(c, [(180, 179), (180, 147)])                       # capacity -> dispatch
    arrow(c, [(180 + 46, 124), (250, 124)])                  # dispatch -> SoC
    arrow(c, [(350, 124), (379, 124)])                       # SoC -> degradation
    arrow(c, [(180, 58), (180, 101)])                        # fixed -> dispatch
    arrow(c, [(180 - 58, 275), (89, 275)])                   # sizing -> optimum

    # gradient feedback loop: degradation -> up -> left -> into sizing
    arrow(c, [(445, 147), (445, 330), (180, 330), (180, 299)])
    edge_label(c, 312, 336, "gradient  dNPV / dE")
    edge_label(c, 312, 317, "update capacity, repeat until convergence")

    # blocks on top
    parallelogram(c, fixed)
    rounded_box(c,   dispatch)
    parallelogram(c, soc)
    rect_box(c,      degr)
    parallelogram(c, ecap)
    rounded_box(c,   sizing)
    parallelogram(c, estar)

    c.showPage(); c.save()


# ----------------------------------------------------------------------
def write_png(pdf_path, png_path, dpi=DPI):
    """Rasterise a PDF to PNG. PyMuPDF first, then Poppler's pdftoppm."""
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
    if exe is None:
        print("PNG not written. Install PyMuPDF with: pip install pymupdf")
        return False
    subprocess.run([exe, "-png", "-r", str(dpi), "-singlefile",
                    str(pdf_path), str(png_path.with_suffix(""))], check=True)
    return True


def emit(build_fn, out_dir, stem):
    """Build once, then write the formats OUTPUT asks for.

    The PDF is always built, because the PNG is rasterised from it. When only a
    PNG is wanted the PDF goes to a temporary directory and is discarded, so the
    two formats can never be built from different code paths.
    """
    keep_pdf = OUTPUT in ("pdf", "both")
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = (out_dir if keep_pdf else Path(tmp)) / f"{stem}.pdf"
        build_fn(pdf_path)
        if keep_pdf:
            print(f"  wrote {pdf_path.name}")
        if OUTPUT in ("png", "both"):
            png_path = out_dir / f"{stem}.png"
            if write_png(pdf_path, png_path):
                print(f"  wrote {png_path.name}  ({DPI} dpi)")


if __name__ == "__main__":
    if OUTPUT not in ("png", "pdf", "both"):
        raise ValueError(f'OUTPUT must be "png", "pdf" or "both", not {OUTPUT!r}')

    out = Path(__file__).parent
    for builder, stem in [(build_post_process, "post_process_xdsm"),
                          (build_iterative,    "iterative_xdsm"),
                          (build_nested,       "nested_xdsm")]:
        emit(builder, out, stem)