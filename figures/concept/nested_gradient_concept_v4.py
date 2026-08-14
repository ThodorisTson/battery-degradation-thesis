r"""
nested_gradient_concept_v4.py   (v4.2)

Generates Figure 3.13, the conceptual gradient-based sizing method.

OUTPUT below selects "png", "pdf" or "both". WRITE_SVG is separate, because the SVG is a working format rather than a thesis deliverable.

Content changes from v3
-----------------------
  1. E_cap -> $\bar{E}$, matching the thesis notation and the surrounding text.
  2. The outer loop decides the SoC window limits as well as the capacity.
  3. Conditional tense where a verb appears, since the method is future work.

Type size
---------
v3 used a 680 unit canvas with a 14 px title and 11 px body. Included at \textwidth on A4, roughly 454 pt, that is 0.667 pt per unit, so the title
printed at 9.3 pt and the body at 7.3 pt. Figure text below 8 pt is hard to read on paper.

v4.2 raises the title to 19 and the body to 15, which print at 12.7 pt and 10.0 pt. Two boxes had to grow to hold the larger text: "Total degradation"
needs 141 units at 19 against the 122 v3 allowed, and the gradient description needs 355 units at 15 against 360. Side margins drop from 40 to 24 to pay for
it. The redundant arrow label is removed, since the same variables are named inside the outer-loop box.

Fonts
-----
The stack is Carlito, then Calibri, then DejaVu Sans. Carlito is metric compatible with Calibri, so the first two render identically. DejaVu Sans is
about 25 percent wider and will overflow the boxes. The script reports which font was resolved and checks every string against its box, so a substitution
is visible in the console rather than in the printed figure.

Usage
-----
Run from VS Code with no arguments. Outputs are written next to this file.
Requires only matplotlib.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SCRIPT_VERSION = "4.3"
OUT = Path(__file__).parent
STEM = "nested_gradient_concept_v4"

# -- Output ------------------------------------------------------------------ #
OUTPUT = "png"       # "png", "pdf" or "both"
WRITE_SVG = False     # working format, not a thesis deliverable
DPI = 300

# Colours read from nested_gradient_concept_v3.svg
PURPLE = "#6F5B7E"
BLUE = "#0076C2"
CRIMSON = "#A50034"
BROWN = "#8A5A2B"
INK = "#3C3C3B"

FONT_STACK = ["Carlito", "Calibri", "DejaVu Sans"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = FONT_STACK
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"

FS_TITLE = 19.0
FS_BODY = 15.0
FS_STEP = 15.0
FS_LEGEND = 15.0

# Canvas. One data unit renders as one point, so the on-page size of any text is
# its size here multiplied by (textwidth in pt) / W.
W, H = 680.0, 380.0
PAGE_PT = 454.0          # \textwidth on A4 with typical thesis margins

# Top row: x, width. Gaps of 64 units carry the arrows.
OUTER = (24.0, 140.0)
INNER = (228.0, 200.0)
DEGR = (492.0, 164.0)
ROW_Y, ROW_H = 42.0, 80.0          # y measured from the top edge

GRAD = (240.0, 416.0)
GRAD_Y, GRAD_H = 210.0, 80.0

_fit_report = []


def fy(y_top):
    """Convert a top-down y coordinate to matplotlib's bottom-up axes."""
    return H - y_top


def box(ax, x, y_top, w, h, fill):
    ax.add_patch(
        FancyBboxPatch((x, fy(y_top) - h), w, h,
                       boxstyle="round,pad=0,rounding_size=8",
                       linewidth=0, facecolor=fill, zorder=2)
    )


def txt(ax, x, y_top, s, size, color=INK, weight="normal", ha="center",
        box_w=None, tag=""):
    t = ax.text(x, fy(y_top), s, ha=ha, va="center", fontsize=size,
                fontweight=weight, color=color, zorder=3)
    if box_w is not None:
        _fit_report.append((tag or s, size, box_w, t))
    return t


def arrow(ax, pts, lw=1.4):
    p = [(x, fy(y)) for x, y in pts]
    for (x0, y0), (x1, y1) in zip(p, p[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=INK, lw=lw,
                solid_capstyle="round", zorder=1)
    ax.add_patch(FancyArrowPatch(p[-2], p[-1], arrowstyle="-|>",
                                 mutation_scale=11, linewidth=lw, color=INK,
                                 shrinkA=0, shrinkB=0, zorder=1))


def report(fig):
    """Print the resolved font and check every boxed string against its box."""
    resolved = findfont(FontProperties(family=FONT_STACK))
    name = Path(resolved).stem
    print(f"  resolved sans-serif font: {name}")
    if "Carlito" not in name and "calibri" not in name.lower():
        print("  WARNING: neither Carlito nor Calibri was found. Text will be")
        print("           wider than this layout allows. Install Carlito, or")
        print("           reduce FS_TITLE and FS_BODY until the check passes.")

    r = fig.canvas.get_renderer()
    scale = fig.dpi / 72.0
    print(f"  {'string':<44}{'size':>5}{'width':>8}{'box':>7}   fit")
    bad = 0
    for label, size, box_w, t in _fit_report:
        wu = t.get_window_extent(renderer=r).width / scale
        ok = wu <= box_w - 12
        bad += 0 if ok else 1
        print(f"  {label[:42]:<44}{size:>5.0f}{wu:>8.1f}{box_w:>7.0f}   "
              f"{'ok' if ok else 'OVERFLOW'}")
    ppu = PAGE_PT / W
    print(f"  on-page sizes at {PAGE_PT:.0f} pt textwidth: "
          f"title {FS_TITLE*ppu:.1f} pt, body {FS_BODY*ppu:.1f} pt")
    print(f"  {'all strings fit' if bad == 0 else f'{bad} OVERFLOWING string(s)'}")


def main():
    if OUTPUT not in ("png", "pdf", "both"):
        raise ValueError(f'OUTPUT must be "png", "pdf" or "both", not {OUTPUT!r}')

    print(f"nested_gradient_concept_v4.py  v{SCRIPT_VERSION}")
    print(f"output folder: {OUT}")

    fig, ax = plt.subplots(figsize=(W / 72.0, H / 72.0))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    ox, ow = OUTER
    ix, iw = INNER
    dx, dw = DEGR
    gx, gw = GRAD
    oc, ic, dc, gc = ox + ow / 2, ix + iw / 2, dx + dw / 2, gx + gw / 2

    # ------------------------------------------------------------- boxes
    box(ax, ox, ROW_Y, ow, ROW_H, PURPLE)
    box(ax, ix, ROW_Y, iw, ROW_H, BLUE)
    box(ax, dx, ROW_Y, dw, ROW_H, CRIMSON)
    box(ax, gx, GRAD_Y, gw, GRAD_H, BROWN)

    # ------------------------------------------------------ step captions
    txt(ax, oc, 28, "1) sizing decision", FS_STEP, weight="bold")
    txt(ax, ic, 28, "2) optimize dispatch, evaluate degradation",
        FS_STEP, weight="bold")
    txt(ax, dc, 28, "3) degradation output", FS_STEP, weight="bold")

    # -------------------------------------------------------- box content
    txt(ax, oc, 62, "Outer loop", FS_TITLE, color="white", weight="bold",
        box_w=ow, tag="Outer loop")
    txt(ax, oc, 88, r"would decide $\bar{E}$,", FS_BODY, color="white",
        box_w=ow, tag="would decide E-bar,")
    txt(ax, oc, 108, r"$\sigma_{\min}$, $\sigma_{\max}$", FS_BODY,
        color="white", box_w=ow, tag="sigma_min, sigma_max")

    txt(ax, ic, 64, "Inner loop", FS_TITLE, color="white", weight="bold",
        box_w=iw, tag="Inner loop")
    txt(ax, ic, 91, r"LP dispatch $\rightarrow$ SoC $\rightarrow$", FS_BODY,
        color="white", box_w=iw, tag="LP dispatch -> SoC ->")
    txt(ax, ic, 111, r"rainflow $\rightarrow$ degradation", FS_BODY,
        color="white", box_w=iw, tag="rainflow -> degradation")

    txt(ax, dc, 68, "Total degradation", FS_TITLE, color="white",
        weight="bold", box_w=dw, tag="Total degradation")
    txt(ax, dc, 96, r"$f_d$ accumulated", FS_BODY, color="white",
        box_w=dw, tag="f_d accumulated")

    txt(ax, gc, 232, r"NPV gradient   d NPV / d$\bar{E}$", FS_TITLE,
        color="white", weight="bold", box_w=gw, tag="NPV gradient")
    txt(ax, gc, 260,
        "how lifetime value would respond to a change in capacity,",
        FS_BODY, color="white", box_w=gw, tag="how lifetime value...")
    txt(ax, gc, 280,
        "through the degradation sensitivity of the dispatch",
        FS_BODY, color="white", box_w=gw, tag="through the degradation...")

    # ------------------------------------------------------------ arrows
    mid = ROW_Y + ROW_H / 2
    arrow(ax, [(ox + ow, mid), (ix, mid)])
    arrow(ax, [(ix + iw, mid), (dx, mid)])
    arrow(ax, [(dc, ROW_Y + ROW_H), (dc, GRAD_Y)])
    arrow(ax, [(gx, GRAD_Y + GRAD_H / 2), (oc, GRAD_Y + GRAD_H / 2),
               (oc, ROW_Y + ROW_H)])

    txt(ax, gc, 306, "4) compute gradient", FS_STEP, weight="bold")
    txt(ax, 168, 306, "5) gradient feeds back, the outer", FS_STEP,
        weight="bold")
    txt(ax, 168, 326, "loop updates the design variables", FS_STEP,
        weight="bold")

    # ------------------------------------------------------------ legend
    for x, color, label in [
        (24, PURPLE, "Sizing (outer)"),
        (188, BLUE, "Dispatch (inner)"),
        (368, CRIMSON, "Degradation output"),
        (560, BROWN, "Gradient signal"),
    ]:
        ax.add_patch(FancyBboxPatch((x, fy(365)), 15, 15,
                                    boxstyle="round,pad=0,rounding_size=3",
                                    linewidth=0, facecolor=color, zorder=2))
        txt(ax, x + 23, 357, label, FS_LEGEND, ha="left")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    report(fig)

    formats = []
    if OUTPUT in ("png", "both"):
        formats.append(("png", {"dpi": DPI}))
    if OUTPUT in ("pdf", "both"):
        formats.append(("pdf", {}))
    if WRITE_SVG:
        formats.append(("svg", {}))
    for ext, kwargs in formats:
        path = OUT / f"{STEM}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05,
                    facecolor="white", **kwargs)
        print(f"  wrote {path.name}")

    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()