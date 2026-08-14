r"""
style.py -- single source of truth for figure styling in the thesis.

Call apply_thesis_style() at the top of EVERY figure script. Fonts, colours,
line widths and output format then come from one place, so the 12 matplotlib
figures stop drifting.

Reproducible on Windows / VS Code: depends only on matplotlib + numpy and uses
the bundled DejaVu Sans font (no system-font install, no LaTeX toolchain).

Two decisions are exposed as one-line toggles:
  - palette : "brand" (TU Delft, default) or "vibrant" (presentation teal/coral)
  - usetex  : False (portable, default) or True (perfect match to a LaTeX body,
              but needs a working LaTeX install on Windows -- slower, more fragile;
              keep this for the green-light -> final cosmetic pass if you want it)
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# PAGE GEOMETRY
# --------------------------------------------------------------------------- #
# Width of one \textwidth in INCHES. You MUST measure yours:
#   put  \the\textwidth  (or \showthe\textwidth) anywhere in the .tex,
#   read the value in points from the log, divide by 72.27.
# Your group's full-width figures are ~17.9 cm; the placeholder below is a
# guess for the TU Delft report template. Replace it once you have the number.
TEXTWIDTH_IN = 6.201         # measured from log (448.1309pt / 72.27)

# --------------------------------------------------------------------------- #
# COLOURS -- official TU Delft brand palette (same hexes your group code uses)
# --------------------------------------------------------------------------- #
TUDELFT = {
    "cyan":    "#00A6D6",
    "navy":    "#0C2340",
    "turq":    "#00B8C8",
    "blue":    "#0076C2",
    "purple":  "#6F1D77",
    "pink":    "#EF60A3",
    "darkred": "#A50034",
    "red":     "#E03C31",
    "orange":  "#EC6842",
    "yellow":  "#FFB81C",
    "green":   "#6CC24A",
    "dgreen":  "#009B77",
}

# Semantic slots used in the thesis. Edit the right-hand side, not the figures.
PALETTE = {
    # two-line plots (hue contrast, colour-blind OK): Xu vs Shi, 2019 vs 2022
    "primary":   TUDELFT["navy"],     # main series / Xu
    "secondary": TUDELFT["darkred"],  # contrast series / Shi
    # stacked bars (classic colour-blind-safe blue/orange): cycle vs calendar
    "fill_a":    TUDELFT["blue"],     # cycle fd
    "fill_b":    TUDELFT["orange"],   # calendar fd
    "accent":    TUDELFT["cyan"],     # third category, used sparingly
    "neutral":   "#404040",           # text, arrows, threshold lines
    "grid":      "#cccccc",
    "bg":        "#f7f9fc",
    "shade":     "#dfe6ef",           # highlight bands / fills
}

# Default categorical cycle, drawn from the brand palette, ordered for contrast.
PROP_CYCLE = [TUDELFT["navy"], TUDELFT["cyan"], TUDELFT["darkred"],
              TUDELFT["orange"], TUDELFT["green"], TUDELFT["yellow"]]

# Presentation-only alternative (the praised teal/coral schematic).
PALETTE_VIBRANT = {
    "primary":   "#1D9E75",
    "secondary": "#D85A30",
    "fill_a":    "#1D9E75",
    "fill_b":    "#D85A30",
    "accent":    "#3B6EA5",
}

# --------------------------------------------------------------------------- #
# FONT SIZES (points). Because figures are drawn at the width they are included,
# these ARE the on-page point sizes. Body text is 11 pt, so all of these sit
# just below it -- correct. To mirror your supervisor's conference sizing
# exactly, set FS_BASE=7 and FS_LABEL=8.
# --------------------------------------------------------------------------- #
FS_BASE   = 8     # ticks, default text, legend
FS_LABEL  = 9     # axis labels
FS_LEGEND = 8
FS_ANNOT  = 7     # in-figure annotations / thresholds


def apply_thesis_style(palette: str = "brand", usetex: bool = False) -> dict:
    """Apply the shared style. Returns the active PALETTE dict for convenience."""
    if palette == "vibrant":
        PALETTE.update(PALETTE_VIBRANT)

    mpl.rcParams.update({
        # ---- font -------------------------------------------------------- #
        "font.family":      "sans-serif",
        "font.sans-serif":  ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",   # internal consistency of delta/sigma/Phi
        "text.usetex":      usetex,          # True only with a working LaTeX install
        "font.size":        FS_BASE,
        "axes.labelsize":   FS_LABEL,
        "axes.titlesize":   FS_LABEL,        # titles are OFF, but keep sane
        "xtick.labelsize":  FS_BASE,
        "ytick.labelsize":  FS_BASE,
        "legend.fontsize":  FS_LEGEND,
        # ---- lines / spines --------------------------------------------- #
        "axes.linewidth":   0.6,
        "lines.linewidth":  1.2,
        "lines.markersize": 4,
        "grid.linewidth":   0.6,
        "grid.color":       PALETTE["grid"],
        "grid.alpha":       0.7,
        "axes.grid":        False,           # your style: no grid by default
        # ---- colours / background --------------------------------------- #
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   PALETTE["neutral"],
        "axes.prop_cycle":  mpl.cycler(color=PROP_CYCLE),
        # ---- ticks ------------------------------------------------------- #
        "xtick.direction":  "out",
        "ytick.direction":  "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        # ---- layout + output -------------------------------------------- #
        "figure.constrained_layout.use": True,  # margins without bbox='tight'
        "savefig.bbox":     "standard",          # NOT 'tight' -> width stays fixed
        "savefig.dpi":      300,                 # only matters for PNG previews
        "pdf.fonttype":     42,                  # embed editable TrueType fonts
        "ps.fonttype":      42,
    })
    return PALETTE


def figsize(width_frac: float = 1.0, aspect: float = 0.60) -> tuple[float, float]:
    r"""Figure size in inches for a figure occupying `width_frac` of \textwidth.

    aspect = height / width. Draw at the include width so that matplotlib points
    equal on-page points (no LaTeX scaling). Examples:
        figsize(1.0)        -> full text width
        figsize(0.7, 0.75)  -> include with width=0.7\textwidth, taller aspect
    """
    w = TEXTWIDTH_IN * width_frac
    return (w, w * aspect)
