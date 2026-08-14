"""
fig_subgrad_repairing.py

Schematic for Appendix C, Test 2: how a rainflow re-pairing produces a corner in the degradation cost.

The figure carries no data from a run. It is a drawn explanation of the mechanism, and the caption says so. Two panels:

  left   a short state-of-charge trace with one ambiguous excursion. The perturbed time step t is marked. Everything to the right of t lifts
         together, so only the swing crossing t changes size. The counter can read the excursion as a cycle in its own right or as part of a larger
         one, and the two readings are drawn as arcs.

  right  the resulting degradation cost as a function of the charging power at that one time step. The two readings meet at the tie, so the cost is
         continuous, but they arrive with different slopes. The two one-sided difference quotients of Equation eq:app_sg_quotients are the slopes of
         the two branches, and the sub-differential is the closed interval between them.

Reproducible in VS Code on Windows. Depends only on numpy, matplotlib and the shared style module. Output format is set by OUTPUT below; files are written
beside this script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from degradation.style import apply_thesis_style, figsize, FS_ANNOT

# -- Output ------------------------------------------------------------------ #
OUTPUT = "png"                    # "png", "pdf" or "both"
STEM = "fig_subgrad_repairing"     # output filename without extension

OUT_DIR = Path(__file__).resolve().parent
DPI = 300

# Schematic trace. Turning points only; the counter sees nothing else.
# Chosen so that the excursion marked below sits close to a tied comparison.
TURNS_X = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
TURNS_Y = np.array([0.30, 0.78, 0.58, 0.88, 0.34, 0.70])
T_MARK = 1.5           # the perturbed time step, between two turning points

# Right panel: the two branches of the cost, meeting at the tie.
SLOPE_LO = -0.55        # slope the counter gives on the low side of the tie
SLOPE_HI = 0.95         # slope on the high side
COST_AT_TIE = 0.42


def arc(ax, x0, x1, y, height, color, ls="-"):
    """Draw a shallow arc linking the two turning points of one cycle."""
    xs = np.linspace(x0, x1, 100)
    ys = y + height * np.sin(np.pi * (xs - x0) / (x1 - x0))
    ax.plot(xs, ys, color=color, lw=1.2, ls=ls, solid_capstyle="round")


def panel_trace(ax, pal):
    ax.plot(TURNS_X, TURNS_Y, color=pal["primary"], lw=1.4,
            marker="o", ms=3.2, solid_joinstyle="round", zorder=3)

    # the perturbed step, and the direction everything after it moves
    ax.plot([T_MARK, T_MARK], [0.28, 0.98], color=pal["neutral"],
            lw=0.8, ls=(0, (3, 3)), zorder=2)
    ax.text(T_MARK + 0.08, 0.45, r"step $t$", fontsize=FS_ANNOT,
            color=pal["neutral"], ha="left", va="center")
    ax.annotate("", xy=(5.1, 1.20), xytext=(1.7, 1.20),
                arrowprops=dict(arrowstyle="-|>", color=pal["neutral"],
                                lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(3.4, 1.24, "all later values shift together", fontsize=FS_ANNOT,
            color=pal["neutral"], ha="center", va="bottom")

    # The two readings are stacked so that they read top to bottom in the order
    # the caption states them: the excursion counted on its own, or absorbed.
    arc(ax, 1.0, 2.0, 0.92, 0.08, pal["fill_a"])
    ax.text(1.5, 1.04, "read as its own cycle", fontsize=FS_ANNOT,
            color=pal["fill_a"], ha="center", va="bottom")

    arc(ax, 0.0, 3.0, 0.24, -0.07, pal["fill_b"], ls=(0, (4, 2)))
    ax.text(1.5, 0.12, "or absorbed into a larger one", fontsize=FS_ANNOT,
            color=pal["fill_b"], ha="center", va="top")

    ax.set_xlim(-0.35, 5.6)
    ax.set_ylim(0.0, 1.35)
    ax.set_xticks([])
    ax.set_yticks([0.3, 0.6, 0.9])
    ax.set_xlabel("Time step")
    ax.set_ylabel("State of charge  (\u2013)")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def panel_cost(ax, pal):
    lo = np.linspace(-1.0, 0.0, 60)
    hi = np.linspace(0.0, 1.0, 60)
    ax.plot(lo, COST_AT_TIE + SLOPE_LO * lo, color=pal["fill_a"], lw=1.4)
    ax.plot(hi, COST_AT_TIE + SLOPE_HI * hi, color=pal["fill_b"], lw=1.4)
    ax.plot([0.0], [COST_AT_TIE], "o", ms=4, color=pal["primary"], zorder=4)

    # the two one-sided slopes, extended past the corner as thin guides
    ax.plot(hi, COST_AT_TIE + SLOPE_LO * hi, color=pal["fill_a"],
            lw=0.8, ls=(0, (2, 2)))
    ax.plot(lo, COST_AT_TIE + SLOPE_HI * lo, color=pal["fill_b"],
            lw=0.8, ls=(0, (2, 2)))

    ax.text(-0.98, COST_AT_TIE - SLOPE_LO * 0.98 + 0.04, r"slope $D^{-}$",
            fontsize=FS_ANNOT, color=pal["fill_a"], ha="left", va="bottom")
    ax.text(0.98, COST_AT_TIE + SLOPE_HI * 0.98 + 0.04, r"slope $D^{+}$",
            fontsize=FS_ANNOT, color=pal["fill_b"], ha="right", va="bottom")

    # the sub-differential, as the span of admissible slopes at the corner
    ax.add_patch(FancyArrowPatch((0.52, COST_AT_TIE + SLOPE_LO * 0.52),
                                 (0.52, COST_AT_TIE + SLOPE_HI * 0.52),
                                 arrowstyle="<|-|>", mutation_scale=7,
                                 lw=0.8, color=pal["neutral"], zorder=5))
    ax.text(0.60, COST_AT_TIE + 0.5 * (SLOPE_LO + SLOPE_HI) * 0.52,
            "admissible\nsub-gradients", fontsize=FS_ANNOT,
            color=pal["neutral"], ha="left", va="center")

    ax.set_xlim(-1.15, 1.55)
    ax.set_xticks([0.0])
    ax.set_xticklabels(["tie"])
    ax.set_yticks([])
    ax.set_xlabel("Charging power $p_t^{\\mathrm{ch}}$  (MW)")
    ax.set_ylabel("Degradation cost $f$  (EUR)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    if OUTPUT not in ("png", "pdf", "both"):
        raise ValueError(f'OUTPUT must be "png", "pdf" or "both", not {OUTPUT!r}')

    pal = apply_thesis_style(palette="brand", usetex=False)
    fig, axes = plt.subplots(1, 2, figsize=figsize(1.0, aspect=0.40))
    panel_trace(axes[0], pal)
    panel_cost(axes[1], pal)

    if OUTPUT in ("pdf", "both"):
        path = OUT_DIR / f"{STEM}.pdf"
        fig.savefig(path)
        print(f"saved to {path}")
    if OUTPUT in ("png", "both"):
        path = OUT_DIR / f"{STEM}.png"
        fig.savefig(path, dpi=DPI)
        print(f"saved to {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()