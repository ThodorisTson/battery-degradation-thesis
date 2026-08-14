"""
fig_week_rainflow_redesign.py
-----------------------------
Figure 4.4, the rainflow cycles of week 20, drawn so that the counted cycles can be matched to the state-of-charge trajectory.

Every counted record is drawn in STATE-OF-CHARGE coordinates, on the same y-axis as the trajectory, instead of on a cycle-amplitude axis. A record is drawn as an
I-beam: a vertical tick at each of its two turning points spanning the two state-of-charge levels it joins, connected by a horizontal line at the record's
mean state of charge. The vertical extent is the cycle amplitude delta_i and the horizontal line sits at the mean state of charge sigma_bar_i, which are the two
stress inputs of the Xu model.

This is not the figure produced by verification/verify_week_snapshot.py. That script writes fig_week_rainflow.png, on a cycle-amplitude axis, which is the earlier 
design and is not used in the thesis. Figure 4.4 is the _A file written here. A second candidate, Design B, added a zoom panel on hours 18 to 58 and was not adopted; 
it was removed in version 1.2.

Records whose midpoint falls outside the week are skipped, which is the three straddling cycles the caption of Figure 4.4 mentions.

Inputs, from results/week_snapshot/, written by verify_week_snapshot.py:
    week_snapshot_cache.npz    keys: e__0.10_0.90, e_cap
    week_snapshot_cycles.csv   published cycle list

Output, beside this script, format selected by OUTPUT:
    fig_week_rainflow_A.pdf / .png

Run:  python figures/from_results/fig_week_rainflow_redesign.py
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from degradation.paths import RESULTS_DIR, require
from degradation.style import (apply_thesis_style, figsize, TUDELFT,
                               FS_ANNOT, FS_LEGEND)

SCRIPT_VERSION = "1.1"

HERE = Path(__file__).resolve().parent

# Written by verification/verify_week_snapshot.py.
SNAPSHOT_DIR = RESULTS_DIR / "week_snapshot"
CACHE  = require(SNAPSHOT_DIR / "week_snapshot_cache.npz")
CYCLES = require(SNAPSHOT_DIR / "week_snapshot_cycles.csv")

# -- Output ------------------------------------------------------------------ #
OUTPUT = "png"     # "png", "pdf" or "both"
DPI = 300
OUT_DIR = HERE


def _save(fig, stem):
    """Write the formats OUTPUT asks for, beside this script."""
    if OUTPUT in ("pdf", "both"):
        fig.savefig(OUT_DIR / f"{stem}.pdf")
    if OUTPUT in ("png", "both"):
        fig.savefig(OUT_DIR / f"{stem}.png", dpi=DPI)
    plt.close(fig)

WINDOW_CACHE_KEY = "e__0.10_0.90"
WINDOW_CSV_KEY = "10-90%"
SOC_MIN, SOC_MAX = 0.10, 0.90
WEEK_START, WEEK_END = 3192, 3360


def load_records():
    """Return week and adjacent records as dicts with turning-point hours."""
    recs = []
    with open(CYCLES, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["window"] != WINDOW_CSV_KEY:
                continue
            recs.append({
                "scope": r["scope"],
                "h1": float(r["hour_in_week_start"]),
                "h2": float(r["hour_in_week_end"]),
                "delta": float(r["depth"]),
                "mean": float(r["mean_soc"]),
                "n": float(r["count"]),
            })
    return recs


def draw_records(ax, recs, soc, pal, zoom=None, label_event=False):
    """Draw one vertical bar per counted record, in state-of-charge coordinates.

    The bar spans the two state-of-charge levels the record joins, so its height is the cycle amplitude delta_i and the dot marks the mean state of charge
    sigma_bar_i. It is placed at the midpoint in time of the two turning points. A record has no duration, so no horizontal extent is drawn.
    """
    cap = 1.6 if zoom is None else 0.7      # half-width of the end caps, in hours

    for rec in sorted(recs, key=lambda r: -r["delta"]):
        lo = rec["mean"] - 0.5 * rec["delta"]
        hi = rec["mean"] + 0.5 * rec["delta"]
        if rec["scope"] == "adjacent":
            continue          # midpoint lies outside the week; see caption
        if rec["n"] == 1.0:
            colour, alpha, ls, lw, z = TUDELFT["darkred"], 0.95, "-", 1.4, 4
        else:
            colour, alpha, ls, lw, z = TUDELFT["blue"], 0.95, (0, (2.5, 1.8)), 1.4, 3

        xm = 0.5 * (rec["h1"] + rec["h2"])
        ax.plot([xm, xm], [lo, hi], color=colour, lw=lw, ls=ls, alpha=alpha,
                zorder=z, solid_capstyle="butt")
        for lvl in (lo, hi):
            ax.plot([xm - cap, xm + cap], [lvl, lvl], color=colour, lw=lw,
                    alpha=alpha, zorder=z, solid_capstyle="butt")
        ax.plot([xm], [rec["mean"]], marker="o", ms=2.6, color=colour,
                alpha=alpha, zorder=z + 0.2)

    if label_event:
        ax.annotate("full cycle\n$\\delta_i = 0.230$", xy=(43.5, 0.72),
                    xytext=(51.0, 0.56), fontsize=FS_ANNOT,
                    color=TUDELFT["darkred"], ha="center", va="top",
                    arrowprops=dict(arrowstyle="-", lw=0.7,
                                    color=TUDELFT["darkred"]))
        ax.annotate("half-cycle\n$\\delta_i = 0.800$", xy=(37.0, 0.30),
                    xytext=(26.5, 0.20), fontsize=FS_ANNOT,
                    color=TUDELFT["blue"], ha="center", va="top",
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=TUDELFT["blue"]))


def draw_trajectory(ax, soc, pal, turning=None, zoom=None):
    n = soc.size
    for lvl in (SOC_MIN, SOC_MAX):
        ax.axhline(lvl, color=pal["neutral"], lw=0.7, ls=(0, (5, 4)), alpha=0.7,
                   zorder=1)
    ax.plot(np.arange(n), soc, color=TUDELFT["navy"], lw=1.2, zorder=3)
    if turning is not None and len(turning):
        ts = np.array(sorted(turning))
        ts = ts[(ts >= 0) & (ts < n)]
        ax.plot(ts, soc[ts.astype(int)], linestyle="none", marker="o", ms=2.8,
                mfc="white", mec=TUDELFT["navy"], mew=0.9, zorder=4)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


DELTA_C = 0.1437


def delta_c_marker(ax, pal, x, y0=0.06):
    """Vertical scale bar of height delta_c, since delta is a height here."""
    ax.plot([x, x], [y0, y0 + DELTA_C], color=pal["neutral"], lw=1.0, zorder=5)
    for lvl in (y0, y0 + DELTA_C):
        ax.plot([x - 1.6, x + 1.6], [lvl, lvl], color=pal["neutral"], lw=1.0,
                zorder=5)
    ax.annotate("$\\delta_c$", xy=(x, y0 + DELTA_C), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=FS_ANNOT,
                color=pal["neutral"])


def legend_handles(pal):
    return [
        Line2D([0], [0], color=TUDELFT["darkred"], lw=1.3,
               label="Full cycle ($n_i = 1$)"),
        Line2D([0], [0], color=TUDELFT["blue"], lw=1.3, ls=(0, (3, 2)),
               label="Half-cycle ($n_i = 0.5$)"),
        Line2D([0], [0], color=TUDELFT["navy"], lw=0, marker="o", ms=2.8,
               mfc="white", mec=TUDELFT["navy"], mew=0.9, label="Turning point"),
    ]


def main() -> None:
    if OUTPUT not in ("png", "pdf", "both"):
        raise ValueError(f'OUTPUT must be "png", "pdf" or "both", not {OUTPUT!r}')
    pal = apply_thesis_style(palette="brand", usetex=False)

    z = np.load(CACHE, allow_pickle=True)
    soc = (z[WINDOW_CACHE_KEY] / float(z["e_cap"][0]))[WEEK_START:WEEK_END]
    n = soc.size

    recs = load_records()
    week = [r for r in recs if r["scope"] == "week"]
    turning = sorted({int(round(h)) for r in week for h in (r["h1"], r["h2"])})

    xticks = [0, 24, 48, 72, 96, 120, 144, 168]

    # ---------------- Design A: full week, two panels ----------------------
    fig, ax = plt.subplots(2, 1, figsize=figsize(1.0, aspect=0.62), sharex=True)

    draw_trajectory(ax[0], soc, pal, turning=turning)
    ax[0].set_ylabel("State of charge\n(-)")

    draw_records(ax[1], recs, soc, pal)
    for lvl in (SOC_MIN, SOC_MAX):
        ax[1].axhline(lvl, color=pal["neutral"], lw=0.7, ls=(0, (5, 4)),
                      alpha=0.5, zorder=1)
    ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[1].set_ylabel("Counted cycle range\n(state of charge)")
    ax[1].set_xlabel("Hour of week 20")
    ax[1].set_xlim(-5, 177)
    ax[1].set_xticks(xticks)
    delta_c_marker(ax[1], pal, x=173.0)
    ax[1].legend(handles=legend_handles(pal), frameon=False, fontsize=FS_LEGEND,
                 loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)

    _save(fig, "fig_week_rainflow_A")

    print(f"fig_week_rainflow_redesign.py v{SCRIPT_VERSION}")
    print(f"  records drawn : {len(week)} in week, "
          f"{len(recs) - len(week)} adjacent")
    print(f"  written to    : {OUT_DIR}")


if __name__ == "__main__":
    main()
