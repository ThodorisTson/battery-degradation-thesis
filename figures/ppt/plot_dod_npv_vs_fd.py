r"""
plot_dod_npv_vs_fd.py
=====================
The two SoC window sweeps as one question: what does the design earn, and
how fast does it wear.

Two figures, one per series, drawn on the same axis definitions:

  x   annual degradation f_d in year 1, Xu model, in percent per year
  y   lifetime NPV over the 20-year horizon, in MEUR

One marker per window setting, joined in sweep order. The direction the
line travels is the finding:

  width series    up and to the right, a genuine trade
  center series   up and to the left, less wear at higher value

Revenue is deliberately absent. It is already inside NPV, and showing both
raises the question of why the two do not move together.

NO FIGURE TITLES
----------------
Per the thesis convention, the figure carries no title. The held-constant
condition of each sweep therefore has to be stated by whatever plays the
role of the caption. On a slide that is the takeaway band, so use:

  width panel   "Center held at 0.50. Revenue is first-year, DK1 2022."
  center panel  "Width held at 0.80. Revenue is first-year, DK1 2022."

STYLING
-------
Everything comes from thesis_style. Note that PALETTE has no "navy" or
"dark_red" key; those are in the TUDELFT dict. Sizes are FS_LABEL for axis
labels, FS_LEGEND for point labels, FS_ANNOT for the grey notes, and the
rcParams default for ticks. Line width 1.2 and marker size 4, matching
verify_week_snapshot.py.

What the data does and does not contain
---------------------------------------
NPV is a single number per run, the discounted 20-year total. There is no
per-year NPV series. Degradation is available for year 1 only; f_d is not
constant over the horizon, so a per-year degradation line cannot be drawn
from this file. End-of-horizon state of health and the replacement year are
available and are printed to the console instead.

Reproducible on Windows / VS Code: matplotlib + numpy + pandas, bundled
DejaVu font, no LaTeX toolchain. Anchored on Path(__file__).parent.
"""
from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from degradation.style import (apply_thesis_style, TUDELFT,
                               FS_LABEL, FS_LEGEND, FS_ANNOT)
from degradation.paths import RESULTS_DIR, require

pal = apply_thesis_style(palette="brand", usetex=False)
plt.rcParams["figure.constrained_layout.use"] = False

# ===========================================================================
# CONFIGURATION
# ===========================================================================
HERE       = Path(__file__).parent
WINDOW_DIR = RESULTS_DIR / "window_sweep"
CSV_GLOB   = "window_dodsweep_*.csv"
OUT_DIR    = HERE                        # figures land beside this script

OUTPUT = "png"        # "png", "pdf" or "both"
DPI    = 300

# Timestamp embedded in every run filename: YYYYMMDD_HHMMSS
TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")

# Set to a timestamp substring, e.g. "20260813_163037", to pin one run.
RUN_TAG = None
STEM_W = "fig_window_width_npv"
STEM_C = "fig_window_center_npv"

BASE_WIDTH, BASE_CENTER = 0.80, 0.50

C_LINE = TUDELFT["navy"]
C_BASE = TUDELFT["darkred"]

FIG_W1, FIG_H1 = 5.60, 3.30        # single panel, sits beside slide callouts


def _run_timestamp(path: Path) -> datetime:
    """Parse the YYYYMMDD_HHMMSS stamp out of a run filename."""
    m = TIMESTAMP_RE.search(path.name)
    if m is None:
        raise ValueError(f"No YYYYMMDD_HHMMSS timestamp in {path.name}")
    return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")


def find_csv() -> Path:
    """Most recent window sweep, by the timestamp the run wrote into the
    filename. Same rule as plot_window_levers.py, so this slide figure and
    Figure 4.14 are always drawn from the same run."""
    require(WINDOW_DIR)
    runs = []
    for p in sorted(WINDOW_DIR.glob(CSV_GLOB)):
        try:
            runs.append((_run_timestamp(p), p))
        except ValueError:
            print(f"  skipped, no timestamp in name: {p.name}")
    if not runs:
        present = ", ".join(sorted(p.name for p in WINDOW_DIR.glob("*.csv")))
        raise FileNotFoundError(
            f"No '{CSV_GLOB}' in {WINDOW_DIR}\n"
            f"  CSV files present: {present or '(none)'}")

    runs.sort(key=lambda t: t[0], reverse=True)

    if RUN_TAG is not None:
        hits = [r for r in runs if RUN_TAG in r[1].name]
        if len(hits) != 1:
            raise ValueError(
                f"RUN_TAG '{RUN_TAG}' matches {len(hits)} runs.\n  "
                + "\n  ".join(p.name for _, p in runs))
        return hits[0][1]

    if len(runs) > 1:
        print(f"  {len(runs)} runs present, taking the most recent. "
              f"Set RUN_TAG to pin one.")
    ts, path = runs[0]
    print(f"  run {ts:%Y-%m-%d %H:%M:%S}")
    return path


def window_label(center, width):
    return f"{int(round((center - width / 2) * 100))}-" \
           f"{int(round((center + width / 2) * 100))}%"


def panel(ax, fd, npv, labels, base_i, offsets, arrow_text, arrow_xy):
    ax.plot(fd, npv, lw=1.2, marker="o", ms=4, color=C_LINE, zorder=3)
    ax.plot([fd[base_i]], [npv[base_i]], lw=0, marker="o", ms=9,
            markerfacecolor="none", markeredgecolor=C_BASE,
            markeredgewidth=1.1, zorder=4)

    for n, (xx, yy, lab) in enumerate(zip(fd, npv, labels)):
        ax.annotate(lab, (xx, yy), textcoords="offset points",
                    xytext=offsets[n] if n < len(offsets) else (0, 10),
                    ha="center", fontsize=FS_LEGEND,
                    color=C_BASE if n == base_i else C_LINE)

    ax.set_xlabel("Annual degradation rate $f_d$   (% per year)",
                  fontsize=FS_LABEL)
    ax.set_ylabel("Lifetime NPV   (MEUR)", fontsize=FS_LABEL)
    ax.grid(True, color=pal["grid"], lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    ax.annotate(arrow_text, xy=arrow_xy[0], xytext=arrow_xy[1],
                textcoords="axes fraction", xycoords="axes fraction",
                fontsize=FS_ANNOT, color=pal["neutral"], ha="center",
                va="center", alpha=0.9,
                arrowprops=dict(arrowstyle="-|>", color=pal["neutral"],
                                lw=0.8, alpha=0.8, shrinkA=2, shrinkB=2))


def _base_index(values, target, kind):
    hits = np.where(np.isclose(values, target))[0]
    if len(hits) != 1:
        raise ValueError(
            f"The {kind} series has {len(hits)} rows at {target}; exactly one "
            f"is required. Values present: {np.round(values, 3).tolist()}")
    return int(hits[0])


def _series(df, kind):
    if kind == "width":
        d = df[df["series"] == "width"].sort_values("width")
        lab = [window_label(0.50, v) for v in d["width"]]
        base = _base_index(d["width"].to_numpy(), BASE_WIDTH, "width")
    else:
        d = df[df["series"] == "center"].sort_values("center")
        lab = [window_label(v, 0.80) for v in d["center"]]
        base = _base_index(d["center"].to_numpy(), BASE_CENTER, "center")
    fd = 100 * d["fd_yr1_xu"].to_numpy()
    npv = d["npv_bat_multiyear_xu_MEUR"].to_numpy()
    return d, fd, npv, lab, base


W_OFFSETS = [(0, -16), (30, -11), (-6, 12)]
C_OFFSETS = [(0, 13), (0, -17), (34, -3), (0, -17), (0, -17)]
W_ARROW = ("wider window", ((0.74, 0.56), (0.32, 0.28)))
C_ARROW = ("lower center", ((0.10, 0.10), (0.48, 0.10)))
W_NOTE = "0-100% sits outside the range\nthe model can bound"


def save(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    if OUTPUT in ("pdf", "both"):
        fig.savefig(OUT_DIR / f"{stem}.pdf")
        written.append("pdf")
    if OUTPUT in ("png", "both"):
        fig.savefig(OUT_DIR / f"{stem}.png", dpi=DPI)
        written.append("png")
    plt.close(fig)
    print(f"saved: {OUT_DIR / stem}.{{{','.join(written)}}}")


def main():
    """Load the sweep and report the numbers behind both panels.

    No combined two-panel figure is produced. The thesis carries that comparison as Figure 4.14 (figures/from_results/plot_window_levers.py);
    the slides use the two single panels below.
    """
    csv = find_csv()
    df = pd.read_csv(csv)
    print(f"reading {csv.name}")

    for name in ("width", "center"):
        d, _, _, _, _ = _series(df, name)
        print(f"\n-- {name} series --")
        for _, r in d.iterrows():
            lab = (window_label(0.50, r["width"]) if name == "width"
                   else window_label(r["center"], 0.80))
            print(f"  {lab:>8}  fd={100*r['fd_yr1_xu']:.2f}%/yr  "
                  f"NPV={r['npv_bat_multiyear_xu_MEUR']:6.2f} MEUR  "
                  f"repl yr {int(r['first_repl_yr_xu'])}  "
                  f"SoH_end {r['final_soh_pct_xu']:.1f}%")
    return df

def single(df, kind, stem):
    """One panel on its own, sized to sit left of the callouts on a slide."""
    fig, ax = plt.subplots(figsize=(FIG_W1, FIG_H1))

    _, fd, npv, lab, base = _series(df, kind)
    if kind == "width":
        panel(ax, fd, npv, lab, base, W_OFFSETS, *W_ARROW)
        ax.annotate(W_NOTE, xy=(0.98, 0.19), xycoords="axes fraction",
                    ha="right", va="center", fontsize=FS_ANNOT,
                    color=pal["neutral"], alpha=0.9)
    else:
        panel(ax, fd, npv, lab, base, C_OFFSETS, *C_ARROW)

    ax.margins(x=0.18, y=0.22)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.97, bottom=0.155)
    save(fig, stem)


if __name__ == "__main__":
    _df = main()
    single(_df, "width", STEM_W)
    single(_df, "center", STEM_C)