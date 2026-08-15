r"""
plot_window_levers.py
=====================
Produces Figure 4.14 only: fig_window_levers.

The two SoC window sweeps on one set of axes, so the two levers can be compared
rather than inspected one at a time.

    x   annual degradation rate f_d in year 1, Xu branch, percent per year
    y   lifetime NPV over the 20-year horizon, MEUR

Both series contain the 10 to 90 percent window, which is the single point where
the two lines meet. The direction each line travels is the finding:

    width series    up and to the right, trading wear for revenue
    center series   up and to the left, less wear at no revenue cost

The shared y axis is the point of the figure. The width series spans roughly
73 MEUR and the center series roughly 3.5 MEUR; drawn on independent axes the
two would look comparable, which they are not. Both spans are printed at
runtime so the caption can quote them from the data.

An inset magnifying the center series is available behind SHOW_INSET. It is off
in the committed figure.

INPUT
-----
The most recent window sweep in results/window_sweep/, chosen by the
YYYYMMDD_HHMMSS stamp in the filename. The fork searched recursively from the
script's own folder, which from inside figures/from_results/ can reach the whole
repository and match a file from the wrong branch. The directory is now pinned
and the run is chosen by the timestamp the run itself wrote, not by
modification time, which git resets on clone.

Output is written to figures/from_results/.

Values printed at runtime: the file used, both series ranges and spans, the
two span ratios, and a consistency check on the shared window.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from degradation.style import (apply_thesis_style, figsize, TUDELFT,
                               TEXTWIDTH_IN, FS_LABEL, FS_LEGEND, FS_ANNOT)
from degradation.paths import RESULTS_DIR, require

PALETTE = apply_thesis_style(palette="brand", usetex=False)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
OUTPUT = "png"         # "png", "pdf" or "both"
DPI    = 300

WINDOW_DIR = RESULTS_DIR / "window_sweep"
CSV_GLOB   = "window_dodsweep_*.csv"

# Timestamp embedded in every run filename: YYYYMMDD_HHMMSS
TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")

# Set to a timestamp substring, for example "20260813_163037", to pin one run
# and bypass the most-recent rule.
RUN_TAG = None

# Written here regardless of where this file sits.
OUT_DIR  = RESULTS_DIR.parent / "figures" / "from_results"
OUT_STEM = "fig_window_levers"

# The window the two series share. The width series holds the center fixed at
# BASE_CENTER, the center series holds the width fixed at BASE_WIDTH, so the
# point (BASE_CENTER, BASE_WIDTH) belongs to both.
BASE_WIDTH, BASE_CENTER = 0.80, 0.50

SHOW_INSET  = False
X_TICK_STEP = 0.2

# Fraction of \textwidth this figure is included at. It must match the width=
# argument of \includegraphics, otherwise LaTeX rescales the figure and every
# font size on the page changes with it.
INCLUDE_WIDTH_FRAC = 0.9
ASPECT = 0.58

REQUIRED_COLS = ["series", "width", "center", "fd_yr1_xu",
                 "npv_bat_multiyear_xu_MEUR"]

NAVY    = TUDELFT["navy"]
DARKRED = TUDELFT["darkred"]
BLUE    = TUDELFT["blue"]
GRID    = PALETTE["grid"]
NEUTRAL = PALETTE["neutral"]


# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════

def _run_timestamp(path: Path) -> datetime:
    """Parse the YYYYMMDD_HHMMSS stamp out of a run filename."""
    m = TIMESTAMP_RE.search(path.name)
    if m is None:
        raise ValueError(f"No YYYYMMDD_HHMMSS timestamp in {path.name}")
    return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")


def find_csv(window_dir: Path) -> Path:
    """Return the most recent window sweep, by the timestamp in its filename."""
    runs = []
    for p in sorted(window_dir.glob(CSV_GLOB)):
        try:
            runs.append((_run_timestamp(p), p))
        except ValueError:
            print(f"  skipped, no timestamp in name: {p.name}")
    if not runs:
        present = sorted(p.name for p in window_dir.glob("*.csv"))
        raise FileNotFoundError(
            f"No file matching '{CSV_GLOB}' in {window_dir}\n"
            f"  CSV files present: {', '.join(present) or '(none)'}")

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


def load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"{csv.name} is missing {len(missing)} column(s) this figure "
            f"needs: {', '.join(missing)}.\n  Present: {', '.join(df.columns)}")
    print(f"  using {csv.relative_to(RESULTS_DIR.parent)}  ({len(df)} rows)")
    return df


def window_label(center: float, width: float) -> str:
    lo = int(round((center - width / 2) * 100))
    hi = int(round((center + width / 2) * 100))
    return f"{lo}-{hi}\\%" if plt.rcParams.get("text.usetex") else f"{lo}-{hi}%"


def series(df: pd.DataFrame, kind: str):
    """Return (f_d in percent per year, NPV in MEUR, window labels)."""
    if kind == "width":
        d = df[df["series"] == "width"].sort_values("width")
        lab = [window_label(BASE_CENTER, v) for v in d["width"]]
    else:
        d = df[df["series"] == "center"].sort_values("center")
        lab = [window_label(v, BASE_WIDTH) for v in d["center"]]
    if d.empty:
        raise ValueError(f"No rows with series == '{kind}' in the CSV")
    fd = 100.0 * d["fd_yr1_xu"].to_numpy()
    npv = d["npv_bat_multiyear_xu_MEUR"].to_numpy()
    return fd, npv, lab


def shared_point(df: pd.DataFrame) -> tuple[float, float]:
    """Locate the window both series contain, and check the two agree.

    The fork found this point by taking the width-series row whose NPV was
    closest to the middle row of the center series. That is a numeric
    coincidence rather than a definition, and it moves if either series gains a
    point. The window is defined by BASE_WIDTH and BASE_CENTER, so it is looked
    up directly here and the two rows are then required to match, which also
    verifies that the two sweeps came from the same configuration.
    """
    w = df[(df["series"] == "width") & np.isclose(df["width"], BASE_WIDTH)]
    c = df[(df["series"] == "center") & np.isclose(df["center"], BASE_CENTER)]
    if len(w) != 1 or len(c) != 1:
        raise ValueError(
            f"The shared window (width {BASE_WIDTH}, center {BASE_CENTER}) "
            f"resolves to {len(w)} width row(s) and {len(c)} center row(s); "
            f"exactly one of each is required.")

    fd_w = 100.0 * float(w["fd_yr1_xu"].iloc[0])
    fd_c = 100.0 * float(c["fd_yr1_xu"].iloc[0])
    npv_w = float(w["npv_bat_multiyear_xu_MEUR"].iloc[0])
    npv_c = float(c["npv_bat_multiyear_xu_MEUR"].iloc[0])

    if not (np.isclose(fd_w, fd_c, rtol=1e-6)
            and np.isclose(npv_w, npv_c, rtol=1e-6)):
        raise ValueError(
            f"The two series disagree at the shared "
            f"{window_label(BASE_CENTER, BASE_WIDTH)} window: "
            f"f_d {fd_w:.4f} against {fd_c:.4f} percent per year, "
            f"NPV {npv_w:.4f} against {npv_c:.4f} MEUR. The two sweeps are not "
            f"the same configuration.")

    print(f"  shared window {window_label(BASE_CENTER, BASE_WIDTH)}: "
          f"f_d {fd_w:.3f} percent per year, NPV {npv_w:.2f} MEUR, "
          f"both series agree")
    return fd_w, npv_w


def draw_series(ax, fd, npv, color, marker, label, lw=1.4, ms=5):
    ax.plot(fd, npv, lw=lw, marker=marker, ms=ms, color=color,
            label=label, zorder=3)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════

def make_figure(fd_w, npv_w, lab_w, fd_c, npv_c, lab_c,
                base_fd, base_npv) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize(INCLUDE_WIDTH_FRAC, ASPECT),
                           constrained_layout=True)

    draw_series(ax, fd_w, npv_w, NAVY, "o",
                f"Width series, center {BASE_CENTER:.2f}")
    draw_series(ax, fd_c, npv_c, BLUE, "s",
                f"Center series, width {BASE_WIDTH:.2f}")

    # The window both series contain
    ax.plot([base_fd], [base_npv], lw=0, marker="o", ms=11,
            markerfacecolor="none", markeredgecolor=DARKRED,
            markeredgewidth=1.3, zorder=5)

    for xx, yy, lb in zip(fd_w, npv_w, lab_w):
        off = (0, -13) if yy < npv_w.mean() else (0, 9)
        ax.annotate(lb, (xx, yy), textcoords="offset points", xytext=off,
                    ha="center", fontsize=FS_LEGEND, color=NAVY)

    # Only the two ends of the center series are labelled; the middle point
    # carries the shared-window label from the width series above.
    for j, off in ((0, (-10, 8)), (len(fd_c) - 1, (4, -14))):
        ax.annotate(lab_c[j], (fd_c[j], npv_c[j]), textcoords="offset points",
                    xytext=off, ha="center", fontsize=FS_LEGEND, color=BLUE)

    ax.set_xlabel(r"Annual degradation rate $f_d$   (% per year)",
                  fontsize=FS_LABEL)
    ax.set_ylabel("Lifetime NPV   (MEUR)", fontsize=FS_LABEL)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.margins(x=0.10, y=0.13)
    lo = np.floor(min(fd_w.min(), fd_c.min()) / X_TICK_STEP) * X_TICK_STEP
    hi = np.ceil(max(fd_w.max(), fd_c.max()) / X_TICK_STEP) * X_TICK_STEP
    ax.set_xticks(np.arange(lo, hi + 1e-9, X_TICK_STEP))
    ax.legend(loc="upper left", frameon=False, fontsize=FS_LEGEND)

    if SHOW_INSET:
        axi = ax.inset_axes([0.55, 0.08, 0.42, 0.36], zorder=6)
        axi.set_facecolor("white")
        axi.patch.set_alpha(1.0)
        draw_series(axi, fd_c, npv_c, BLUE, "s", None, lw=1.2, ms=4)
        axi.plot([base_fd], [base_npv], lw=0, marker="s", ms=9,
                 markerfacecolor="none", markeredgecolor=DARKRED,
                 markeredgewidth=1.1, zorder=5)
        for xx, yy, lb in zip(fd_c, npv_c, lab_c):
            axi.annotate(lb, (xx, yy), textcoords="offset points",
                         xytext=(0, 7), ha="center", fontsize=FS_ANNOT - 1,
                         color=BLUE)
        axi.grid(True, color=GRID, lw=0.5, alpha=0.7)
        axi.set_axisbelow(True)
        axi.margins(x=0.20, y=0.34)
        axi.tick_params(labelsize=FS_ANNOT - 1)
        for s in ("top", "right"):
            axi.spines[s].set_visible(True)
        for s in axi.spines.values():
            s.set_linewidth(0.7)
            s.set_color(NEUTRAL)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    if OUTPUT not in ("png", "pdf", "both"):
        raise ValueError(f"OUTPUT must be 'png', 'pdf' or 'both', got {OUTPUT!r}")

    window_dir = require(WINDOW_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading the window sweep")
    df = load(find_csv(window_dir))

    fd_w, npv_w, lab_w = series(df, "width")
    fd_c, npv_c, lab_c = series(df, "center")
    base_fd, base_npv = shared_point(df)

    span_w = npv_w.max() - npv_w.min()
    span_c = npv_c.max() - npv_c.min()
    print(f"\n  width  series: NPV {npv_w.min():.2f} to {npv_w.max():.2f} MEUR "
          f"(span {span_w:.2f}), f_d {fd_w.min():.2f} to {fd_w.max():.2f} "
          f"percent per year")
    print(f"  center series: NPV {npv_c.min():.2f} to {npv_c.max():.2f} MEUR "
          f"(span {span_c:.2f}), f_d {fd_c.min():.2f} to {fd_c.max():.2f} "
          f"percent per year")
    print(f"  NPV span ratio {span_w / span_c:.1f}x, f_d span ratio "
          f"{(fd_w.max() - fd_w.min()) / (fd_c.max() - fd_c.min()):.1f}x")

    print(f"\nDrawn at {INCLUDE_WIDTH_FRAC:.2f} x textwidth = "
          f"{INCLUDE_WIDTH_FRAC * TEXTWIDTH_IN:.2f} in. Include at "
          f"width={INCLUDE_WIDTH_FRAC:.2f}\\textwidth.")

    print(f"\nWriting figure to {OUT_DIR}")
    fig = make_figure(fd_w, npv_w, lab_w, fd_c, npv_c, lab_c,
                      base_fd, base_npv)
    w, h = fig.get_size_inches()
    print(f"  canvas {w:.2f} x {h:.2f} in"
          + ("  (inset on)" if SHOW_INSET else ""))

    written = []
    if OUTPUT in ("pdf", "both"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.pdf")
        written.append("pdf")
    if OUTPUT in ("png", "both"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.png", dpi=DPI)
        written.append("png")
    plt.close(fig)
    print(f"  wrote {OUT_STEM}.{{{','.join(written)}}}")
    print("\nDone. Caption values are printed above.")


if __name__ == "__main__":
    main()