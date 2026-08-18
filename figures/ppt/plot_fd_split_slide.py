r"""
plot_fd_split_slide.py
======================
Slide version of the cycle / calendar split of annual f_d.

One stacked bar per price year. STATS selects which annual statistic the bars carry, and one figure is emitted per entry. Companion to plot_fd_components.py,
which keeps the year-by-year detail for the thesis and the appendix.

Why a separate figure: the thesis figure carries 40 bars so that the year-to-year trend is available. On a slide the claim is a single share,
so the figure carries two bars and states that share directly.

Reproducible on Windows / VS Code: matplotlib + numpy + pandas, bundled DejaVu font, no LaTeX toolchain. Anchored on Path(__file__).parent.

Design (from thesis_style):
  - Stacked bar, so components use the fill_a / fill_b slots (blue = cycle, orange = calendar), matching plot_fd_components.py.
  - Price years are separated by position and axis label, not by colour, so colour continues to mean "component".
  - Zero-based y-axis.
  - A short dashed rule at half of each bar total makes the "larger half" claim readable without arithmetic.
"""
from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from degradation.style import apply_thesis_style, figsize, FS_ANNOT
from degradation.paths import RESULTS_DIR, require

P = apply_thesis_style(palette="brand", usetex=False)

# ===========================================================================
# CONFIGURATION
# ===========================================================================
HERE     = Path(__file__).parent
BRANCH   = "xu"                          # "xu" or "shi"
DATA_DIR = RESULTS_DIR / "baseline" / BRANCH
OUT_DIR  = HERE                          # figures land beside this script
STEM     = "fig_fd_split_slide"

OUTPUT = "png"        # "png", "pdf" or "both"
DPI    = 300

# Timestamp embedded in every run filename: YYYYMMDD_HHMMSS
TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")

# None takes the most recent run carrying the year tag, by that timestamp.
# Set to a timestamp substring, e.g. "20260812_000942", to pin one run.
RUN_TAG_2019 = None
RUN_TAG_2022 = None

C_CYCLE, C_CAL, C_NEU = P["fill_a"], P["fill_b"], P["neutral"]

YEARS = [("dk2019", "DK1 2019"), ("dk2022", "DK1 2022")]
BAR_W = 0.46
FS_BIG = 11          # in-bar share labels
FS_TOT = 9           # total above bar


def _present() -> str:
    names = sorted(p.name for p in DATA_DIR.glob("multiyear_trajectory_*.csv"))
    return ", ".join(names) or "(none)"


def _run_timestamp(path: Path) -> datetime:
    """Parse the YYYYMMDD_HHMMSS stamp out of a run filename."""
    m = TIMESTAMP_RE.search(path.name)
    if m is None:
        raise ValueError(f"No YYYYMMDD_HHMMSS timestamp in {path.name}")
    return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")


def find_traj(year_tag: str, run_tag: str | None) -> Path:
    """Most recent trajectory for one price year, by the timestamp the run itself wrote into the filename. Modification time is not used: git resets
    it on clone. No recursive search: a missing file should name the folder it is missing from."""
    runs = []
    for p in sorted(DATA_DIR.glob(f"multiyear_trajectory_*{year_tag}*.csv")):
        try:
            runs.append((_run_timestamp(p), p))
        except ValueError:
            print(f"  skipped, no timestamp in name: {p.name}")
    if not runs:
        raise FileNotFoundError(
            f"No 'multiyear_trajectory_*{year_tag}*.csv' in {DATA_DIR}\n"
            f"  trajectories present: {_present()}")

    runs.sort(key=lambda t: t[0], reverse=True)

    if run_tag is not None:
        hits = [r for r in runs if run_tag in r[1].name]
        if len(hits) != 1:
            raise ValueError(
                f"Run tag '{run_tag}' matches {len(hits)} {year_tag} runs.\n  "
                + "\n  ".join(p.name for _, p in runs))
        return hits[0][1]

    if len(runs) > 1:
        print(f"  {len(runs)} {year_tag} runs present, taking the most recent. "
              f"Set RUN_TAG_{year_tag[-4:]} to pin one.")
    ts, path = runs[0]
    print(f"  {year_tag} run {ts:%Y-%m-%d %H:%M:%S}")
    return path


REQUIRED_COLS = ["year", "fd_cycle", "fd_calendar"]
REPL_COL = "replacement_this_year"

# Three statistics, all from the same annual series:
#   year1        the first project year. Matches Table 4.2 of the thesis.
#   first_life   mean over the first battery, years 1 to the replacement year. The two price years replace in different
#                years, so the two bars average over different spans and the axis labels state which.
#   project_mean mean over the 20-year horizon. This spans two generations and the second is truncated, so early years
#                are counted twice. It sits within 0.2 percentage points of first_life on these runs.
STATS = ["year1", "first_life"]

STAT_LABEL = {
    "year1":        "Year-1 annual $f_d$  (-)",
    "first_life":   "Mean annual $f_d$, first battery  (-)",
    "project_mean": "20-year mean annual $f_d$  (-)",
}

STAT_STEM = {
    "year1":        f"{STEM}_year1",
    "first_life":   f"{STEM}_first_life",
    "project_mean": f"{STEM}_project_mean",
}


def load_split(path: Path, stat: str):
    """Cycle and calendar components under one statistic. Returns
    (cycle, calendar, n_years, span), where span names the years averaged."""
    need = list(REQUIRED_COLS) + ([REPL_COL] if stat == "first_life" else [])
    df = pd.read_csv(path)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(
            f"{path.name} is missing {len(missing)} column(s) the '{stat}' "
            f"statistic needs: {', '.join(missing)}.\n"
            f"  Present: {', '.join(df.columns)}")
    df = df.sort_values("year")

    if stat == "year1":
        row = df.iloc[0]
        return float(row["fd_cycle"]), float(row["fd_calendar"]), 1, "year 1"

    if stat == "first_life":
        repl = df.loc[df[REPL_COL].astype(bool), "year"].tolist()
        if not repl:
            raise ValueError(
                f"{path.name} records no replacement, so the first battery "
                f"life has no upper bound. Drop 'first_life' from STATS.")
        R = int(repl[0])
        d = df[df["year"] <= R]
        return (float(d["fd_cycle"].mean()), float(d["fd_calendar"].mean()),
                len(d), f"years 1\u2013{R}")

    if stat == "project_mean":
        last = int(df["year"].max())
        return (float(df["fd_cycle"].mean()), float(df["fd_calendar"].mean()),
                len(df), f"years 1\u2013{last}")

    raise ValueError(
        f"Unknown statistic {stat!r}; expected one of {list(STAT_LABEL)}")

def build(stat: str, paths: list[tuple[str, Path]]):
    data = []
    for label, path in paths:
        c, k, n, span = load_split(path, stat)
        data.append((label, c, k, n, span))

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.52))

    x = np.arange(len(data), dtype=float)
    cyc = np.array([d[1] for d in data])
    cal = np.array([d[2] for d in data])
    tot = cyc + cal

    ax.bar(x, cyc, BAR_W, color=C_CYCLE, edgecolor="white",
           linewidth=0.6, zorder=2, label="Cycle")
    ax.bar(x, cal, BAR_W, bottom=cyc, color=C_CAL, edgecolor="white",
           linewidth=0.6, zorder=2, label="Calendar")

    # in-bar share labels
    for xi, c, k, t in zip(x, cyc, cal, tot):
        ax.text(xi, c / 2, f"Cycle\n{100*c/t:.0f}%", ha="center",
                va="center", color="white", fontsize=FS_BIG,
                fontweight="bold", zorder=4)
        ax.text(xi, c + k / 2, f"Calendar\n{100*k/t:.0f}%", ha="center",
                va="center", color="white", fontsize=FS_BIG,
                fontweight="bold", zorder=4)
        # half-of-total rule
        ax.plot([xi - BAR_W / 2 - 0.04, xi + BAR_W / 2 + 0.04], [t / 2, t / 2],
                ls=(0, (3, 2)), lw=1.1, color=C_NEU, zorder=5)
        ax.text(xi + BAR_W / 2 + 0.07, t / 2, "half", ha="left", va="center",
                fontsize=FS_ANNOT, color=C_NEU, zorder=5)
        ax.text(xi, t * 1.03, f"{t:.4f} / yr", ha="center", va="bottom",
                fontsize=FS_TOT, color=C_NEU, zorder=4)

    # For a single year the span is identical on both bars and adds nothing.
    # For an average it differs between the price years and has to be visible.
    ax.set_xticks(x)
    ticks = ([d[0] for d in data] if stat == "year1"
             else [f"{d[0]}\n{d[4]}" for d in data])
    ax.set_xticklabels(ticks, fontsize=FS_BIG)
    ax.set_xlim(-0.62, len(data) - 0.38)
    ax.set_ylim(0, tot.max() * 1.16)
    ax.set_ylabel(STAT_LABEL[stat])
    ax.tick_params(axis="x", length=0)

    stem = STAT_STEM[stat]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT in ("pdf", "both"):
        fig.savefig(OUT_DIR / f"{stem}.pdf")
    if OUTPUT in ("png", "both"):
        fig.savefig(OUT_DIR / f"{stem}.png", dpi=DPI)
    plt.close(fig)

    print(f"\n-- caption values, {stat} --")
    for (label, c, k, n, span) in data:
        t = c + k
        basis = span if n == 1 else f"{span}, {n} yr mean"
        print(f"  {label}: {basis} | cycle {c:.5f} ({100*c/t:.1f}%) | "
              f"calendar {k:.5f} ({100*k/t:.1f}%) | total {t:.5f}")
    print(f"  saved: {OUT_DIR / stem}  ({OUTPUT})")


def main():
    if OUTPUT not in ("png", "pdf", "both"):
        raise ValueError(f"OUTPUT must be 'png', 'pdf' or 'both', got {OUTPUT!r}")
    unknown = [s for s in STATS if s not in STAT_LABEL]
    if unknown:
        raise ValueError(
            f"STATS contains {unknown}; expected values from {list(STAT_LABEL)}")

    require(DATA_DIR)
    print(f"Reading the {BRANCH} baseline from {DATA_DIR}")
    tags = {"dk2019": RUN_TAG_2019, "dk2022": RUN_TAG_2022}
    # Resolved once, so every statistic is drawn from the same pair of runs.
    paths = [(label, find_traj(tag, tags[tag])) for tag, label in YEARS]

    for stat in STATS:
        build(stat, paths)


if __name__ == "__main__":
    main()