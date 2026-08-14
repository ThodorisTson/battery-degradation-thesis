r"""
plot_soh_trajectory_combined.py
===============================
Single combined SoH figure: overlays DK1 2019 and DK1 2022 on one axis.

Reproducible on Windows / VS Code: matplotlib + numpy + pandas, DejaVu font, no LaTeX toolchain. Anchored on Path(__file__).parent.

Design (from thesis_style):
  - DK1 2022 navy, DK1 2019 dark red = primary/secondary (two-line slots).
  - Each dataset is ONE continuous same-colour line; at its replacement year it runs down-then-up (solid vertical connector, same colour) from end-of-life
    SoH back to 100%, so each price year is one run with battery swaps.
  - 80/70% EoL levels are neutral-gray dashed lines (self-labelling on the y-ticks); the IEC/warranty meaning goes in the caption.
  - One dot per simulated year.

Changes in this revision:
  1. RUN_ID_2019 / RUN_ID_2022 select the run explicitly. The old behaviour picked the alphabetically last matching file, which silently mixes cycle
     branches once more than one run exists per price year.
  2. The cycle branch of both runs is read from the matching degradation report and compared. Mixing a Shi run with an Xu run now raises instead of
     producing a figure that looks correct.
  3. End-of-life crossings are computed on the FIRST battery life only, extended one year past replacement. Under the Xu branch the 2022 battery
     is replaced at 70.05% SoH, above the 70% reporting level but below the 70.5% replacement trigger, so the 70% level is never reached while that
     battery is installed. The old routine returned None here and the script crashed on the caption print after the figure had already been written.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from degradation.paths import RESULTS_DIR, require
from degradation.style import apply_thesis_style, figsize, FS_ANNOT
P = apply_thesis_style(palette="brand", usetex=False)

# ===========================================================================
# CONFIGURATION
# Set the two run labels explicitly. Both must come from the same cycle branch. Set either to None to fall back to "latest file matching the year
# tag", which prints a warning.
# ===========================================================================
HERE   = Path(__file__).resolve().parent
BRANCH = "xu"                      # "xu" or "shi"; names the folder under results/baseline
DATA_DIR = RESULTS_DIR / "baseline" / BRANCH
STEM     = "fig_soh_trajectory_combined"

# The runs Chapter 4 reports. Both must come from the same cycle branch; the guard below reads deg_model from each degradation report and compares.
RUN_ID_2019 = "20260812_001610_dk2019_150mw_300mwh_soc10_90_baseline_rte910"
RUN_ID_2022 = "20260812_000942_dk2022_150mw_300mwh_soc10_90_baseline_rte910"

# -- Output ------------------------------------------------------------------ #
OUTPUT = "png"     # "png", "pdf" or "both"
DPI = 300
OUT_DIR = HERE

C_2022, C_2019, C_THR = P["primary"], P["secondary"], P["neutral"]
EOL_LEVELS = (80.0, 70.0)
LINE_ALPHA, LINE_LW, MARKER_MS = 0.45, 1.2, 3.5


# --------------------------------------------------------------------------
# File resolution and branch guard
# --------------------------------------------------------------------------
def resolve_traj(year_tag, run_id):
    if run_id:
        return require(DATA_DIR / f"multiyear_trajectory_{run_id}.csv")
    hits = sorted(DATA_DIR.glob(f"multiyear_trajectory_*{year_tag}*.csv"))
    if not hits:
        raise FileNotFoundError(
            f"No 'multiyear_trajectory_*{year_tag}*.csv' in {DATA_DIR}. "
            f"Set BRANCH or the RUN_ID constants.")
    if len(hits) > 1:
        print(f"  WARNING: {len(hits)} runs match '{year_tag}'; using the last "
              f"by name. Set the RUN_ID constants to choose explicitly.")
    return hits[-1]


def cycle_branch(traj_path):
    """Read 'deg_model' from the degradation report next to the trajectory."""
    rep = traj_path.parent / (traj_path.name
                              .replace("multiyear_trajectory_", "degradation_report_")
                              .replace(".csv", ".txt"))
    if not rep.exists():
        return None
    for line in rep.read_text(encoding="utf-8").splitlines():
        if line.startswith("deg_model"):
            return line.split(":", 1)[1].strip()
    return None


def check_same_branch(p19, p22):
    b19, b22 = cycle_branch(p19), cycle_branch(p22)
    if b19 is None or b22 is None:
        print("  WARNING: no degradation report found next to one of the "
              "trajectories; cycle branch not verified.")
        return b19 or b22 or "unknown"
    if b19 != b22:
        raise ValueError(
            f"Cycle branch mismatch: 2019 run is '{b19}', 2022 run is '{b22}'. "
            f"Both series in one figure must come from the same branch.")
    return b19


# --------------------------------------------------------------------------
# Data shaping
# --------------------------------------------------------------------------
def load_segments(path):
    """Return (segments, replacement_year, years, soh); each segment starts at 100%."""
    df = pd.read_csv(path).sort_values("year")
    years, soh = df["year"].to_numpy(float), df["soh_pct"].to_numpy(float)
    repl = df.loc[df["replacement_this_year"].astype(bool), "year"].tolist()
    R = int(repl[0]) if repl else int(years.max())
    m1, m2 = years <= R, years > R
    seg1 = (np.concatenate([[0], years[m1]]), np.concatenate([[100.0], soh[m1]]))
    seg2 = (np.concatenate([[R], years[m2]]), np.concatenate([[100.0], soh[m2]]))
    return [seg1, seg2], R, years, soh


def life_axis(years, soh, R):
    """Year and SoH arrays for the first battery life, from (0, 100) to the replacement year, extended one year by the last annual loss.

    The extension exists because the replacement trigger carries a tolerance: a battery can be replaced slightly above a reporting level, in which case
    that level is never reached while the battery is installed. Extending by one year gives the year at which the level would have been crossed had the
    battery stayed in service. The linear extension agrees with the SEI fade function to better than 0.01 yr over this range.
    """
    m = years <= R
    x = np.concatenate([[0.0], years[m]])
    y = np.concatenate([[100.0], soh[m]])
    last_loss = y[-2] - y[-1]
    return np.append(x, x[-1] + 1.0), np.append(y, y[-1] - last_loss)


def crossing_year(x, y, level):
    """Linear interpolation of the year at which SoH crosses `level`, on the year-end points. This is the crossing quoted in the text, NOT the
    first-year-rate extrapolation printed in the degradation report header."""
    for i in range(1, len(y)):
        if y[i - 1] >= level > y[i]:
            f = (y[i - 1] - level) / (y[i - 1] - y[i])
            return x[i - 1] + f * (x[i] - x[i - 1])
    return None


def plot_run(ax, segments, color):
    """Faint connecting line (including the vertical reset) plus crisp solid markers, so years where the two runs overlap read as two layered series."""
    x = np.concatenate([s[0] for s in segments])
    y = np.concatenate([s[1] for s in segments])
    ax.plot(x, y, color=color, lw=LINE_LW, alpha=LINE_ALPHA, zorder=3,
            solid_capstyle="round")
    ax.plot(x, y, color=color, lw=0, marker="o", markersize=MARKER_MS, zorder=3.4)


# --------------------------------------------------------------------------
def main():
    if OUTPUT not in ("png", "pdf", "both"):
        raise ValueError(f'OUTPUT must be "png", "pdf" or "both", not {OUTPUT!r}')
    file19 = resolve_traj("dk2019", RUN_ID_2019)
    file22 = resolve_traj("dk2022", RUN_ID_2022)
    branch = check_same_branch(file19, file22)

    if branch not in (None, "unknown") and branch != BRANCH:
        raise ValueError(
            f"results/baseline/{BRANCH}/ holds runs from the '{branch}' branch. "
            f"Either the files are in the wrong folder or BRANCH is wrong.")

    segs22, R22, yr22, soh22 = load_segments(file22)
    segs19, R19, yr19, soh19 = load_segments(file19)

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.50))
    for lvl in EOL_LEVELS:
        ax.axhline(lvl, color=C_THR, lw=0.9, ls=(0, (5, 4)), alpha=0.9, zorder=1)
    plot_run(ax, segs22, C_2022)   # navy under
    plot_run(ax, segs19, C_2019)   # dark red on top

    ax.set_xlim(0, 20)
    ax.set_ylim(np.floor(min(soh22.min(), soh19.min())) - 2, 101)
    ax.set_xticks(range(0, 21, 2))
    ax.set_xlabel("Project year")
    ax.set_ylabel("State of health  (%)")
    handles = [Line2D([0], [0], color=C_2022, lw=1.6, marker="o", markersize=4,
                      label="DK1 2022"),
               Line2D([0], [0], color=C_2019, lw=1.6, marker="o", markersize=4,
                      label="DK1 2019")]
    ax.legend(handles=handles, frameon=False, fontsize=FS_ANNOT,
              loc="upper right", handlelength=1.8)

    if OUTPUT in ("pdf", "both"):
        fig.savefig(OUT_DIR / f"{STEM}.pdf")
    if OUTPUT in ("png", "both"):
        fig.savefig(OUT_DIR / f"{STEM}.png", dpi=DPI)
    plt.close(fig)

    print(f"read: {file19.name}")
    print(f"read: {file22.name}")
    print(f"cycle branch: {branch}  (the calendar term is Xu in both branches)")
    print("-- caption values --")
    for tag, yr, soh, R in [("2019", yr19, soh19, R19),
                            ("2022", yr22, soh22, R22)]:
        x, y = life_axis(yr, soh, R)
        parts = []
        for lvl in EOL_LEVELS:
            c = crossing_year(x, y, lvl)
            if c is None:
                parts.append(f"{lvl:.0f}% not reached")
            elif c > R:
                parts.append(f"{lvl:.0f}% at yr {c:.2f} (after replacement)")
            else:
                parts.append(f"{lvl:.0f}% at yr {c:.2f}")
        soh_at_repl = soh[yr == R][0]
        print(f"  DK1 {tag}: " + " | ".join(parts) +
              f" | replaced end of yr {R} at {soh_at_repl:.2f}%"
              f" | year-20 SoH {soh[-1]:.2f}%")
    print(f"saved: {STEM} ({OUTPUT}) in {OUT_DIR}")


if __name__ == "__main__":
    main()