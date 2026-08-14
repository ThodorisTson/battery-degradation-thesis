r"""
plot_fd_components.py
=====================
Single combined figure: annual degradation rate f_d split into its cycle and calendar components, for DK1 2019 and DK1 2022 on one axis.

Companion to plot_fd_calendar_share.py: this figure carries the absolute magnitudes, the share figure carries the trend.

Reproducible on Windows / VS Code: matplotlib + numpy + pandas, bundled DejaVu font, no LaTeX toolchain. Anchored on Path(__file__).parent.

Design (from thesis_style):
  - This IS a stacked bar, so the components use the fill_a / fill_b slots (blue = cycle, orange = calendar), the convention reserved for exactly this
    figure. The two-line slots (navy / dark red) are NOT used here; they belong to the two-year line comparisons.
  - The two price years are distinguished by position and hatching, not by colour, so colour continues to mean "component" throughout the chapter.
  - Zero-based y-axis: bar length encodes magnitude, so the baseline is not truncated.

Changes in this revision:
  1. RUN_ID_2019 / RUN_ID_2022 select the run explicitly, and the cycle branch of both runs is checked for agreement. The old "latest file by name"
     behaviour silently mixes a Shi run with an Xu run.
  2. The printout reports which component is larger in year 1 and the first year in which the cycle component overtakes the calendar component. Under
     the Xu branch the two components are close enough that this ordering is the point of the figure, and it should not be read off by eye.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from degradation.paths import RESULTS_DIR, require
from degradation.style import apply_thesis_style, figsize, FS_ANNOT
P = apply_thesis_style(palette="brand", usetex=False)

# ===========================================================================
# CONFIGURATION
# ===========================================================================
HERE   = Path(__file__).resolve().parent
BRANCH = "xu"                      # "xu" or "shi"; names the folder under results/baseline
DATA_DIR = RESULTS_DIR / "baseline" / BRANCH
STEM     = "fig_fd_components"

# The runs Chapter 4 reports. Both must come from the same cycle branch;
# the guard below reads deg_model from each degradation report and compares.
RUN_ID_2019 = "20260812_001610_dk2019_150mw_300mwh_soc10_90_baseline_rte910"
RUN_ID_2022 = "20260812_000942_dk2022_150mw_300mwh_soc10_90_baseline_rte910"

# -- Output ------------------------------------------------------------------ #
OUTPUT = "png"     # "png", "pdf" or "both"
DPI = 300
OUT_DIR = HERE

C_CYCLE, C_CAL, C_NEU = P["fill_a"], P["fill_b"], P["neutral"]
BAR_W = 0.40
HATCH_2022 = "//"


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


def load_components(path):
    """Return (years, fd_cycle, fd_calendar, fd_total, replacement_year)."""
    df = pd.read_csv(path).sort_values("year")
    repl = df.loc[df["replacement_this_year"].astype(bool), "year"].tolist()
    R = int(repl[0]) if repl else int(df["year"].max())
    return (df["year"].to_numpy(float),
            df["fd_cycle"].to_numpy(float),
            df["fd_calendar"].to_numpy(float),
            df["fd_annual"].to_numpy(float),
            R)


def first_cycle_dominant_year(years, cyc, cal, R):
    """First year of the first battery life in which cycle exceeds calendar."""
    for y, c, k in zip(years, cyc, cal):
        if y > R:
            break
        if c > k:
            return int(y)
    return None


# --------------------------------------------------------------------------
def main():
    if OUTPUT not in ("png", "pdf", "both"):
        raise ValueError(f'OUTPUT must be "png", "pdf" or "both", not {OUTPUT!r}')
    f19 = resolve_traj("dk2019", RUN_ID_2019)
    f22 = resolve_traj("dk2022", RUN_ID_2022)
    branch = check_same_branch(f19, f22)

    if branch not in (None, "unknown") and branch != BRANCH:
        raise ValueError(
            f"results/baseline/{BRANCH}/ holds runs from the '{branch}' branch. "
            f"Either the files are in the wrong folder or BRANCH is wrong.")

    y19, c19, k19, t19, R19 = load_components(f19)
    y22, c22, k22, t22, R22 = load_components(f22)

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.46))

    # 2019 = left bar, plain fill
    ax.bar(y19 - BAR_W / 2, c19, BAR_W, color=C_CYCLE,
           edgecolor="white", linewidth=0.3, zorder=2)
    ax.bar(y19 - BAR_W / 2, k19, BAR_W, bottom=c19, color=C_CAL,
           edgecolor="white", linewidth=0.3, zorder=2)

    # 2022 = right bar, hatched
    ax.bar(y22 + BAR_W / 2, c22, BAR_W, color=C_CYCLE,
           edgecolor=C_NEU, linewidth=0.45, hatch=HATCH_2022, zorder=2)
    ax.bar(y22 + BAR_W / 2, k22, BAR_W, bottom=c22, color=C_CAL,
           edgecolor=C_NEU, linewidth=0.45, hatch=HATCH_2022, zorder=2)

    ax.set_xlim(0.3, 20.7)
    ax.set_xticks(range(2, 21, 2))
    ax.set_xlabel("Project year")
    ax.set_ylabel("Annual $f_d$  (-)")

    handles = [
        Patch(facecolor=C_CYCLE, label="Cycle"),
        Patch(facecolor=C_CAL,   label="Calendar"),
        Patch(facecolor="white", edgecolor=C_NEU, linewidth=0.45,
              label="DK1 2019 (left)"),
        Patch(facecolor="white", edgecolor=C_NEU, linewidth=0.45,
              hatch=HATCH_2022, label="DK1 2022 (right)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=FS_ANNOT,
              loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4,
              handlelength=1.5, columnspacing=1.4)

    if OUTPUT in ("pdf", "both"):
        fig.savefig(OUT_DIR / f"{STEM}.pdf")
    if OUTPUT in ("png", "both"):
        fig.savefig(OUT_DIR / f"{STEM}.png", dpi=DPI)
    plt.close(fig)

    print(f"read: {f19.name}")
    print(f"read: {f22.name}")
    print(f"cycle branch: {branch}  (the calendar term is Xu in both branches)")
    print("-- caption values --")
    for tag, y, c, k, t, R in [("2019", y19, c19, k19, t19, R19),
                               ("2022", y22, c22, k22, t22, R22)]:
        n = int(R)
        larger = "cycle" if c[0] > k[0] else "calendar"
        first = first_cycle_dominant_year(y, c, k, R)
        print(f"  DK1 {tag}: cycle {c[0]:.5f} -> {c[n-1]:.5f} "
              f"({(c[n-1]/c[0]-1)*100:+.1f}%) | "
              f"calendar {k[0]:.5f} -> {k[n-1]:.5f} "
              f"({(k[n-1]/k[0]-1)*100:+.2f}%) | "
              f"total {t[0]:.5f} -> {t[n-1]:.5f} | replaced end of yr {R}")
        print(f"           larger component in year 1: {larger} | "
              f"first year cycle exceeds calendar: "
              f"{first if first else 'never within this battery life'}")
    print(f"saved: {STEM} ({OUTPUT}) in {OUT_DIR}")


if __name__ == "__main__":
    main()