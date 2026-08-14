r"""
plot_fd_calendar_share.py
=========================
Single combined figure: calendar share of the annual degradation rate f_d over the 20-year project, for DK1 2019 and DK1 2022 on one axis.

Plotting the share rather than the absolute components is the point: the absolute calendar term is close to constant by construction (it is a function
of battery age, mean state of charge and temperature), so the absolute bars carry little information. The share does move, because the cycle component
grows with capacity fade while the calendar component does not.

Reproducible on Windows / VS Code: matplotlib + numpy + pandas, bundled DejaVu font, no LaTeX toolchain. Anchored on Path(__file__).parent.

Design (from thesis_style):
  - This is a two-line comparison (2019 vs 2022), so it uses the primary/secondary slots (navy / dark red), NOT the fill_a/fill_b slots that
    the stacked-bar figures use.
  - Faint connecting line plus solid markers, matching the state-of-health trajectory figure, so the two baseline figures read as a pair.
  - Vertical connector at the replacement year, so each price year is one continuous run.
  - Neutral-gray dashed line at 50% marks an equal split.

Changes in this revision:
  1. RUN_ID_2019 / RUN_ID_2022 select the run explicitly, and the cycle branch of both runs is checked for agreement.
  2. The 50% line is forced inside the y-limits. On the Xu branch the series runs from about 50% down to about 46%, so the equal-split line sits at the
     top edge of the data instead of below it, and the old auto-limits could clip it. The line now always shows, because whether the split is above or
     below 50% is the reading the figure exists to support.
  3. The printout reports the share at year 1 and at the end of each battery life, plus whether the series crosses 50%.
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
# ===========================================================================
HERE   = Path(__file__).resolve().parent
BRANCH = "xu"                      # "xu" or "shi"; names the folder under results/baseline
DATA_DIR = RESULTS_DIR / "baseline" / BRANCH

# -- Output ------------------------------------------------------------------ #
OUTPUT = "png"     # "png", "pdf" or "both"
DPI = 300
OUT_DIR = HERE
STEM     = "fig_fd_calendar_share"

RUN_ID_2019 = "20260812_001610_dk2019_150mw_300mwh_soc10_90_baseline_rte910"
RUN_ID_2022 = "20260812_000942_dk2022_150mw_300mwh_soc10_90_baseline_rte910"

C_2022, C_2019, C_THR = P["primary"], P["secondary"], P["neutral"]
LINE_ALPHA, LINE_LW, MARKER_MS = 0.45, 1.2, 3.5
EQUAL_SPLIT = 50.0


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


def load_share(path):
    """Return (years, calendar_share_pct, replacement_year).

    The share is recomputed from the two components rather than read from the fd_calendar_pct column, which the run script rounds to one decimal.
    """
    df = pd.read_csv(path).sort_values("year")
    years = df["year"].to_numpy(float)
    share = 100.0 * df["fd_calendar"].to_numpy(float) / df["fd_annual"].to_numpy(float)
    repl = df.loc[df["replacement_this_year"].astype(bool), "year"].tolist()
    R = int(repl[0]) if repl else int(years.max())
    return years, share, R


def plot_run(ax, years, share, R, color):
    """One continuous line with a vertical connector at the replacement year."""
    m1, m2 = years <= R, years > R
    x = np.concatenate([years[m1], years[m2]])
    y = np.concatenate([share[m1], share[m2]])
    ax.plot(x, y, color=color, lw=LINE_LW, alpha=LINE_ALPHA, zorder=3,
            solid_capstyle="round")
    ax.plot(x, y, color=color, lw=0, marker="o", markersize=MARKER_MS, zorder=3.4)


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

    y22, s22, R22 = load_share(f22)
    y19, s19, R19 = load_share(f19)

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.50))
    ax.axhline(EQUAL_SPLIT, color=C_THR, lw=0.9, ls=(0, (5, 4)), alpha=0.9,
               zorder=1)

    plot_run(ax, y22, s22, R22, C_2022)
    plot_run(ax, y19, s19, R19, C_2019)

    ax.set_xlim(0, 20)
    lo = min(s22.min(), s19.min())
    hi = max(s22.max(), s19.max())
    # Keep the equal-split line inside the frame whichever side the data sits on.
    ax.set_ylim(min(np.floor(lo) - 1.0, EQUAL_SPLIT - 1.0),
                max(np.ceil(hi) + 1.0, EQUAL_SPLIT + 1.0))
    ax.set_xticks(range(0, 21, 2))
    ax.set_xlabel("Project year")
    ax.set_ylabel("Calendar share of annual $f_d$  (%)")

    handles = [Line2D([0], [0], color=C_2022, lw=1.6, marker="o", markersize=4,
                      label="DK1 2022"),
               Line2D([0], [0], color=C_2019, lw=1.6, marker="o", markersize=4,
                      label="DK1 2019")]
    ax.legend(handles=handles, frameon=False, fontsize=FS_ANNOT,
              loc="lower left", handlelength=1.8)

    if OUTPUT in ("pdf", "both"):
        fig.savefig(OUT_DIR / f"{STEM}.pdf")
    if OUTPUT in ("png", "both"):
        fig.savefig(OUT_DIR / f"{STEM}.png", dpi=DPI)
    plt.close(fig)

    print(f"read: {f19.name}")
    print(f"read: {f22.name}")
    print(f"cycle branch: {branch}  (the calendar term is Xu in both branches)")
    print("-- caption values --")
    for tag, y, sh, R in [("2019", y19, s19, R19), ("2022", y22, s22, R22)]:
        first = sh[0]
        last = sh[y <= R][-1]
        above = "above" if first > EQUAL_SPLIT else "below"
        print(f"  DK1 {tag}: share yr1 {first:.2f}% -> yr{R} {last:.2f}%  "
              f"(fall {first-last:.2f} pp) | starts {above} the equal split | "
              f"range over the project {sh.min():.2f}-{sh.max():.2f}%")
    print(f"saved: {STEM} ({OUTPUT}) in {OUT_DIR}")


if __name__ == "__main__":
    main()