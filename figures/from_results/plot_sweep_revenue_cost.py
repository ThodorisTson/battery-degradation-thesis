r"""
plot_sweep_revenue_cost.py
==========================
Produces Figure 4.13 only: fig_sweep_revenue_cost_p175.

Discounted revenue, discounted cost, and the rates that set the optimum, as
functions of battery energy capacity at fixed power capacity. DK1 2022 prices.
Two stacked panels:

    (a) revenue and cost levels, with the replacement component shaded
    (b) change in lifetime NPV and in the lifetime cost of degradation per
        added MWh

Panel (b) is what earns the figure. The cost of degradation per added MWh is
nearly flat while the no-degradation NPV per added MWh falls through it near
E = 550 MWh, so the crossing locates the optimum rather than merely describing
it.

METHOD
------
The revenue and cost components are reconstructed from the NPV identity used by
the sweep script:

    NPV = -capex + sum_k rev_k (1+r)^-k - sum_repl repl_cost (1+r)^-k_repl

so that

    discounted revenue, no degradation = npv_no_deg + capex
    discounted revenue, Xu             = npv_with_xu + capex + discounted replacement
    discounted replacement, Xu         = n_repl (repl_e E + repl_p P) (1+r)^-eol_year

Cost conventions follow the capex and replacement_cost definitions in the
economics module.

INPUTS
------
Discovered in results/sizing_sweep/, using the same rule as
plot_sweep_thesis.py: the two most recent runs by the YYYYMMDD_HHMMSS stamp in
the filename, classified as coarse and refined by measured grid extent. The
refined pass takes precedence on shared design points.

The discovery helpers below are duplicated from plot_sweep_thesis.py. Two
copies of the same rule is one copy too many; they belong in the degradation
package, as degradation/sweeps.py, in a later consolidation pass.

Outputs are written to figures/from_results/.

Values printed at runtime: the two files selected, the full component table on
the slice, the crossing interval in panel (b), and the canvas size.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from degradation.style import (apply_thesis_style, figsize, TUDELFT,
                               TEXTWIDTH_IN, FS_LABEL, FS_ANNOT)
from degradation.paths import RESULTS_DIR, require

PALETTE = apply_thesis_style(palette="brand", usetex=False)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
OUTPUT = "png"         # "png", "pdf" or "both"
DPI    = 300

SWEEP_DIR = RESULTS_DIR / "sizing_sweep"
CSV_GLOB  = "*lifetime_sweep*.csv"

# Written here regardless of where this file sits.
OUT_DIR  = RESULTS_DIR.parent / "figures" / "from_results"
OUT_STEM = "fig_sweep_revenue_cost_p175"

# Run selection. Leave both as None to discover the two most recent runs.
COARSE_TAG = None
ZOOM_TAG   = None

TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")

P_FIXED_MW = 175.0     # power capacity of the slice
E_OPT_MWH  = 550.0     # Xu grid optimum, marked with a dotted line

DISCOUNT_RATE      = 0.03
REPL_E_EUR_PER_MWH = 72.0e3    # DEA energy expansion cost, 72 EUR per kWh
REPL_P_EUR_PER_MW  = 96.0e3    # DEA power expansion cost, 96 EUR per kW

# Panel (b) starts here. Below 450 MWh the coarse grid steps by 150 MWh, so its
# difference quotient is not comparable with the 50 MWh steps above and is left
# out of the rate panel.
SLOPE_E_MIN = 450.0

# Fraction of \textwidth this figure is included at. It must match the width=
# argument of \includegraphics, otherwise LaTeX rescales the figure and every
# font size on the page changes with it.
INCLUDE_WIDTH_FRAC = 0.9
ASPECT = 1.00          # height over width, two stacked panels

# Columns this script needs beyond the NPV columns.
REQUIRED_COLS = ["e_cap", "p_cap", "npv_no_deg", "npv_with_xu",
                 "capex_eur", "eol_year_xu", "n_repl_xu"]

NAVY    = TUDELFT["navy"]
DARKRED = TUDELFT["darkred"]
GREY    = PALETTE["neutral"]

# This figure hides the top and right spines and uses slightly heavier lines
# than the rest of the thesis. It is the only figure that does so; the
# deviation is deliberate and predates the migration.
mpl.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   1.3,
    "lines.markersize":  4,
})


# ═══════════════════════════════════════════════════════════════════════════
# RUN DISCOVERY  (duplicated from plot_sweep_thesis.py, see header)
# ═══════════════════════════════════════════════════════════════════════════

def _run_timestamp(path: Path) -> datetime:
    """Parse the YYYYMMDD_HHMMSS stamp out of a run filename."""
    m = TIMESTAMP_RE.search(path.name)
    if m is None:
        raise ValueError(f"No YYYYMMDD_HHMMSS timestamp in {path.name}")
    return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")


def _list_runs(sweep_dir: Path) -> list[tuple[datetime, Path]]:
    """Every sweep CSV in `sweep_dir`, newest first, by embedded timestamp.

    File modification time is deliberately not used. Git sets it to the clone
    time in arbitrary order, so an ordering based on it would differ between
    machines. The timestamp in the filename is written by the run itself.
    """
    runs = []
    for p in sweep_dir.glob(CSV_GLOB):
        try:
            runs.append((_run_timestamp(p), p))
        except ValueError:
            print(f"  skipped, no timestamp in name: {p.name}")
    runs.sort(key=lambda t: t[0], reverse=True)
    return runs


def _pin(runs: list[tuple[datetime, Path]], tag: str) -> tuple[datetime, Path]:
    """Select the one run whose filename contains `tag`."""
    hits = [r for r in runs if tag in r[1].name]
    if len(hits) == 1:
        return hits[0]
    listing = "\n    ".join(p.name for _, p in runs) or "(none)"
    if not hits:
        raise FileNotFoundError(
            f"No sweep CSV matching '{tag}'.\n  Runs found:\n    {listing}")
    raise ValueError(
        f"'{tag}' matches {len(hits)} runs. Use a longer tag.\n"
        f"  Matches:\n    " + "\n    ".join(p.name for _, p in hits))


def _check_columns(df: pd.DataFrame, name: str) -> None:
    """Fail with the missing column names rather than a bare KeyError later."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"{name} is missing {len(missing)} column(s) this figure needs: "
            f"{', '.join(missing)}.\n  Present: {', '.join(df.columns)}")


def select_runs(sweep_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (coarse, refined) sweep frames, in EUR as stored on disk.

    With both tags unset, the two most recent runs are taken and the one
    covering the larger area of the E-P plane is the coarse sweep.
    """
    runs = _list_runs(sweep_dir)
    if not runs:
        raise FileNotFoundError(f"No file matching '{CSV_GLOB}' in {sweep_dir}")

    if COARSE_TAG is not None and ZOOM_TAG is not None:
        picked, classify = [_pin(runs, COARSE_TAG), _pin(runs, ZOOM_TAG)], False
    else:
        if len(runs) < 2:
            raise FileNotFoundError(
                f"Need two sweep runs in {sweep_dir}, found {len(runs)}: "
                f"{runs[0][1].name}")
        picked, classify = runs[:2], True
        if len(runs) > 2:
            print(f"  {len(runs)} runs present, taking the two most recent. "
                  f"Set COARSE_TAG and ZOOM_TAG to pin specific runs.")

    frames = []
    for ts, p in picked:
        df = pd.read_csv(p)
        _check_columns(df, p.name)
        frames.append((ts, p, df))
        print(f"  {ts:%Y-%m-%d %H:%M:%S}  {p.name}  ({len(df)} rows)")

    def area(d):
        return ((d.e_cap.max() - d.e_cap.min())
                * (d.p_cap.max() - d.p_cap.min()))

    if classify:
        frames.sort(key=lambda f: area(f[2]), reverse=True)
    (_, path_c, df_coarse), (_, path_z, df_zoom) = frames
    print(f"  coarse  : {path_c.name}")
    print(f"  refined : {path_z.name}")
    return df_coarse, df_zoom


# ═══════════════════════════════════════════════════════════════════════════
# DATA ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════

def load_pooled(coarse: pd.DataFrame, refined: pd.DataFrame) -> pd.DataFrame:
    """Pool the two passes, with the refined pass taking precedence."""
    key = ["e_cap", "p_cap"]
    refined_index = refined.set_index(key).index
    coarse_only = coarse[~coarse.set_index(key).index.isin(refined_index)]

    pooled = pd.concat([refined, coarse_only], ignore_index=True)
    pooled = pooled.dropna(subset=["npv_with_xu"])
    print(f"  pooled: {len(refined)} refined + {len(coarse_only)} coarse-only "
          f"= {len(pooled)} rows")
    return pooled.sort_values(["p_cap", "e_cap"]).reset_index(drop=True)


def add_components(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct discounted revenue and cost components, in MEUR."""
    out = df.copy()

    repl_cost_eur = (REPL_E_EUR_PER_MWH * out["e_cap"]
                     + REPL_P_EUR_PER_MW * out["p_cap"])
    discount = (1.0 + DISCOUNT_RATE) ** (-out["eol_year_xu"].astype(float))
    out["repl_disc"] = out["n_repl_xu"] * repl_cost_eur * discount / 1e6

    out["capex"] = out["capex_eur"] / 1e6
    out["npv_nodeg"] = out["npv_no_deg"] / 1e6
    out["npv_xu"] = out["npv_with_xu"] / 1e6

    out["rev_nodeg"] = out["npv_nodeg"] + out["capex"]
    out["rev_xu"] = out["npv_xu"] + out["capex"] + out["repl_disc"]
    out["cost_xu"] = out["capex"] + out["repl_disc"]
    out["deg_cost"] = out["npv_nodeg"] - out["npv_xu"]
    return out


def midpoint_slopes(x: np.ndarray, y: np.ndarray):
    """Forward-difference slopes and the midpoints they apply to."""
    xm = 0.5 * (x[:-1] + x[1:])
    return xm, np.diff(y) / np.diff(x)


def report_crossing(xm: np.ndarray, a: np.ndarray, b: np.ndarray) -> None:
    """Print the interval where curve `a` falls through curve `b`."""
    d = a - b
    sign_change = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if len(sign_change) == 0:
        print("  panel (b): the two rate curves do not cross on the plotted "
              "range")
        return
    for i in sign_change:
        x0, x1 = xm[i], xm[i + 1]
        # Linear interpolation of the difference to zero
        xc = x0 - d[i] * (x1 - x0) / (d[i + 1] - d[i])
        print(f"  panel (b): rates cross between E = {x0:.0f} and "
              f"{x1:.0f} MWh, interpolated at E = {xc:.0f} MWh")
    print(f"  panel (b): cost of degradation per added MWh ranges "
          f"{b.min():.3f} to {b.max():.3f} MEUR/MWh over the plotted range")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════

def make_figure(sec: pd.DataFrame) -> plt.Figure:
    e = sec["e_cap"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1,
                             figsize=figsize(INCLUDE_WIDTH_FRAC, ASPECT),
                             sharex=True, constrained_layout=True)

    # -- (a) revenue and cost levels -----------------------------------------
    ax = axes[0]
    ax.plot(e, sec["rev_nodeg"], color=NAVY, marker="o",
            label="Battery revenue, no degradation")
    ax.plot(e, sec["rev_xu"], color=NAVY, marker="o", linestyle="--",
            markerfacecolor="white", label="Battery revenue, Xu")
    ax.plot(e, sec["cost_xu"], color=DARKRED, marker="s",
            label="Capital and replacement cost, Xu")
    ax.plot(e, sec["capex"], color=DARKRED, marker="s", linestyle="--",
            markerfacecolor="white", label="Capital cost")
    ax.fill_between(e, sec["capex"], sec["cost_xu"], color=DARKRED,
                    alpha=0.12, linewidth=0)
    ax.set_ylabel("Present value  (MEUR)")
    ax.legend(loc="upper left", frameon=False)

    # -- (b) rates ------------------------------------------------------------
    ax = axes[1]
    mask = e >= SLOPE_E_MIN
    xm, s_nodeg = midpoint_slopes(e[mask], sec["npv_nodeg"].to_numpy(float)[mask])
    _, s_deg = midpoint_slopes(e[mask], sec["deg_cost"].to_numpy(float)[mask])
    report_crossing(xm, s_nodeg, s_deg)

    ax.axhline(0.0, color=GREY, linewidth=0.7, linestyle="--")
    ax.plot(xm, s_nodeg, color=NAVY, marker="o",
            label="Lifetime NPV, no degradation")
    ax.plot(xm, s_deg, color=DARKRED, marker="s",
            label="Lifetime cost of degradation")
    ax.set_ylabel("Change per added MWh  (MEUR/MWh)")
    ax.set_xlabel("Energy capacity  E  (MWh)")
    ax.legend(loc="upper right", frameon=False)

    for ax, lab in zip(axes, ["(a)", "(b)"]):
        ax.axvline(E_OPT_MWH, color=DARKRED, linestyle=":", linewidth=1.0)
        ax.margins(x=0.02)
        ax.text(0.0, 1.015, lab, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=FS_LABEL)

    ax_b = axes[1]
    ax_b.text(E_OPT_MWH - 16,
              ax_b.get_ylim()[0] + 0.12 * np.ptp(ax_b.get_ylim()),
              f"E* = {E_OPT_MWH:.0f} MWh", color=DARKRED, fontsize=FS_ANNOT,
              ha="right", va="center")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    if OUTPUT not in ("png", "pdf", "both"):
        raise ValueError(f"OUTPUT must be 'png', 'pdf' or 'both', got {OUTPUT!r}")

    sweep_dir = require(SWEEP_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSelecting sweep runs")
    df_coarse, df_zoom = select_runs(sweep_dir)
    pooled = add_components(load_pooled(df_coarse, df_zoom))

    sec = pooled[np.isclose(pooled["p_cap"], P_FIXED_MW)].sort_values("e_cap")
    if sec.empty:
        raise SystemExit(f"No rows at P = {P_FIXED_MW} MW in either sweep")
    print(f"  slice at P = {P_FIXED_MW:.0f} MW: {len(sec)} points, "
          f"E from {sec.e_cap.min():.0f} to {sec.e_cap.max():.0f} MWh")

    cols = ["e_cap", "eol_year_xu", "rev_nodeg", "rev_xu", "capex",
            "repl_disc", "cost_xu", "npv_nodeg", "npv_xu", "deg_cost"]
    print("\nComponent table on the slice (MEUR)")
    print(sec[cols].round(2).to_string(index=False))

    print(f"\nDrawn at {INCLUDE_WIDTH_FRAC:.2f} x textwidth = "
          f"{INCLUDE_WIDTH_FRAC * TEXTWIDTH_IN:.2f} in. Include at "
          f"width={INCLUDE_WIDTH_FRAC:.2f}\\textwidth.")

    print(f"\nWriting figure to {OUT_DIR}")
    fig = make_figure(sec)
    w, h = fig.get_size_inches()
    print(f"  canvas {w:.2f} x {h:.2f} in")

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
