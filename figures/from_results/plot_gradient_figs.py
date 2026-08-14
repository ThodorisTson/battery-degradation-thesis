r"""
plot_gradient_figs.py

Produces the three figures of the degradation gradient subsection:

    Figure C.4  plot_gradient_bars     normalized g_k as bars, shaded by alignment regime
    (not used)  plot_gradient_lines    the two decomposition terms as clean lines
    Figure C.5  plot_gradient_overlay  C.4's bars at low opacity behind the lines

The middle figure is kept because the overlay is built from it, but the thesis
uses only the first and the third.

All three share one normalized y-axis (value / annual mean), so the overlay lines
up by construction and no figure carries the EUR/MWh axis that does not survive a
dimensional check on g_k. The absolute mean is reported in the caption instead.

Decomposition
-------------
    g_k = -E_k <s_k, lambda_k>
        = ( E_k |s_bar_k| |lambda_bar_k| )  x  ( <s_k, lambda_k> / (|s_bar_k| |lambda_bar_k|) )
          \_________ magnitude _________/      \_____________ alignment _____________________/

magnitude : declines slowly with capacity, weak correlation with g_k
alignment : oscillates, strong correlation with g_k (this is what drives the swing)

Data layout in multiyear_*.npy, per-year tuple (slim, g[:6]):
    [0] g_k    [1] mean |subgradient|    [3] mean |dual|    [4] effective capacity

Reproducible on Windows / VS Code: matplotlib + numpy only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from degradation.paths import RESULTS_DIR as REPO_RESULTS_DIR, require
from degradation.style import apply_thesis_style, figsize, TUDELFT

# -- Output ------------------------------------------------------------------ #
OUTPUT = "both"     # "png", "pdf" or "both"
DPI = 300

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = REPO_RESULTS_DIR / "baseline" / "shi"

# The Shi branch, not the Xu branch: the per-year gradient is computed only when
# deg_model != "xu" in run_baseline.py, because the sub-gradient is built on the
# fitted polynomial. The Xu run in results/baseline/xu/ carries
# annual_gradient = None for every year, which is expected rather than a fault.
NPY_FILE   = require(RESULTS_DIR /
    "multiyear_20260814_223617_dk2022_150mw_300mwh_soc10_90_baseline_rte910.npy")
N_GEN      = 12          # first battery generation length (years before first replacement)
PRICE_TAG  = "dk1_2022"

# One colour for the bars. Earlier versions shaded them by alignment regime,
# dark red above the mean and blue below. That encoding was built on the
# superseded half-cycle attribution, under which the alignment regime and the
# bar height agreed in all twelve years. Under the spanning-set construction
# they disagree in three of twelve, so the colour would contradict the height.
# The decomposition is carried by the two lines of the overlay instead.
C_BAR   = TUDELFT["navy"]      # g_k bars
C_ALIGN = TUDELFT["navy"]      # alignment term line
C_MAG   = TUDELFT["orange"]    # magnitude term line
C_REF   = "#404040"            # 1.0 reference / mean line

# --- tune the OVERLAY here (Fig 4.6) --------------------------------------- #
OVERLAY_BAR_ALPHA = 0.18       # 0.12 (fainter) .. 0.25 (more prominent)
OVERLAY_BAR_WIDTH = 0.82       # 0.65 (column feel) .. 0.95 (solid bands)
# --------------------------------------------------------------------------- #

BARS_BAR_WIDTH = 0.72          # Fig 4.4 standalone bar width (full opacity)

PALETTE = apply_thesis_style(palette="brand")


# --------------------------------------------------------------------------- #
# DATA
# --------------------------------------------------------------------------- #
def load_decomposition(npy_file: Path, n_gen: int) -> dict:
    d  = np.load(npy_file, allow_pickle=True).item()
    ag = d["annual_gradient"][:n_gen]
    if any(g is None for g in ag):
        raise ValueError("Gradient tuple missing for a year in the first generation.")

    g0   = np.array([g[0] for g in ag])
    s_ab = np.array([g[1] for g in ag])
    l_ab = np.array([g[3] for g in ag])
    ecap = np.array([g[4] for g in ag])

    magnitude = ecap * s_ab * l_ab
    alignment = g0 / magnitude

    nrm = lambda x: x / np.mean(x)
    out = {
        "years":   np.arange(1, n_gen + 1),
        "g0_n":    nrm(g0),
        "mag_n":   nrm(magnitude),
        "align_n": nrm(alignment),
        "r_align": float(np.corrcoef(nrm(g0), nrm(alignment))[0, 1]),
        "r_mag":   float(np.corrcoef(nrm(g0), nrm(magnitude))[0, 1]),
        "mean_g0": float(np.mean(g0)),
        "swing":   float(g0.max() / g0.min()),
    }
    out["above_mean"] = out["align_n"] >= 1.0     # alignment regime
    return out


def _finish(ax, n_gen: int, ymax: float) -> None:
    ax.set_xlabel("Project year")
    ax.set_ylabel("Value / annual mean")
    ax.set_xticks(np.arange(1, n_gen + 1))
    ax.set_xlim(0.4, n_gen + 0.6)
    ax.set_ylim(0.0, ymax)


def _save(fig, out_dir: Path, stem: str) -> None:
    """Write the formats OUTPUT asks for."""
    if OUTPUT in ("pdf", "both"):
        fig.savefig(out_dir / f"{stem}.pdf")
    if OUTPUT in ("png", "both"):
        fig.savefig(out_dir / f"{stem}.png", dpi=DPI)
    plt.close(fig)
    print(f"  wrote {stem} ({OUTPUT})")


# --------------------------------------------------------------------------- #
# FIG 4.4  bars
# --------------------------------------------------------------------------- #
def plot_gradient_bars(dec: dict, out_dir: Path) -> None:
    y, ng = dec["years"], dec["g0_n"]
    fig, ax = plt.subplots(figsize=figsize(0.95, 0.56))

    ax.bar(y, ng, width=BARS_BAR_WIDTH, color=C_BAR,
           edgecolor="none", zorder=1)
    ax.axhline(1.0, color=C_REF, linestyle=":", linewidth=0.8, zorder=2)

    _finish(ax, len(y), ng.max() * 1.15)
    handles = [Line2D([0], [0], color=C_REF, linestyle=":", label="Annual mean")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7)
    _save(fig, out_dir, f"fig44_gradient_bars_{PRICE_TAG}")


# --------------------------------------------------------------------------- #
# FIG 4.5  lines (clean)
# --------------------------------------------------------------------------- #
def plot_gradient_lines(dec: dict, out_dir: Path) -> None:
    y = dec["years"]
    fig, ax = plt.subplots(figsize=figsize(0.95, 0.56))

    ax.axhline(1.0, color=C_REF, linestyle=":", linewidth=0.8, zorder=1)
    ax.plot(y, dec["mag_n"], marker="s", linestyle="--", color=C_MAG,
            linewidth=1.3, markersize=4.0, zorder=2,
            label=f"Magnitude term  (r = {dec['r_mag']:+.2f})")
    ax.plot(y, dec["align_n"], marker="o", linestyle="-", color=C_ALIGN,
            linewidth=1.7, markersize=4.5, zorder=3,
            label=f"Alignment term  (r = {dec['r_align']:+.2f})")

    _finish(ax, len(y), max(dec["align_n"].max(), dec["mag_n"].max()) * 1.15)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    _save(fig, out_dir, f"fig45_gradient_lines_{PRICE_TAG}")


# --------------------------------------------------------------------------- #
# FIG 4.6  overlay = Fig 4.4 bars (low opacity) + Fig 4.5 lines
# --------------------------------------------------------------------------- #
def plot_gradient_overlay(dec: dict, out_dir: Path) -> None:
    y, ng = dec["years"], dec["g0_n"]
    fig, ax = plt.subplots(figsize=figsize(0.95, 0.56))

    # Fig 4.4's bars, faded, as background
    ax.bar(y, ng, width=OVERLAY_BAR_WIDTH, color=C_BAR,
           alpha=OVERLAY_BAR_ALPHA, edgecolor="none", zorder=1)
    ax.axhline(1.0, color=C_REF, linestyle=":", linewidth=0.8, zorder=2)

    # Fig 4.5's lines on top
    ax.plot(y, dec["mag_n"], marker="s", linestyle="--", color=C_MAG,
            linewidth=1.3, markersize=4.0, zorder=3,
            label=f"Magnitude term  (r = {dec['r_mag']:+.2f})")
    ax.plot(y, dec["align_n"], marker="o", linestyle="-", color=C_ALIGN,
            linewidth=1.7, markersize=4.5, zorder=4,
            label=f"Alignment term  (r = {dec['r_align']:+.2f})")

    _finish(ax, len(y), max(ng.max(), dec["align_n"].max()) * 1.15)
    handles = [
        Line2D([0], [0], color=C_ALIGN, marker="o", linestyle="-",
               linewidth=1.7, markersize=4.5, label=f"Alignment term  (r = {dec['r_align']:+.2f})"),
        Line2D([0], [0], color=C_MAG, marker="s", linestyle="--",
               linewidth=1.3, markersize=4.0, label=f"Magnitude term  (r = {dec['r_mag']:+.2f})"),
        Patch(facecolor=C_BAR, alpha=OVERLAY_BAR_ALPHA, edgecolor="none",
              label=r"$g_k$"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7,
              handlelength=1.8)
    _save(fig, out_dir, f"fig46_gradient_overlay_{PRICE_TAG}")


# --------------------------------------------------------------------------- #
# What Appendix C.3 reports for the run the thesis figures were drawn from.
# Checked rather than assumed, because this script was repointed to a different
# baseline run during the migration.
# Measured on this run. The superseded half-cycle attribution map gave +0.97 and
# +0.17 on the same configuration (run 20260709_131650); the exact spanning-set
# construction of Section 2.8.2 gives the values below. See Appendix C.3.
EXPECTED = {"r_align": 0.38, "r_mag": 0.92}


def main() -> None:
    if OUTPUT not in ("png", "pdf", "both"):
        raise ValueError(f'OUTPUT must be "png", "pdf" or "both", not {OUTPUT!r}')

    print(f"  source: {NPY_FILE.name}")
    dec = load_decomposition(NPY_FILE, N_GEN)
    print(f"  mean g_k              : {dec['mean_g0']:.4e}")
    print(f"  g_k swing (max/min)   : {dec['swing']:.2f}x")
    print(f"  corr(g_k, alignment)  : {dec['r_align']:+.3f}   (thesis {EXPECTED['r_align']:+.2f})")
    print(f"  corr(g_k, magnitude)  : {dec['r_mag']:+.3f}   (thesis {EXPECTED['r_mag']:+.2f})")
    for k, want in EXPECTED.items():
        if abs(dec[k] - want) > 0.05:
            print(f"  NOTE: {k} differs from the figure in the thesis by "
                  f"{abs(dec[k] - want):.2f}; the caption and the surrounding "
                  f"text in Appendix C.3 need rechecking.")

    out_dir = SCRIPT_DIR

    plot_gradient_bars(dec, out_dir)
    plot_gradient_lines(dec, out_dir)
    plot_gradient_overlay(dec, out_dir)


if __name__ == "__main__":
    main()