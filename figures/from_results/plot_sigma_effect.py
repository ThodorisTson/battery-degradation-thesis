r"""
plot_sigma_effect.py
====================
Produces Figure 2.3 only: fig23_sigma_effect.

Two rainflow cycles from the same annual dispatch with identical cycle
amplitude delta = 0.30 but different mean state of charge sigma, so the mean
SoC stress factor S_sigma is the only term that differs between them.

    Cycle A   window 0.10 to 0.40, sigma = 0.25
    Cycle B   window 0.60 to 0.90, sigma = 0.75

INPUT
-----
results/window_sweep/storage_e_fixed.npy, the stored energy trace in MWh,
divided by E_CAP_MWH to give state of charge.

The fork fell back to a synthetic trace when the file was absent, printed a
warning, and wrote the figure anyway. A thesis figure script that can emit
fabricated data on a missing input is a liability, so the fallback is removed
and a missing file is now an error. Layout can be checked by pointing SOC_NPY
at any array of the right length.

VERIFICATION
------------
The window bounds, the amplitude and the mean SoC quoted in the caption are
measured from the highlighted segments at runtime and compared against the
declared values. A mismatch beyond CHECK_TOL is reported. The caption claims an
exact rainflow match, so the numbers should come from the data rather than from
the panel configuration.

Output is written to figures/from_results/.

CAPTION
-------
Two cycles from the 2022 IEA Wind Task 50 dispatch, both with cycle amplitude
delta = 0.30 identified by rainflow counting. Cycle A operates in the 0.10 to
0.40 SoC window, sigma = 0.25; Cycle B operates in the 0.60 to 0.90 window,
sigma = 0.75. S_delta is identical for both, so the mean SoC stress S_sigma is
the sole driver of the difference in degradation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from degradation.style import (apply_thesis_style, figsize, TUDELFT,
                               TEXTWIDTH_IN, FS_BASE, FS_ANNOT)
from degradation.paths import RESULTS_DIR, require

PALETTE = apply_thesis_style(palette="brand", usetex=False)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
OUTPUT = "png"         # "png", "pdf" or "both"
DPI    = 300

SOC_NPY = RESULTS_DIR / "window_sweep" / "storage_e_fixed.npy"
E_CAP_MWH = 300.0      # divisor turning stored energy into state of charge

OUT_DIR  = RESULTS_DIR.parent / "figures" / "from_results"
OUT_STEM = "fig23_sigma_effect"

# Fraction of \textwidth this figure is included at. It must match the width=
# argument of \includegraphics, otherwise LaTeX rescales the figure and every
# font size on the page changes with it.
INCLUDE_WIDTH_FRAC = 1.0
ASPECT = 0.52

DELTA = 0.30           # cycle amplitude shared by both panels
CHECK_TOL = 0.01       # tolerance on the measured window bounds and sigma

PANELS = [
    dict(win_s=2200, win_e=2272, hi_s=2228, hi_e=2254,
         cyc_lo=0.10, cyc_hi=0.40, sigma=0.25,
         col=TUDELFT["blue"], label="Cycle A  \u2192  Low \u03c3",
         day_start=91),
    dict(win_s=288, win_e=360, hi_s=311, hi_e=356,
         cyc_lo=0.60, cyc_hi=0.90, sigma=0.75,
         col=TUDELFT["darkred"], label="Cycle B  \u2192  High \u03c3",
         day_start=12),
]


# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_soc(path: Path) -> np.ndarray:
    """Load the stored-energy trace and convert it to state of charge."""
    soc = np.load(require(path)) / E_CAP_MWH
    need = max(p["win_e"] for p in PANELS)
    if soc.size < need:
        raise ValueError(
            f"{path.name} has {soc.size} samples; the panels need at least "
            f"{need}. Check that E_CAP_MWH and the panel windows match this "
            f"run.")
    if soc.min() < -1e-9 or soc.max() > 1 + 1e-9:
        raise ValueError(
            f"State of charge from {path.name} spans {soc.min():.3f} to "
            f"{soc.max():.3f}, which is outside [0, 1]. E_CAP_MWH = "
            f"{E_CAP_MWH:.0f} MWh is probably not the capacity of this run.")
    print(f"  loaded {path.relative_to(RESULTS_DIR.parent)}  "
          f"({soc.size} samples, SoC {soc.min():.3f} to {soc.max():.3f})")
    return soc


def check_panel(soc: np.ndarray, p: dict, name: str) -> None:
    """Measure the highlighted cycle and compare against the declared values."""
    seg = soc[p["hi_s"]:p["hi_e"] + 1]
    lo, hi = float(seg.min()), float(seg.max())
    amp = hi - lo
    mid = 0.5 * (lo + hi)

    print(f"  {name}: measured window {lo:.3f} to {hi:.3f}, "
          f"amplitude {amp:.3f}, mean SoC {mid:.3f}")

    issues = []
    if abs(lo - p["cyc_lo"]) > CHECK_TOL:
        issues.append(f"lower bound {lo:.3f} against declared {p['cyc_lo']:.2f}")
    if abs(hi - p["cyc_hi"]) > CHECK_TOL:
        issues.append(f"upper bound {hi:.3f} against declared {p['cyc_hi']:.2f}")
    if abs(amp - DELTA) > CHECK_TOL:
        issues.append(f"amplitude {amp:.3f} against declared {DELTA:.2f}")
    if abs(mid - p["sigma"]) > CHECK_TOL:
        issues.append(f"mean SoC {mid:.3f} against declared {p['sigma']:.2f}")
    if issues:
        print(f"  [warning] {name} does not match the caption: "
              + "; ".join(issues))


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════

def plot_panel(fig, ax, soc, p, show_ylabel=True):
    seg = soc[p["win_s"]:p["win_e"]]
    n = len(seg)
    t = np.arange(n)
    col = p["col"]
    lo, hi = p["cyc_lo"], p["cyc_hi"]

    hl_s = p["hi_s"] - p["win_s"]
    hl_e = p["hi_e"] - p["win_s"]

    # Deliberate exception to the shared style: the light panel background
    # separates the context trace from the highlighted cycle.
    ax.set_facecolor(PALETTE["bg"])

    for y, lbl in [(0.10, r"SoC$_\mathrm{min}$"), (0.90, r"SoC$_\mathrm{max}$")]:
        ax.axhline(y, color="#c0c0c0", lw=0.7, ls=":", alpha=0.8, zorder=1)
        ax.text(n - 0.5, y, lbl, va="center", ha="right",
                fontsize=FS_ANNOT, color="#b8b8b8", zorder=6)

    ax.axvspan(hl_s, hl_e, color=col, alpha=0.07, zorder=1, lw=0)

    ax.plot(t[:hl_s + 1], seg[:hl_s + 1], color=PALETTE["neutral"], lw=0.9,
            alpha=0.55, zorder=2, solid_capstyle="round")
    ax.plot(t[hl_e - 1:], seg[hl_e - 1:], color=PALETTE["neutral"], lw=0.9,
            alpha=0.55, zorder=2, solid_capstyle="round")

    t_hl, s_hl = t[hl_s:hl_e + 1], seg[hl_s:hl_e + 1]
    ax.fill_between(t_hl, lo, s_hl, where=s_hl >= lo,
                    color=col, alpha=0.15, zorder=3)
    ax.plot(t_hl, s_hl, color=col, lw=1.5, zorder=4, solid_capstyle="round")

    for y in (lo, hi):
        ax.plot([hl_s, hl_e], [y, y], color=col, lw=0.8, ls="--",
                alpha=0.50, zorder=3)

    # delta bracket, with serif half-width scaled to the window so it holds at
    # any figure size
    x_brk = hl_s + 0.5 * (hl_e - hl_s)
    sw = n * 0.025
    ax.annotate("", xy=(x_brk, hi), xytext=(x_brk, lo),
                arrowprops=dict(arrowstyle="<->", color=col, lw=1.4,
                                mutation_scale=8, shrinkA=0, shrinkB=0))
    for y_s in (lo, hi):
        ax.plot([x_brk - sw, x_brk + sw], [y_s, y_s], color=col, lw=1.1)

    # delta and sigma share a right edge so they read as one annotation column
    ax.text(x_brk - n * 0.06, lo + (hi - lo) * 0.15,
            f"$\\delta$ = {DELTA:.2f}", va="bottom", ha="right",
            fontsize=FS_BASE, color=col, fontweight="bold")

    sigma = p["sigma"]
    ax.plot([hl_s, hl_e], [sigma, sigma], color=col, lw=0.9,
            ls=(0, (5, 3)), alpha=0.80, zorder=5)
    ax.text(x_brk - n * 0.06, sigma + 0.030,
            f"$\\sigma$ = {sigma:.2f}", va="bottom", ha="right",
            fontsize=FS_BASE, color=col, style="italic", zorder=6)

    ax.text(0.03, 0.96, p["label"], transform=ax.transAxes,
            va="top", ha="left", fontsize=FS_BASE, fontweight="bold",
            color=col)
    # Sits at the foot of the highlight band. At the fork's 14 in canvas this
    # was at the top, where it now collides with the panel label.
    ax.text((hl_s + hl_e) / 2, 0.02, "\u2190 target cycle \u2192",
            va="bottom", ha="center", fontsize=FS_ANNOT, color=col,
            style="italic", alpha=0.7, zorder=7)

    ax.set_xlim(0, n - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel("Hour  (\u2013)")

    day0 = p["day_start"]
    xticks = list(range(0, n, 24))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"Day {day0 + i}\n00:00" for i in range(len(xticks))],
                       fontsize=FS_ANNOT)

    if show_ylabel:
        ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1.1, 0.1)],
                           fontsize=FS_ANNOT)
        ax.set_ylabel("State of charge  (\u2013)")
        fig.canvas.draw()   # tick label objects must exist before recolouring
        for tick, val in zip(ax.yaxis.get_ticklabels(), np.arange(0, 1.01, 0.1)):
            if abs(round(val, 1) - lo) < 0.01 or abs(round(val, 1) - hi) < 0.01:
                tick.set_color(col)
                tick.set_fontweight("bold")
    else:
        ax.set_yticklabels([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
        for y in (lo, hi):
            ax.text(-0.015, y, f"{y:.2f}", va="center", ha="right",
                    fontsize=FS_ANNOT, color=col, fontweight="bold",
                    transform=ax.get_yaxis_transform(), clip_on=False)


def make_figure(soc: np.ndarray) -> plt.Figure:
    # constrained_layout is off so the two panels can be held close together;
    # the margins below replace it.
    fig, axes = plt.subplots(1, 2,
                             figsize=figsize(INCLUDE_WIDTH_FRAC, aspect=ASPECT),
                             constrained_layout=False)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.97,
                        bottom=0.22, wspace=0.06)
    plot_panel(fig, axes[0], soc, PANELS[0], show_ylabel=True)
    plot_panel(fig, axes[1], soc, PANELS[1], show_ylabel=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    if OUTPUT not in ("png", "pdf", "both"):
        raise ValueError(f"OUTPUT must be 'png', 'pdf' or 'both', got {OUTPUT!r}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading the dispatch trace")
    soc = load_soc(SOC_NPY)

    print("\nChecking the two cycles against the caption")
    for p, name in zip(PANELS, ("Cycle A", "Cycle B")):
        check_panel(soc, p, name)

    print(f"\nDrawn at {INCLUDE_WIDTH_FRAC:.2f} x textwidth = "
          f"{INCLUDE_WIDTH_FRAC * TEXTWIDTH_IN:.2f} in. Include at "
          f"width={INCLUDE_WIDTH_FRAC:.2f}\\textwidth.")

    print(f"\nWriting figure to {OUT_DIR}")
    fig = make_figure(soc)
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
    print("\nDone.")


if __name__ == "__main__":
    main()
