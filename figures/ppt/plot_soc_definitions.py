"""
Definition figure: operating window, cycle amplitude and mean state of charge.
Emitted as a click build so the slide can be revealed one element at a time.

Notation follows Chapter 2 of the thesis:
  sigma_min, sigma_max  operating window bounds
  sigma_max - sigma_min window width; bounds every cycle amplitude
  window center         (sigma_min + sigma_max)/2, the design choice swept in Chapter 4. NOT sigma-bar, which Chapter 2 reserves for
                        the mean SoC over the whole simulation period.
  delta_i               cycle amplitude of rainflow half-cycle i
  sigma_i               mean SoC of cycle i, (SoC_max,i + SoC_min,i)/2

Outputs, in build order:
  1_trace       state of charge only
  2_bounds      + sigma_min and sigma_max
  3_window      + window width and window center
  4_deep        + the deep cycle, window kept on screen
  5_shallow     + the shallow cycle, window kept on screen
  6_all         everything

Every frame uses the same figure size, axis limits and layout, and none use bbox_inches='tight', so the frames register exactly when overlaid in
PowerPoint. No title is drawn; put it in a text box on the slide.

Requires soc_trace.py in the same directory. That module holds the shared trajectory and is also imported by the rainflow explainer, so the two slides
cannot drift apart.

Run from VS Code on Windows. Outputs land next to this file."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

OUTPUT = "png"        # "png", "pdf" or "both"
DPI    = 300

NAVY = "#0C2340"
DARK_RED = "#A50034"
FILL_BLUE = "#0076C2"
GREY = "#6B7A88"

from soc_trace import (SIG_MIN, SIG_MAX, CENTER, WIDTH, trace, DEEP_CYCLE, SHALLOW_CYCLE)

FIGSIZE = (10.0, 5.0)
XLIM = (0, 62)
YLIM = (0, 1.02)

def _cyc(cycle, sym, gap, d_y, s_y):
    t0, t1, s0, s1, delta, sigma = cycle
    return dict(t0=t0, t1=t1, s0=s0, s1=s1, sym=sym,
                dim_x=max(t0, t1) + gap, lab_x=max(t0, t1) + gap + 0.8,
                d_y=d_y, s_y=s_y)


# Both cycles are verified in soc_trace.py as full cycles the counter returns.
DEEP = _cyc(DEEP_CYCLE, "i", 1.6, 0.80, 0.53)
SHALLOW = _cyc(SHALLOW_CYCLE, "j", 1.6, 0.60, 0.39)

HALO = [pe.withStroke(linewidth=3.4, foreground="white")]
THALO = [pe.withStroke(linewidth=3.0, foreground="white")]


def _cycle(ax, c):
    delta = abs(c["s0"] - c["s1"])
    sigma = 0.5 * (c["s0"] + c["s1"])
    lo, hi = min(c["s0"], c["s1"]), max(c["s0"], c["s1"])

    ax.plot([c["t0"], c["t1"]], [c["s0"], c["s1"]], color=DARK_RED, lw=3.0,
            zorder=6, solid_capstyle="round", path_effects=HALO)

    for yy in (lo, hi):
        ax.plot([min(c["t0"], c["t1"]), c["dim_x"] + 0.4], [yy, yy],
                color=DARK_RED, lw=0.9, ls=(0, (2, 2)), zorder=5,
                path_effects=HALO)
    ax.annotate("", xy=(c["dim_x"], hi), xytext=(c["dim_x"], lo),
                arrowprops=dict(arrowstyle="<->", color=DARK_RED, lw=2.0,
                                path_effects=HALO), zorder=7)
    ax.text(c["lab_x"], c["d_y"], rf"$\delta_{{{c['sym']}}}$ = {delta:.2f}",
            ha="left", va="center", fontsize=11.5, color=DARK_RED, zorder=8,
            path_effects=THALO)

    ax.plot([c["t0"] - 1.4, c["dim_x"] + 0.4], [sigma, sigma],
            color=FILL_BLUE, lw=1.7, ls="-.", zorder=6, path_effects=HALO)
    ax.text(c["lab_x"], c["s_y"], rf"$\sigma_{{{c['sym']}}}$ = {sigma:.2f}",
            ha="left", va="center", fontsize=11.5, color=FILL_BLUE, zorder=8,
            path_effects=THALO)


def render(show, fname):
    """show may contain: 'bounds', 'window', 'deep', 'shallow'."""
    t, soc = trace()

    plt.rcParams.update({"font.size": 10.5, "text.color": NAVY,
                         "axes.labelcolor": NAVY, "xtick.color": NAVY,
                         "ytick.color": NAVY, "axes.edgecolor": NAVY})

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    if "bounds" in show:
        ax.axhspan(SIG_MIN, SIG_MAX, color=FILL_BLUE, alpha=0.055, zorder=0)
        for b, lab, va, off in (
                (SIG_MAX, rf"$\sigma_{{\max}}$ = {SIG_MAX:.2f}", "bottom", 0.022),
                (SIG_MIN, rf"$\sigma_{{\min}}$ = {SIG_MIN:.2f}", "top", -0.022)):
            ax.axhline(b, color=GREY, ls="--", lw=1.1, zorder=1)
            ax.text(44.0, b + off, lab, va=va, ha="right", fontsize=11,
                    color=GREY, zorder=8, path_effects=THALO)

    ax.plot(t, soc, color=NAVY, lw=1.9, zorder=3)

    if "window" in show:
        ax.axhline(CENTER, color=NAVY, ls=":", lw=1.4, zorder=2)
        ax.annotate("", xy=(53.0, SIG_MAX), xytext=(53.0, SIG_MIN),
                    arrowprops=dict(arrowstyle="<->", color=FILL_BLUE, lw=2.2),
                    zorder=7)
        ax.text(54.2, CENTER + 0.20,
                "window width\n" + r"$\sigma_{\max}-\sigma_{\min}$"
                + "\n" + rf"$\delta_{{\max}}$ = {WIDTH:.2f}",
                ha="left", va="center", fontsize=11.5, color=FILL_BLUE,
                linespacing=1.5, zorder=8)
        ax.text(54.2, CENTER - 0.055,
                "window center\n"
                + r"$(\sigma_{\max}+\sigma_{\min})/2$"
                + f"\n= {CENTER:.2f}",
                va="top", ha="left", fontsize=11.5, color=NAVY,
                linespacing=1.5, zorder=8)

    if "deep" in show:
        _cycle(ax, DEEP)
    if "shallow" in show:
        _cycle(ax, SHALLOW)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel("Time")
    ax.set_ylabel("State of charge  (\u2013)")
    ax.set_xticks([])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    OUT = Path(__file__).parent
    # No bbox_inches='tight': the frames must register exactly when overlaid.
    if OUTPUT in ("pdf", "both"):
        fig.savefig(OUT / f"{fname}.pdf", dpi=DPI, facecolor="white")
    if OUTPUT in ("png", "both"):
        fig.savefig(OUT / f"{fname}.png", dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  {fname}  ({OUTPUT})")


# The window stays on screen while each cycle is shown. An element that
# disappears reads as a constraint that stopped applying, and keeping it drawn
# makes delta_i <= sigma_max - sigma_min visible rather than asserted.
BUILD = [
    ((), "soc_def_1_trace"),
    (("bounds",), "soc_def_2_bounds"),
    (("bounds", "window"), "soc_def_3_window"),
    (("bounds", "window", "deep"), "soc_def_4_deep"),
    (("bounds", "window", "shallow"), "soc_def_5_shallow"),
    (("bounds", "window", "deep", "shallow"), "soc_def_6_all"),
]

if __name__ == "__main__":
    if OUTPUT not in ("png", "pdf", "both"):
        raise ValueError(f"OUTPUT must be 'png', 'pdf' or 'both', got {OUTPUT!r}")
    print(f"Writing {len(BUILD)} build frames to {Path(__file__).parent}")
    for show, name in BUILD:
        render(show, name)