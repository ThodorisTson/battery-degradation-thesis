r"""
plot_rainflow_explainer_v4.py

Rainflow counting shown on two unrelated irregular signals: a structural fatigue load spectrum and a battery state-of-charge trajectory. The point of
the figure is that the counting procedure is the same in both domains.

Conventions:

1. Amplitude is the full range, matching the Chapter 2 definition delta_i = |SoC_max,i - SoC_min,i|.
2. Mean level is named sigma_i, and the window bounds are sigma_min and sigma_max, again matching Chapter 2.
3. Panel titles are dropped per the thesis convention; the column headers identify the two domains.

Requires soc_trace.py in the same directory. The battery panel uses the same trajectory as plot_soc_definitions.py, so the two slides show one trace and the
audience does not have to re-orient. The assertion block below checks that the counter still returns the two cycles that slide annotates.

Toggles:
    SLIDE       True  -> wide format and larger type for the presentation
                False -> one text width, thesis sizing
    BOTTOM_ROW  False -> top row only, which is the stronger slide

Run from VS Code on Windows. Outputs land next to this file.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from soc_trace import (SIG_MAX, SIG_MIN, trace,
                       DEEP_CYCLE, SHALLOW_CYCLE)
from degradation.style import apply_thesis_style, TUDELFT, figsize

OUTPUT = "png"        # "png", "pdf" or "both"
DPI    = 300

SLIDE = True
BOTTOM_ROW = True
# Multiplicity (full vs half cycles) is deliberately NOT distinguished. The rest of the presentation never uses it, and introducing it here would need a
# minute of explanation for a concept that does not return. The counter still tracks it; the figure just does not draw the distinction.
SHOW_MULTIPLICITY = False
# The symbols delta_i, sigma_i, sigma_min and sigma_max are introduced on the NEXT slide. This figure comes first, so it stays notation-free: the window
# band, its bounds and the amplitude bound are all suppressed, and the axes are named in words.
NOTATION = False


# --------------------------------------------------------------------------- #
# ASTM E1049 rainflow counting
# --------------------------------------------------------------------------- #
def _extract_peaks(x):
    pts = [x[0]]
    for i in range(1, len(x) - 1):
        if (x[i] > x[i - 1] and x[i] >= x[i + 1]) or \
           (x[i] < x[i - 1] and x[i] <= x[i + 1]):
            pts.append(x[i])
    pts.append(x[-1])
    return np.array(pts)


def rainflow_count(signal):
    """Return (amplitude, mean, multiplicity) per cycle.

    Amplitude is the full range |max - min| of the cycle, matching the
    Chapter 2 definition of delta_i. Multiplicity is 1.0 for a full cycle and
    0.5 for a half cycle.
    """
    pts = _extract_peaks(signal)
    stack, cycles = [], []
    for pt in pts:
        stack.append(pt)
        while len(stack) >= 4:
            X = abs(stack[-1] - stack[-2])
            Y = abs(stack[-2] - stack[-3])
            if X >= Y:
                cycles.append((Y, (stack[-2] + stack[-3]) / 2.0, 1.0))
                stack.pop(-2)
                stack.pop(-2)
            else:
                break
    for i in range(len(stack) - 1):
        cycles.append((abs(stack[i + 1] - stack[i]),
                       (stack[i] + stack[i + 1]) / 2.0, 0.5))
    return cycles


# --------------------------------------------------------------------------- #
# Synthetic signals
# --------------------------------------------------------------------------- #
t = np.linspace(0, 1, 800)

rng = np.random.default_rng(13)
n = 30
st = np.sort(np.concatenate(([0.0], rng.uniform(0, 1, n - 2), [1.0])))
sv = np.empty(n)
sv[0::2] = rng.uniform(0.30, 1.00, len(sv[0::2]))
sv[1::2] = rng.uniform(-1.00, -0.20, len(sv[1::2]))
sv[0], sv[-1] = 0.10, 0.05
stress = np.interp(t, st, sv)

# The battery panel uses the same trajectory as the definitions slide, so the
# two slides show one trace and the audience does not have to re-orient.
t_soc, soc = trace(len(t), normalized_time=True)

cyc_s = rainflow_count(stress)
cyc_b = rainflow_count(soc)


def _split(cycles):
    full = [(m, a) for a, m, w in cycles if w == 1.0]
    half = [(m, a) for a, m, w in cycles if w == 0.5]
    return full, half


full_s, half_s = _split(cyc_s)
full_b, half_b = _split(cyc_b)

# The definitions slide annotates two cycles on this same trace. Confirm the
# counter really returns them, so the two slides cannot contradict each other.
for _name, _c in (("deep", DEEP_CYCLE), ("shallow", SHALLOW_CYCLE)):
    _d, _s = _c[4], _c[5]
    _hit = [x for x in cyc_b if abs(x[0] - _d) < 1e-6 and abs(x[1] - _s) < 1e-6]
    assert _hit, f"annotated {_name} cycle is not returned by the counter"
    print(f"check: annotated {_name} cycle delta={_d:.2f} sigma={_s:.2f} "
          f"found, n_i={_hit[0][2]}")

print(f"stress: {len(full_s)} full + {len(half_s)} half"
      f"  |  soc: {len(full_b)} full + {len(half_b)} half")
print(f"max soc amplitude {max(a for a, _, _ in cyc_b):.3f}"
      f"  (window width {SIG_MAX - SIG_MIN:.2f})")

# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
P = apply_thesis_style(palette="brand", usetex=False)
STRESS_COL = TUDELFT["darkred"]
SOC_COL = TUDELFT["blue"]
NEUTRAL = P["neutral"]

if SLIDE:
    FS_T, FS_L, FS_A = 12, 11, 10
    SIZE = (13.0, 7.0) if BOTTOM_ROW else (13.0, 3.9)
    MS = 46
else:
    FS_T, FS_L, FS_A = 9, 9, 7
    SIZE = figsize(1.0, 0.72 if BOTTOM_ROW else 0.40)
    MS = 16

nrows = 2 if BOTTOM_ROW else 1
fig, axes = plt.subplots(nrows, 2, figsize=SIZE, constrained_layout=True)
axes = np.atleast_2d(axes)
ax_s, ax_b = axes[0, 0], axes[0, 1]


def _clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=FS_A)


# --- top row, the signals --------------------------------------------------- #
_clean(ax_s)
ax_s.plot(t, stress, color=STRESS_COL, lw=1.1, zorder=2)
ax_s.axhline(0, color=NEUTRAL, lw=0.6, ls="--", alpha=0.5, zorder=1)
ax_s.set_xlim(0, 1)
ax_s.set_ylim(-1.35, 1.35)
ax_s.set_ylabel("Normalized stress  (\u2013)", fontsize=FS_L)
ax_s.set_xlabel("Time  (\u2013)", fontsize=FS_L)

_clean(ax_b)
if NOTATION:
    ax_b.axhspan(SIG_MIN, SIG_MAX, color=SOC_COL, alpha=0.06, zorder=0)
    for b, lab in ((SIG_MAX, r"$\sigma_{\max}$"), (SIG_MIN, r"$\sigma_{\min}$")):
        ax_b.axhline(b, color=NEUTRAL, lw=0.7, ls="--", alpha=0.6, zorder=1)
        ax_b.text(1.012, b, lab, va="center", fontsize=FS_A, color=NEUTRAL,
                  transform=ax_b.get_yaxis_transform())
ax_b.plot(t_soc, soc, color=SOC_COL, lw=1.2, zorder=2)
ax_b.set_xlim(0, 1)
ax_b.set_ylim(0, 1.05)
ax_b.set_ylabel("State of charge  (\u2013)", fontsize=FS_L)
ax_b.set_xlabel("Time  (\u2013)", fontsize=FS_L)

# --- bottom row, the extracted cycles --------------------------------------- #
if BOTTOM_ROW:
    ax_sc, ax_bc = axes[1, 0], axes[1, 1]

    def scatter_panel(ax, full, half, col, xlab, ylab, note=None,
                      legend_loc="upper right"):
        _clean(ax)
        if SHOW_MULTIPLICITY:
            if full:
                ax.scatter([m for m, _ in full], [a for _, a in full], s=MS,
                           facecolors=col, edgecolors=col, linewidths=0.8,
                           zorder=3, label="Full cycle  $n_i = 1$")
            if half:
                ax.scatter([m for m, _ in half], [a for _, a in half], s=MS,
                           facecolors="white", edgecolors=col, linewidths=1.1,
                           zorder=4, label="Half cycle  $n_i = 1/2$")
            ax.legend(loc=legend_loc, fontsize=FS_A, frameon=True,
                      framealpha=0.95, edgecolor="#DDE3EA", handletextpad=0.5)
        else:
            pts = full + half
            ax.scatter([m for m, _ in pts], [a for _, a in pts], s=MS,
                       facecolors=col, edgecolors=col, linewidths=0.8,
                       alpha=0.85, zorder=3)
        ax.set_xlabel(xlab, fontsize=FS_L)
        ax.set_ylabel(ylab, fontsize=FS_L)
        if note:
            ax.text(0.985, 0.965, note, transform=ax.transAxes, ha="right",
                    va="top", fontsize=FS_A, color=NEUTRAL)

    def _count(full, half):
        n = len(full) + len(half)
        return (f"{len(full)} full, {len(half)} half" if SHOW_MULTIPLICITY
                else f"{n} cycles extracted")

    amp_lab = ("Cycle amplitude  $\\delta_i$  (–)" if NOTATION
               else "Cycle amplitude  (–)")
    soc_mean_lab = (r"Cycle mean SoC  $\sigma_i$  (–)" if NOTATION
                    else "Cycle mean SoC  (–)")

    scatter_panel(ax_sc, full_s, half_s, STRESS_COL,
                  "Cycle mean level  (–)", amp_lab,
                  _count(full_s, half_s))

    scatter_panel(ax_bc, full_b, half_b, SOC_COL,
                  soc_mean_lab, amp_lab,
                  _count(full_b, half_b),
                  legend_loc="lower right")

    if NOTATION:
        ax_bc.axhline(SIG_MAX - SIG_MIN, color=NEUTRAL, ls=":", lw=1.2,
                      zorder=2)
        ax_bc.set_ylim(0, (SIG_MAX - SIG_MIN) * 1.16)
        ax_bc.text(0.015, SIG_MAX - SIG_MIN + 0.012,
                   r"$\delta_i \leq \sigma_{\max}-\sigma_{\min} = 0.80$",
                   transform=ax_bc.get_yaxis_transform(), ha="left",
                   va="bottom", fontsize=FS_A, color=NEUTRAL, zorder=6)
    else:
        ax_bc.set_ylim(0, (SIG_MAX - SIG_MIN) * 1.16)

# --- column headers --------------------------------------------------------- #
ax_s.set_title("STRUCTURAL FATIGUE", fontsize=FS_T, fontweight="bold",
               color=STRESS_COL, pad=8)
ax_b.set_title("BATTERY DEGRADATION", fontsize=FS_T, fontweight="bold",
               color=SOC_COL, pad=8)

if OUTPUT not in ("png", "pdf", "both"):
    raise ValueError(f"OUTPUT must be 'png', 'pdf' or 'both', got {OUTPUT!r}")

OUT = Path(__file__).parent
tag = "full" if BOTTOM_ROW else "toprow"
if OUTPUT in ("pdf", "both"):
    fig.savefig(OUT / f"rainflow_explainer_{tag}.pdf", facecolor="white")
if OUTPUT in ("png", "both"):
    fig.savefig(OUT / f"rainflow_explainer_{tag}.png", dpi=DPI, facecolor="white")
print(f"Saved rainflow_explainer_{tag}  ({OUTPUT})")