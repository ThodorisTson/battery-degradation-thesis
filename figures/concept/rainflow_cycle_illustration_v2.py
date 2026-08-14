r"""
rainflow_cycle_illustration_v2.py   (thesis label fig:rainflow_cycles)

Changes vs v1:
  1. Removed the marker at (10, 0.25). The trajectory falls monotonically from 0.70 at t=6 through 0.25 at t=10 to 0.15 at t=12, so that point is a slope
     change, not a local extremum, and is not a turning point.
  2. Dashed discharge bracket extended from t=6->10 to t=6->12, so it spans a real half-cycle (0.70 -> 0.15, delta = 0.55) instead of a partial descent.
  3. Dotted guide lines added from the rising limb to the delta/sigma arrow, so the annotation is visually attached to the half-cycle it describes
     (0.15 -> 0.85, delta = 0.70, sigma = 0.50).
  4. Duplicate numpy / pyplot imports removed.
  5. Output format is selected by OUTPUT below.

Runs in VS Code on Windows: matplotlib + numpy only, bundled font, no LaTeX.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from degradation.style import (apply_thesis_style, figsize, TUDELFT,
                               FS_BASE, FS_LABEL, FS_ANNOT)

# -- Output ------------------------------------------------------------------ #
OUTPUT = "png"     # "png", "pdf" or "both"
DPI = 300


def save(fig, out_dir, stem):
    """Write the formats OUTPUT asks for, beside the calling script."""
    if OUTPUT in ("pdf", "both"):
        fig.savefig(out_dir / f"{stem}.pdf")
        print(f"  wrote {stem}.pdf")
    if OUTPUT in ("png", "both"):
        fig.savefig(out_dir / f"{stem}.png", dpi=DPI)
        print(f"  wrote {stem}.png  ({DPI} dpi)")


P = apply_thesis_style(palette="brand", usetex=False)

# -- Synthetic SoC trajectory ---------------------------------------------- #
t_pts = [0,    2,    6,    10,   12,   19,   23,   25]
s_pts = [0.30, 0.30, 0.70, 0.25, 0.15, 0.85, 0.25, 0.25]

t_full = np.linspace(0, 25, 500)
s_full = np.interp(t_full, t_pts, s_pts)

# -- Figure ---------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.40))

for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
ax.spines['left'].set_color(P["grid"])
ax.spines['bottom'].set_color(P["grid"])

ax.plot(t_full, s_full, color=P["neutral"], zorder=3, solid_capstyle='round')

ax.set_xlim(-0.5, 27.5)
ax.set_ylim(0.0, 1.0)
ax.set_xticks([])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}'))
ax.tick_params(axis='y', color=P["grid"])
ax.tick_params(axis='x', length=0)
ax.set_ylabel('State of Charge  (–)')
ax.set_xlabel('Time  (hours)')

# -- Semantic colours ------------------------------------------------------ #
c_lo  = TUDELFT["red"]      # lower turning points
c_hi  = TUDELFT["dgreen"]   # upper turning points
c_ann = TUDELFT["blue"]     # annotations

# -- Turning point markers ------------------------------------------------- #
# CHANGED (1): (10, 0.25) removed -- monotone descent, not a local minimum.
for t_tp, s_tp in [(2, 0.30), (12, 0.15), (23, 0.25)]:
    ax.plot(t_tp, s_tp, 'o', ms=9, mfc='none', mec=c_lo, mew=1.5, zorder=5)
    ax.plot(t_tp, s_tp, 'o', ms=3, mfc=c_lo, mec=c_lo, zorder=6)

for t_tp, s_tp in [(6, 0.70), (19, 0.85)]:
    ax.plot(t_tp, s_tp, 'o', ms=9, mfc='none', mec=c_hi, mew=1.5, zorder=5)
    ax.plot(t_tp, s_tp, 'o', ms=3, mfc=c_hi, mec=c_hi, zorder=6)

# -- Two bracketed half-cycles --------------------------------------------- #
y_brk = 0.08

# charge half-cycle: 0.30 -> 0.70
ax.annotate('', xy=(6, y_brk), xytext=(2, y_brk),
            arrowprops=dict(arrowstyle='<->', color=c_ann, lw=1.2,
                            mutation_scale=8, shrinkA=0, shrinkB=0))
for t_s in [2, 6]:
    ax.plot([t_s, t_s], [y_brk - 0.016, y_brk + 0.016], color=c_ann, lw=1.1)
ax.text(4, y_brk - 0.040, 'half-cycle', ha='center', va='top',
        fontsize=FS_ANNOT, color=c_ann)

# CHANGED (2): discharge half-cycle now 0.70 -> 0.15, i.e. t = 6 to 12
ax.annotate('', xy=(12, y_brk), xytext=(6, y_brk),
            arrowprops=dict(arrowstyle='<->', color=c_ann, lw=1.2,
                            mutation_scale=8, shrinkA=0, shrinkB=0,
                            linestyle='--'))
for t_s in [6, 12]:
    ax.plot([t_s, t_s], [y_brk - 0.016, y_brk + 0.016],
            color=c_ann, lw=1.1, ls='--')
ax.text(9, y_brk - 0.040, 'half-cycle', ha='center', va='top',
        fontsize=FS_ANNOT, color=c_ann, style='italic')

# -- Annotated half-cycle: 0.15 -> 0.85 ------------------------------------ #
lo_B, hi_B = 0.15, 0.85
delta_B = hi_B - lo_B
sigma_B = (hi_B + lo_B) / 2
t_brk = 21.5
sw = 0.35

# CHANGED (3): guide lines tie the arrow to the rising limb it describes
ax.plot([12, t_brk], [lo_B, lo_B], color=c_ann, lw=0.7, ls=':',
        alpha=0.55, zorder=2)
ax.plot([19, t_brk], [hi_B, hi_B], color=c_ann, lw=0.7, ls=':',
        alpha=0.55, zorder=2)

ax.annotate('', xy=(t_brk, hi_B), xytext=(t_brk, lo_B),
            arrowprops=dict(arrowstyle='<->', color=c_ann, lw=1.4,
                            mutation_scale=8, shrinkA=0, shrinkB=0))
for y in [lo_B, hi_B]:
    ax.plot([t_brk - sw, t_brk + sw], [y, y], color=c_ann, lw=1.1)

ax.plot([t_brk - sw * 1.6, t_brk + sw * 1.6], [sigma_B, sigma_B],
        color=c_ann, lw=0.9, ls='--', alpha=0.6)

ax.text(t_brk + 1.0, (hi_B + lo_B) / 2, r'$\delta$ = %.2f' % delta_B,
        va='center', ha='left', fontsize=FS_LABEL, color=c_ann)
ax.text(t_brk - 1.0, sigma_B + 0.02, r'$\sigma$ = %.2f' % sigma_B,
        va='bottom', ha='right', fontsize=FS_BASE, color=c_ann, style='italic')

# -- Save ------------------------------------------------------------------ #
if OUTPUT not in ("png", "pdf", "both"):
    raise ValueError(f'OUTPUT must be "png", "pdf" or "both", not {OUTPUT!r}')
save(fig, Path(__file__).parent, "rainflow_cycle_illustration")