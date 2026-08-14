r"""
xu_sdelta_second_derivative.py   (thesis label fig:xu_sdelta_d2)

Second derivative of the physical reference model cycle amplitude stress function. Shows the non-convex region below the sign change at delta = 0.1437.

Rewritten from the standalone slide version:
  1. Driven by thesis_style.py -- fonts, colours, line widths, output format.
  2. x-axis label "Cycle depth of discharge" -> "Cycle amplitude". delta is the per-cycle amplitude; DoD is the window parameter d = 1 - sigma_min.
  3. np.clip on the DATA removed. The old clip at 1e-4 capped the curve at 10 and produced a flat plateau above delta ~ 0.55 that is not in the function
     (S_delta'' reaches 34.5 at delta = 0.80). ylim now clips the rendering only.
  4. Title removed; the caption carries the explanation.
  5. Output format is selected by OUTPUT below.

Runs in VS Code on Windows: matplotlib + numpy only, bundled font, no LaTeX.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from degradation.style import (apply_thesis_style, figsize, TUDELFT,
                               FS_ANNOT, FS_LEGEND)

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

# -- Model parameters ------------------------------------------------------ #
# Xu coefficients: Table 2.1 (k_d1, k_d2, k_d3). Inflection: Equation 2.11.
K1, K2_EXP, K3C = 1.40e5, -0.501, -1.23e5
BOUNDARY = 0.1437          # sign change of S_delta''


def s_dod_d2(d):
    """Second derivative of the Xu cycle amplitude stress function."""
    d   = np.clip(np.asarray(d, float), 1e-6, 1.0)
    D   = K1 * d ** K2_EXP + K3C
    Dp  = K1 * K2_EXP * d ** (K2_EXP - 1)
    Dpp = K1 * K2_EXP * (K2_EXP - 1) * d ** (K2_EXP - 2)
    return 2 * Dp ** 2 / D ** 3 - Dpp / D ** 2


# -- Semantic colours ------------------------------------------------------ #
C_CURVE = P["primary"]        # physical reference model  -- TU Delft navy
C_SHADE = TUDELFT["orange"]   # non-convex region

# -- Data ------------------------------------------------------------------ #
d_all = np.linspace(0.02, 0.80, 800)
xu_d2 = s_dod_d2(d_all) * 1e5          # CHANGED (3): no data clipping

d_nc  = d_all[d_all <= BOUNDARY]
xu_nc = s_dod_d2(d_nc) * 1e5

# -- Figure ---------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.45))

ax.fill_between(d_nc, xu_nc, 0, color=C_SHADE, alpha=0.18, lw=0, zorder=1,
                label=r'non-convex region ($\delta < 0.144$)')
ax.axvline(BOUNDARY, color=C_SHADE, lw=1.0, ls='--', alpha=0.85, zorder=2)
ax.axhline(0, color=P["neutral"], lw=0.8, alpha=0.7, zorder=3)
ax.plot(d_all, xu_d2, color=C_CURVE, zorder=4,
        label=r"$S_\delta''(\delta)$, Xu et al. (2016)")

ax.set_xlim(0.02, 0.80)
ax.set_ylim(-2.5, 12.0)                # CHANGED (3): display limit, not a data cap
ax.set_xlabel(r'Cycle amplitude $\delta$  (–)')             # CHANGED (2)
ax.set_ylabel(r"$S_\delta''(\delta) \times 10^5$  (–)")
ax.legend(fontsize=FS_LEGEND, frameon=False, loc='upper left')

# -- Save ------------------------------------------------------------------ #
if OUTPUT not in ("png", "pdf", "both"):
    raise ValueError(f'OUTPUT must be "png", "pdf" or "both", not {OUTPUT!r}')
save(fig, Path(__file__).parent, "fig34_xu_sdelta_d2")