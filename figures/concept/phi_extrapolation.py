r"""
fig35_phi_extrapolation.py   (Fig. 3.5)

Two-panel comparison of the physical reference model stress function S_delta and
the fitted polynomial Phi_shi. Top: function values. Bottom: second derivatives,
showing that Phi_shi stays convex where S_delta does not.

Rewritten from the standalone slide version:
  1. Driven by thesis_style.py -- fonts, colours, line widths, output format.
  2. x-axis label "Cycle depth of discharge" -> "Cycle amplitude".
  3. np.clip on the DATA removed. The old clip at 1e-4 capped both curves at 10,
     producing flat plateaus that are not in the functions (S_delta'' reaches
     34.5 at delta = 0.80, Phi_shi'' reaches 17.0 at delta = 0.02).
     ylim now clips the rendering only.
  4. Suptitle removed; the caption carries the explanation.
  5. \Phi_{shi} -> \Phi_\mathrm{shi} so the subscript is upright, matching
     Chapters 2 and 3 and Figure 3.6.
  6. PDF written first, PNG for preview.

Coefficients k3 and k4 are the fit over delta in [0.15, 0.80] for the 10-90%
SoC window. Regenerate if degradation_shi.fit_shi_polynomial(0.10, 0.90) changes.

Requires thesis_style.py in the same folder (or on PYTHONPATH).
Runs in VS Code on Windows: matplotlib + numpy only, bundled font, no LaTeX.
"""
import sys
from pathlib import Path

# --- path guard ----------------------------------------------------------- #
for _d in Path(__file__).resolve().parents:
    if (_d / "thesis_style.py").exists():
        sys.path.insert(0, str(_d))
        break
else:
    raise FileNotFoundError("thesis_style.py not found in any parent folder")

import numpy as np
import matplotlib.pyplot as plt
from thesis_style import (apply_thesis_style, figsize, TUDELFT,
                          FS_ANNOT, FS_LEGEND)

P = apply_thesis_style(palette="brand", usetex=False)

# -- Model parameters ------------------------------------------------------ #
K1, K2_EXP, K3C     = 1.40e5, -0.501, -1.23e5
K3_SHI, K4_SHI      = 3.2418e-5, 1.1785     # fit over [0.15, 0.80], 10-90% window
BOUNDARY, FIT_FLOOR = 0.1437, 0.15


def s_dod(d):
    d = np.clip(np.asarray(d, float), 1e-6, 1.0)
    return 1.0 / (K1 * d ** K2_EXP + K3C)


def phi_shi(d):
    d = np.clip(np.asarray(d, float), 1e-9, 1.0)
    return K3_SHI * d ** K4_SHI


def s_dod_d2(d):
    d   = np.clip(np.asarray(d, float), 1e-6, 1.0)
    D   = K1 * d ** K2_EXP + K3C
    Dp  = K1 * K2_EXP * d ** (K2_EXP - 1)
    Dpp = K1 * K2_EXP * (K2_EXP - 1) * d ** (K2_EXP - 2)
    return 2 * Dp ** 2 / D ** 3 - Dpp / D ** 2


def phi_shi_d2(d):
    d = np.clip(np.asarray(d, float), 1e-9, 1.0)
    return K3_SHI * K4_SHI * (K4_SHI - 1) * d ** (K4_SHI - 2)


# -- Semantic colours ------------------------------------------------------ #
C_XU    = P["primary"]        # physical reference model -- TU Delft navy
C_SHI   = P["secondary"]      # polynomial model         -- TU Delft dark red
C_SHADE = TUDELFT["orange"]   # non-convex region

# -- Data ------------------------------------------------------------------ #
d = np.linspace(0.02, 0.80, 800)
xu_val, shi_val = s_dod(d) * 1e5, phi_shi(d) * 1e5
xu_d2,  shi_d2  = s_dod_d2(d) * 1e5, phi_shi_d2(d) * 1e5   # CHANGED (3)
d_nc   = d[d <= BOUNDARY]
xu_ncd2 = s_dod_d2(d_nc) * 1e5

# -- Figure ---------------------------------------------------------------- #
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                               figsize=figsize(1.0, aspect=0.80))

for ax in (ax1, ax2):
    ax.axvspan(0.02, BOUNDARY, color=C_SHADE, alpha=0.12, lw=0, zorder=0)
    ax.axvline(BOUNDARY,  color=C_SHADE,    lw=1.0, ls='--', alpha=0.85, zorder=2)
    ax.axvline(FIT_FLOOR, color=P["grid"],  lw=0.8, ls='--', alpha=0.9,  zorder=2)
    ax.set_xlim(0.02, 0.80)

# -- Top: function values -------------------------------------------------- #
ax1.plot(d, xu_val,  color=C_XU,  zorder=3,
         label=r'$S_\delta(\delta)$, physical reference model')
ax1.plot(d, shi_val, color=C_SHI, ls='--', zorder=3,
         label=r'$\Phi_\mathrm{shi}(\delta) = k_3\,\delta^{k_4}$, polynomial model')
ax1.set_ylabel(r'$\Phi(\delta) \times 10^5$  (–)')
ax1.set_ylim(bottom=0)
ax1.legend(fontsize=FS_LEGEND, frameon=False, loc='upper left',
           handlelength=2.0, labelspacing=0.35)
ax1.text(0.074, 0.55, 'non-\nconvex', transform=ax1.get_xaxis_transform(),
         ha='center', va='center', fontsize=FS_ANNOT,
         color=C_SHADE, style='italic')

# -- Bottom: second derivatives -------------------------------------------- #
ax2.fill_between(d_nc, xu_ncd2, 0, color=C_SHADE, alpha=0.20, lw=0, zorder=1)
ax2.axhline(0, color=P["neutral"], lw=0.8, alpha=0.7, zorder=2)
ax2.plot(d, xu_d2,  color=C_XU,  zorder=3, label=r"$S_\delta''(\delta)$")
ax2.plot(d, shi_d2, color=C_SHI, ls='--', zorder=3,
         label=r"$\Phi_\mathrm{shi}''(\delta)$")
ax2.set_ylabel(r"$\Phi''(\delta) \times 10^5$  (–)")
ax2.set_ylim(-2.5, 12.0)                # CHANGED (3): display limit, not a data cap
ax2.set_xlabel(r'Cycle amplitude $\delta$  (–)')            # CHANGED (2)
ax2.legend(fontsize=FS_LEGEND, frameon=False, loc='center right',
           handlelength=2.0, labelspacing=0.35)

# -- Save ------------------------------------------------------------------ #
out = Path(__file__).parent
fig.savefig(out / 'fig35_phi_extrapolation.pdf')
fig.savefig(out / 'fig35_phi_extrapolation.png', dpi=300)
print(f"Saved -> {out / 'fig35_phi_extrapolation.pdf'} and .png")
