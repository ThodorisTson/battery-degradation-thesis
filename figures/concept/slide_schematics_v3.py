r"""
slide_schematics_v3.py  (patched from v2)

Produces two figures:
  fig:xu_stress      -- physical reference model stress factors S_delta, S_sigma
  fig:shi_polynomial -- polynomial model Phi_shi(delta) and its second derivative

Changes vs v2:
  1. Axis label "Cycle depth of discharge" -> "Cycle amplitude" (3 occurrences).
     delta is the per-cycle amplitude; DoD is the window parameter d = 1 - sigma_min.
  2. k3 annotation now renders as 3.24 x 10^-5 instead of the mathtext "3.24e-05".
  3. Phi'' symbol reordered to \Phi_\mathrm{shi}'' so the primes follow the subscript.
  4. Phi_shi colour changed from blue to PALETTE["secondary"] so the polynomial model
     is not drawn in the same hue as the physical reference model S_delta.
  5. Annotation boxes placed in axes-fraction coordinates, so they no longer move
     if the fitted coefficients change.
  6. Annotation font size unified to FS_ANNOT across both figures.
  7. "(-)" added to every y-axis label for consistency with the x-axis labels.
  8. Headroom added above the Phi'' curve, which previously touched the top spine.
  9. Output format is selected by OUTPUT below.
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

# -- Model parameters ------------------------------------------------------ #
# Xu coefficients: Table 2.1 (k_d1, k_d2, k_d3), Section 2.6.2 (k_sigma,
# sigma_ref). Shi coefficients: Equation 3.10.
K1, K2_EXP, K3C  = 1.40e5, -0.501, -1.23e5
K_SIGMA, SIG_REF = 1.04, 0.50
K3_SHI, K4_SHI   = 3.2418e-5, 1.1785


def s_dod(d):
    return 1.0 / (K1 * np.clip(d, 1e-6, 1.0) ** K2_EXP + K3C)


def s_soc(sigma):
    return np.exp(K_SIGMA * (np.asarray(sigma, float) - SIG_REF))


def phi_shi(d):
    return K3_SHI * np.clip(d, 1e-9, 1.0) ** K4_SHI


def sci_tex(x, sig=2):
    """Format x as a mathtext string a x 10^b, e.g. 3.24 x 10^{-5}."""
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10.0 ** exp
    return rf"{mant:.{sig}f} \times 10^{{{exp}}}"


# -- Semantic colours ------------------------------------------------------ #
# CHANGED (4): the two models now carry different hues across Ch.2 and Ch.3.
C_SD  = P["fill_a"]      # S_delta   -- TU Delft blue    #0076C2
C_SS  = P["fill_b"]      # S_sigma   -- TU Delft orange  #EC6842
C_PHI = P["secondary"]   # Phi_shi   -- TU Delft dark red #A50034
C_CV  = TUDELFT["dgreen"]  # Phi''   -- diagnostic curve, not a model branch

if OUTPUT not in ("png", "pdf", "both"):
    raise ValueError(f'OUTPUT must be "png", "pdf" or "both", not {OUTPUT!r}')

OUT = Path(__file__).parent

# ========================================================================== #
# Fig 2.2 -- physical reference model stress factors
# ========================================================================== #
fig22, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize(1.0, aspect=0.48))

d_arr   = np.linspace(0.02, 0.80, 400)
sig_arr = np.linspace(0.10, 0.90, 400)

# -- Left: S_delta(delta) -------------------------------------------------- #
sd = s_dod(d_arr)
ax1.fill_between(d_arr, 0, sd * 1e5, color=C_SD, alpha=0.08)
ax1.plot(d_arr, sd * 1e5, color=C_SD, zorder=3)
ax1.set_xlabel(r'Cycle amplitude $\delta$  (–)')          # CHANGED (1)
ax1.set_ylabel(r'$S_\delta(\delta) \times 10^5$  (–)')     # CHANGED (7)
ax1.set_xlim(0.02, 0.80)
ax1.set_ylim(bottom=0)

# -- Right: S_sigma(sigma) ------------------------------------------------- #
ss = s_soc(sig_arr)
ax2.fill_between(sig_arr, 1.0, ss, where=(ss >= 1.0), color=C_SS, alpha=0.10)
ax2.fill_between(sig_arr, ss,  1.0, where=(ss < 1.0),  color=C_SS, alpha=0.05)
ax2.plot(sig_arr, ss, color=C_SS, zorder=3)
ax2.axvline(SIG_REF, color=P["neutral"], lw=0.8, ls=':', alpha=0.5, zorder=1)
ax2.axhline(1.0,     color=P["neutral"], lw=0.7, ls=':', alpha=0.4, zorder=1)
# CHANGED (5): axes-fraction placement
ax2.text(0.56, 0.06,
         r'$\sigma_\mathrm{ref} = 0.50$' + '\n' + r'$S_\sigma = 1.0$',
         transform=ax2.transAxes,
         fontsize=FS_ANNOT, color=C_SS, va='bottom', ha='left',
         bbox=dict(boxstyle='round,pad=0.40', facecolor='white',
                   edgecolor=C_SS, linewidth=0.8, alpha=0.92))
ax2.set_xlabel(r'Cycle mean SoC $\sigma$  (–)')
ax2.set_ylabel(r'$S_\sigma(\sigma)$  (–)')
ax2.set_xlim(0.10, 0.90)

save(fig22, OUT, 'fig22_xu_stress')
plt.close(fig22)

# ========================================================================== #
# Fig 3.6 -- polynomial model and its second derivative
# ========================================================================== #
fig36, (ax3, ax4) = plt.subplots(1, 2, figsize=figsize(1.0, aspect=0.48))

d_all   = np.linspace(0.02, 0.80, 400)
phi_all = phi_shi(d_all)

# -- Left: Phi_shi(delta) -------------------------------------------------- #
ax3.fill_between(d_all, 0, phi_all * 1e5, color=C_PHI, alpha=0.08)
ax3.plot(d_all, phi_all * 1e5, color=C_PHI, zorder=3)
# CHANGED (2)(5)(6)
ax3.text(0.05, 0.94,
         rf'$k_3 = {sci_tex(K3_SHI)}$' + '\n' + rf'$k_4 = {K4_SHI:.4f}$',
         transform=ax3.transAxes,
         fontsize=FS_ANNOT, color=C_PHI, va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                   edgecolor=C_PHI, linewidth=0.9, alpha=0.92))
ax3.set_xlabel(r'Cycle amplitude $\delta$  (–)')                    # CHANGED (1)
ax3.set_ylabel(r'$\Phi_\mathrm{shi}(\delta) \times 10^5$  (–)')     # CHANGED (7)
ax3.set_xlim(0.02, 0.80)
ax3.set_ylim(bottom=0)

# -- Right: Phi''_shi(delta) ----------------------------------------------- #
phi_d2 = K3_SHI * K4_SHI * (K4_SHI - 1) * np.clip(d_all, 1e-9, 1.0) ** (K4_SHI - 2)
ax4.fill_between(d_all, 0, phi_d2 * 1e5, color=C_CV, alpha=0.10)
ax4.plot(d_all, phi_d2 * 1e5, color=C_CV, zorder=3)
# CHANGED (5)(6)
ax4.text(0.42, 0.62,
         r"$\Phi_\mathrm{shi}''(\delta) > 0$" + "\neverywhere",     # CHANGED (3)
         transform=ax4.transAxes,
         fontsize=FS_ANNOT, color=C_CV, va='center', ha='left',
         bbox=dict(boxstyle='round,pad=0.42', facecolor='white',
                   edgecolor=C_CV, linewidth=0.9, alpha=0.95))
ax4.set_xlabel(r'Cycle amplitude $\delta$  (–)')                          # CHANGED (1)
ax4.set_ylabel(r"$\Phi_\mathrm{shi}''(\delta) \times 10^5$  (–)")         # CHANGED (3)(7)
ax4.set_xlim(0.02, 0.80)
# CHANGED (8): 8% headroom so the curve no longer touches the top spine
ax4.set_ylim(0, (phi_d2 * 1e5).max() * 1.08)

save(fig36, OUT, 'fig36_shi_convex')
plt.close(fig36)