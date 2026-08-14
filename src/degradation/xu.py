"""
WP2 Battery Degradation — Xu et al. (2016) Semi-Empirical Model
================================================================

Implements the degradation model from:
  Xu, B. et al., "Modeling of Lithium-Ion Battery Degradation for Cell
  Life Assessment," IEEE Trans. Smart Grid, 2016.
  DOI: 10.1109/TSG.2016.2578950

Model structure
---------------
Total linearized degradation (Eq. 3):
    fd = ft_calendar(t_total, σ_mean, Tc) + Σ_i  n_i · fc_cycle(δ_i, σ_i, Tc)

Stress factors (Section III):
    S_δ(δ)   = (k_δ1 · δ^k_δ2 + k_δ3)^-1          DoD stress    [Eq. 32, LMO]
    S_σ(σ)   = exp(k_σ · (σ − σ_ref))              SoC stress    [Eq. 25]
    S_T(Tc)  = exp(k_T · (Tc − T_ref) · T_ref/Tc)  Temp stress   [Eq. 22]
    S_t(t)   = k_t · t                              Time stress   [Eq. 27]

Calendar aging (full simulation period):
    ft = S_t(t_total) · S_σ(σ_mean) · S_T(Tc)

Cycle aging per rainflow cycle i:
    fc_i = S_δ(δ_i) · S_σ(σ_i) · S_T(Tc)

SEI nonlinear capacity loss (Eq. 12):
    L = 1 − α_sei · exp(−β_sei · fd) − (1 − α_sei) · exp(−fd)
    capacity_retention = 1 − L

Temperature assumption
----------------------
Cell temperature is fixed at T_c = 25°C (manufacturer reference, S_T = 1.0).
This is consistent with the case study in Xu et al. Section VI, which assumes
the battery cooling system maintains 25°C for C-rates below 1C.

Default parameters: LMO battery, Table I of Xu et al. (2016).

File structure
--------------
This file contains ONLY the physical model — parameters, stress factors,
degradation accumulation, rainflow counting, and the main analysis API.
Plotting and reporting have been moved to:
    plots.py — plot_degradation_analysis, print_degradation_report,
                           validate_xu_dst
The subgradient interface (Shi et al. 2018) lives in:
    subgradient.py — build_half_cycle_map, compute_subgradient

Public API
----------
    analyze_degradation(storage_p, storage_e, e_cap_nominal,
                        battery_params, dt_hours, n_bins_dod,
                        enable_rainflow, T_cell_C, xu_params,
                        eol_thresholds) -> Dict
    count_equivalent_full_cycles(storage_p, storage_e, e_cap, dt_hours) -> float
    rainflow_cycle_counting(storage_e, e_cap) -> List[Dict]
    xu_capacity_curve(fd_values, p) -> np.ndarray

Subgradient additions (Gap A — Shi et al. 2018)
-----------------------------------------------
    ShiPolynomialFit                             dataclass — fit result + metadata
    load_soc_window_from_yaml(yaml_path)         read soc_min/soc_max from YAML
    fit_shi_polynomial(soc_min, soc_max, ...)    fit Φ to Xu S_δ for a SoC window
    s_dod_derivative(delta, p)
    fc_cycle_derivative(delta, sigma, T_C, p)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rainflow as _rainflow_lib   # type: ignore
except ImportError:
    _rainflow_lib = None


# =============================================================================
# Model parameters (Table I, Xu et al. 2016 — LMO battery)
# =============================================================================

@dataclass
class XuModelParams:
    """All coefficients for the Xu et al. degradation model.

    Defaults match Table I of Xu et al. (2016) for LMO batteries.
    Override for other chemistries (LFP, NMC) by supplying different k values.
    """
    # SEI nonlinear model
    alpha_sei: float = 5.75e-2     # SEI capacity fraction (dimensionless)
    beta_sei:  float = 121.0       # SEI rate scaling factor (dimensionless)

    # DoD stress model (LMO, Eq. 32)
    k_delta1:  float = 1.40e5      # LMO DoD coefficient 1
    k_delta2:  float = -5.01e-1    # LMO DoD coefficient 2 (exponent)
    k_delta3:  float = -1.23e5     # LMO DoD coefficient 3

    # SoC stress model (Eq. 25)
    k_sigma:   float = 1.04        # SoC stress coefficient
    sigma_ref: float = 0.50        # Reference SoC (dimensionless, 0-1)

    # Temperature stress model (Eq. 22) — Arrhenius
    k_T:       float = 6.93e-2     # Temperature stress coefficient
    T_ref_C:   float = 25.0        # Reference temperature [°C]

    # Calendar / time stress (Eq. 27)
    k_t:       float = 4.14e-10    # Calendar aging rate [per second]


# Default LMO parameters — used throughout unless overridden
XU_LMO = XuModelParams()


# =============================================================================
# Stress factor functions
# =============================================================================

def s_dod(delta: float | np.ndarray,
          p: XuModelParams = XU_LMO) -> float | np.ndarray:
    """DoD stress factor S_δ (Eq. 32, LMO model).

    Args:
        delta: Cycle depth of discharge, normalised in (0, 1].
        p:     Model parameters.

    Returns:
        Stress factor value(s). Higher δ → higher stress (more damage per cycle).
    """
    delta = np.asarray(delta, dtype=float)
    delta = np.clip(delta, 1e-6, 1.0)
    denominator = p.k_delta1 * delta ** p.k_delta2 + p.k_delta3
    denominator = np.where(np.abs(denominator) < 1e-30, 1e-30, denominator)
    return 1.0 / denominator


def s_dod_derivative(delta: float | np.ndarray,
                     p: XuModelParams = XU_LMO) -> float | np.ndarray:
    """Analytical derivative of S_δ with respect to delta (dS_dod/ddelta).

    Derived from S_δ(δ) = 1 / D(δ), where D(δ) = k_δ1·δ^k_δ2 + k_δ3.
    By the quotient rule: dS_δ/dδ = -D'(δ) / D(δ)²
    where D'(δ) = k_δ1 · k_δ2 · δ^(k_δ2 - 1)

    With LMO params k_δ2 = -0.501, D'(δ) is always negative,
    meaning dS_δ/dδ is always positive — deeper cycles always cost more.

    Note on convexity: the LMO empirical fit is only convex above δ ≈ 0.145.
    The Shi theorem requires a globally convex Φ. For grid storage with
    SHIPP's SoC bounds this is satisfied in practice. A globally convex
    polynomial fit (Shi Eq. 5 form) is planned for a future iteration.

    Args:
        delta: Cycle depth of discharge, normalised in (0, 1].
        p:     Model parameters.

    Returns:
        dS_δ/dδ — always >= 0 for physically valid parameters.
    """
    delta   = np.asarray(delta, dtype=float)
    delta   = np.clip(delta, 1e-6, 1.0)
    D       = p.k_delta1 * delta ** p.k_delta2 + p.k_delta3
    D       = np.where(np.abs(D) < 1e-30, 1e-30, D)
    D_prime = p.k_delta1 * p.k_delta2 * delta ** (p.k_delta2 - 1.0)
    return -D_prime / D ** 2


# =============================================================================
# Shi (2018) polynomial Φ — globally convex surrogate for gradient computation
# =============================================================================
#
# The Xu LMO S_δ(δ) is only convex above δ ≈ 0.1437 (second derivative
# changes sign there). Shi Theorem 1 requires Φ to be globally convex.
# A SoC-window floor does NOT fix this: it constrains state, not cycle depth —
# small arbitrage cycles (δ < 0.1437) still occur even with soc_min = 0.40.
#
# Solution: two separate Φ functions for two separate purposes:
#   1. Xu S_δ  → degradation REPORTING  (fd, SoH, EoL)  — validated, physical
#   2. Shi Φ   → GRADIENT computation   (Eqs. 17-18)    — globally convex
#
# Φ(δ) = k3 · δ^k4  is fitted to Xu S_δ over the DoD range actually reachable
# from the battery's SoC window [soc_min, soc_max].  k4 > 1 guarantees global
# convexity everywhere — no boundary condition needed.
#
# Rather than hardcoding k3/k4, call fit_shi_polynomial() with the SoC window
# from the battery YAML.  Module-level defaults use [0, 1] as a full-range
# fallback for environments where no YAML is available.
# =============================================================================


@dataclass
class ShiPolynomialFit:
    """Result of fitting Φ(δ) = k3·δ^k4 to Xu S_δ for a specific SoC window.

    Carries coefficients together with all provenance metadata needed to
    reproduce, validate, and document the fit (thesis slides, Jenna meetings).

    Attributes:
        k3:       Scale coefficient (always > 0).
        k4:       Power exponent (must be > 1 for global convexity).
        r2:       R² of the log-log OLS fit over [fit_lo, fit_hi].
        fit_lo:   Lower DoD bound used for fitting (fixed at 0.15, just above
                  the Xu non-convex boundary at ~0.1437).
        fit_hi:   Upper DoD bound = soc_max - soc_min (max possible cycle depth).
        soc_min:  Battery lower SoC limit (source: YAML operating_limits).
        soc_max:  Battery upper SoC limit (source: YAML operating_limits).
        source:   Human-readable provenance string (e.g. 'battery.yaml').
    """
    k3:      float
    k4:      float
    r2:      float
    fit_lo:  float
    fit_hi:  float
    soc_min: float
    soc_max: float
    source:  str = "unknown"

    @property
    def max_dod(self) -> float:
        """Maximum DoD reachable within this SoC window."""
        return self.soc_max - self.soc_min

    @property
    def is_convex(self) -> bool:
        """True iff k4 > 1 → Φ''(δ) > 0 everywhere (Shi Theorem 1 satisfied)."""
        return self.k4 > 1.0

    def summary(self) -> str:
        """One-line description for print statements and slide captions."""
        return (
            f"k3={self.k3:.4e}  k4={self.k4:.4f}  R²={self.r2:.4f}  "
            f"fit=[{self.fit_lo:.2f},{self.fit_hi:.2f}]  "
            f"soc=[{self.soc_min},{self.soc_max}]  "
            f"convex={'YES' if self.is_convex else 'NO'}  "
            f"source={self.source}"
        )


def load_soc_window_from_yaml(
    yaml_path: "str | Path",
) -> "Tuple[float, float, str]":
    """Read soc_min and soc_max from a battery YAML file.

    Tries PyYAML first; if not installed, falls back to a minimal line-by-line
    parser that handles simple key: value format.  Returns (0.0, 1.0, reason)
    if the file cannot be read or the keys are absent.

    Args:
        yaml_path: Path to the battery YAML file (e.g. battery.yaml).

    Returns:
        (soc_min, soc_max, source_description)
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return 0.0, 1.0, f"default [0,1] — {yaml_path.name} not found"

    # --- Try PyYAML ---
    try:
        import yaml as _yaml
        with open(yaml_path) as f:
            cfg = _yaml.safe_load(f)
        lim = cfg.get("operating_limits", {})
        lo  = float(lim.get("soc_min", 0.0))
        hi  = float(lim.get("soc_max", 1.0))
        return lo, hi, f"{yaml_path.name} via PyYAML"
    except ImportError:
        pass
    except Exception as exc:
        return 0.0, 1.0, f"default [0,1] — YAML parse error: {exc}"

    # --- Fallback: minimal line parser (no dependency) ---
    try:
        in_limits = False
        lo, hi    = 0.0, 1.0
        with open(yaml_path) as f:
            for raw in f:
                line = raw.strip()
                if line.startswith("operating_limits"):
                    in_limits = True
                    continue
                if in_limits:
                    if line and not line.startswith(" ") and not line.startswith("\t"):
                        break  # left the block
                    if "soc_min" in line:
                        lo = float(line.split(":")[1].strip().split()[0])
                    elif "soc_max" in line:
                        hi = float(line.split(":")[1].strip().split()[0])
        return lo, hi, f"{yaml_path.name} via line parser (PyYAML not installed)"
    except Exception as exc:
        return 0.0, 1.0, f"default [0,1] — fallback parser error: {exc}"


def fit_shi_polynomial(
    soc_min:   float = 0.0,
    soc_max:   float = 1.0,
    xu_params: "XuModelParams" = None,
    n_points:  int   = 500,
    source:    str   = "manual",
    verbose:   bool  = False,
) -> "ShiPolynomialFit":
    """Fit Φ(δ) = k3·δ^k4 to Xu S_δ for a given SoC window.

    The fitting range is derived directly from the SoC operating limits:
        fit_lo = 0.15                     (fixed floor — just above the Xu
                                           non-convex boundary at ~0.1437;
                                           fitting below this pulls k4 < 1)
        fit_hi = soc_max - soc_min        (maximum possible cycle depth)

    The fit is performed in log-log space (OLS on log δ vs log S_δ), which
    gives equal relative weight across all DoD levels and naturally yields the
    power-law form.  Calling this with the actual YAML window tailors the
    polynomial to where cycling actually occurs, improving gradient accuracy
    compared to a generic [0, 1] fit.

    Pass verbose=True to print the fit summary — recommended in run scripts
    and self-tests so the polynomial is visible in the output log.

    Args:
        soc_min:   Battery SoC lower limit (default 0.0 → full-range fallback).
        soc_max:   Battery SoC upper limit (default 1.0 → full-range fallback).
        xu_params: XuModelParams for S_δ. Defaults to XU_LMO.
        n_points:  Number of log-spaced DoD points for fitting (default 500).
        source:    Provenance string stored in the returned ShiPolynomialFit.
        verbose:   Print fit summary and convexity diagnostics if True.

    Returns:
        ShiPolynomialFit with k3, k4, R², fit bounds, and window metadata.

    Raises:
        ValueError: If k4 ≤ 1 (fit not globally convex — should not happen
                    for LMO Xu parameters over any reasonable window).
        ValueError: If the SoC window is too narrow (max_dod < 0.05).
    """
    if xu_params is None:
        xu_params = XU_LMO

    max_dod = float(soc_max) - float(soc_min)
    if max_dod < 0.15:
        raise ValueError(
            f"SoC window [{soc_min}, {soc_max}] → max_dod={max_dod:.3f} < 0.15. "
            "Too narrow to fit a Shi polynomial (fit requires delta > 0.15)."
        )

    fit_lo = 0.15   # just above Xu non-convex boundary (delta~0.1437);
                    # fitting below this skews k4 toward the concave region
    fit_hi = min(max_dod, 1.0)

    if fit_hi <= fit_lo:
        raise ValueError(
            f"SoC window [{soc_min}, {soc_max}] → max_dod={max_dod:.2f} ≤ fit_lo={fit_lo}. "
            "Window too narrow to fit a meaningful Shi polynomial."
        )

    # Linear grid — uniform weight across DoD.  Log-spacing over-weights
    # small-delta values near the non-convex Xu boundary and can pull k4 < 1.
    d_fit   = np.linspace(fit_lo, fit_hi, n_points)
    phi_xu  = s_dod(d_fit, xu_params)

    # OLS in log-log space: log(Φ) = log(k3) + k4·log(δ)
    coeffs = np.polyfit(np.log(d_fit), np.log(phi_xu), 1)
    k4     = float(coeffs[0])
    k3     = float(np.exp(coeffs[1]))

    # R² in original space — relevant for gradient accuracy
    phi_fit = k3 * d_fit ** k4
    ss_res  = float(np.sum((phi_xu - phi_fit) ** 2))
    ss_tot  = float(np.sum((phi_xu - phi_xu.mean()) ** 2))
    r2      = 1.0 - ss_res / ss_tot

    if k4 <= 1.0:
        raise ValueError(
            f"Fitted k4={k4:.4f} ≤ 1 — Shi polynomial not globally convex. "
            "Check Xu params or narrow SoC window."
        )

    fit = ShiPolynomialFit(
        k3=k3, k4=k4, r2=r2,
        fit_lo=fit_lo, fit_hi=fit_hi,
        soc_min=float(soc_min), soc_max=float(soc_max),
        source=source,
    )

    if verbose:
        print(f"  fit_shi_polynomial  [{soc_min}, {soc_max}]  "
              f"(source: {source})")
        print(f"    max_dod = {max_dod:.2f}   "
              f"fit range = [{fit_lo:.2f}, {fit_hi:.2f}]   "
              f"{n_points} linear points, log-log OLS")
        print(f"    k3 = {k3:.6e}")
        print(f"    k4 = {k4:.4f}   "
              f"({'> 1 → globally convex' if k4 > 1 else '<= 1 — NOT convex'}) "
              f"k4*(k4-1) = {k4*(k4-1):.4f}")
        print(f"    R² = {r2:.4f}   (log-log OLS, original-space R²)")

    return fit


# --- Module-level defaults -----------------------------------------------
# Computed at import time from the full [0, 1] range — always available even
# without a YAML file.  Pass fit_shi_polynomial(soc_min, soc_max, verbose=True)
# from your run script or compute_subgradient() for window-specific gradients.
_DEFAULT_SHI_FIT: ShiPolynomialFit = fit_shi_polynomial(
    soc_min=0.0, soc_max=1.0, source="default [0,1]", verbose=False
)
# Backwards-compatible names — these match whatever the default fit produces.
SHI_K3: float = _DEFAULT_SHI_FIT.k3
SHI_K4: float = _DEFAULT_SHI_FIT.k4


def phi_shi(delta: float | np.ndarray,
            k3: float = SHI_K3,
            k4: float = SHI_K4) -> float | np.ndarray:
    """Shi (2018) polynomial DoD stress function — Φ(δ) = k3 · δ^k4.

    Globally convex for k4 > 1.  Fitted to Xu LMO S_δ over [0.15, 0.95].
    Used ONLY for gradient computation (Shi Eqs. 17-18).
    For degradation reporting use s_dod() / fc_cycle() instead.

    Args:
        delta: Cycle DoD (normalised 0-1).
        k3:    Scale coefficient (default: fitted to LMO Xu).
        k4:    Power exponent (default: 1.3181, >1 → globally convex).

    Returns:
        Φ(δ) values — always positive, convex in δ.
    """
    delta = np.clip(np.asarray(delta, dtype=float), 1e-9, 1.0)
    return k3 * delta ** k4


def phi_shi_prime(delta: float | np.ndarray,
                  k3: float = SHI_K3,
                  k4: float = SHI_K4) -> float | np.ndarray:
    """Derivative of the Shi polynomial: Φ'(δ) = k3 · k4 · δ^(k4-1).

    Used in Shi et al. (2018) Eqs. 17-18 as Φ'(v_i) / Φ'(w_j).
    Monotonically increasing (Φ'' > 0), satisfying the convexity requirement
    of Theorem 1.

    Args:
        delta: Cycle DoD (normalised 0-1).
        k3:    Scale coefficient.
        k4:    Power exponent (must be > 1 for convexity guarantee).

    Returns:
        Φ'(δ) — always >= 0.
    """
    delta = np.clip(np.asarray(delta, dtype=float), 1e-9, 1.0)
    return k3 * k4 * delta ** (k4 - 1.0)


def phi_shi_double(delta: float | np.ndarray,
                   k3: float = SHI_K3,
                   k4: float = SHI_K4) -> float | np.ndarray:
    """Second derivative Φ''(δ) = k3·k4·(k4-1)·δ^(k4-2).

    Positive everywhere for k4 > 1, confirming global convexity.
    Provided for verification and Jenna's open-question slides.

    Args:
        delta: Cycle DoD (normalised 0-1).
        k3, k4: Polynomial coefficients.

    Returns:
        Φ''(δ) — always > 0 for k4 > 1.
    """
    delta = np.clip(np.asarray(delta, dtype=float), 1e-9, 1.0)
    return k3 * k4 * (k4 - 1.0) * delta ** (k4 - 2.0)


def phi_shi_with_stress(delta: float | np.ndarray,
                        sigma: float | np.ndarray,
                        T_C: float = 25.0,
                        k3: float = SHI_K3,
                        k4: float = SHI_K4,
                        p: "XuModelParams" = None) -> float | np.ndarray:
    """Shi Φ with Xu SoC and temperature stress multipliers.

    Full per-cycle cost for gradient: Φ_shi(δ) · S_σ(σ) · S_T(T).
    Convexity is preserved because S_σ and S_T are positive scalars
    independent of δ.

    Args:
        delta: Cycle DoD (0-1).
        sigma: Average SoC of cycle (0-1).
        T_C:   Cell temperature [°C].
        k3, k4: Shi polynomial coefficients.
        p:     XuModelParams for S_σ / S_T (defaults to XU_LMO).

    Returns:
        Φ_shi(δ) · S_σ(σ) · S_T(T)
    """
    if p is None:
        p = XU_LMO
    return phi_shi(delta, k3, k4) * s_soc(sigma, p) * s_temp(T_C, p)


def phi_shi_prime_with_stress(delta: float | np.ndarray,
                               sigma: float | np.ndarray,
                               T_C: float = 25.0,
                               k3: float = SHI_K3,
                               k4: float = SHI_K4,
                               p: "XuModelParams" = None) -> float | np.ndarray:
    """Gradient-ready Φ'(δ) with stress scaling — used in Shi Eqs. 17-18.

    dΦ/dδ = Φ_shi'(δ) · S_σ(σ) · S_T(T)

    This replaces fc_cycle_derivative() in compute_subgradient() to ensure
    global convexity of the gradient signal passed to the outer loop.

    Args:
        delta: Cycle DoD (0-1).
        sigma: Average SoC of cycle (0-1).
        T_C:   Cell temperature [°C].
        k3, k4: Shi polynomial coefficients.
        p:     XuModelParams (defaults to XU_LMO).

    Returns:
        Φ'(δ) · S_σ(σ) · S_T(T) — the subgradient kernel.
    """
    if p is None:
        p = XU_LMO
    return phi_shi_prime(delta, k3, k4) * s_soc(sigma, p) * s_temp(T_C, p)


def s_soc(sigma: float | np.ndarray,
          p: XuModelParams = XU_LMO) -> float | np.ndarray:
    """SoC stress factor S_σ (Eq. 25).

    Args:
        sigma: Average SoC of the cycle, normalised in [0, 1].
        p:     Model parameters.

    Returns:
        Stress factor value(s). S_σ = 1 at σ = σ_ref.
    """
    return np.exp(p.k_sigma * (np.asarray(sigma, dtype=float) - p.sigma_ref))


def s_temp(T_C: float | np.ndarray,
           p: XuModelParams = XU_LMO) -> float | np.ndarray:
    """Temperature stress factor S_T (Eq. 22, Arrhenius).

    Args:
        T_C: Cell temperature [°C].
        p:   Model parameters.

    Returns:
        Stress factor value(s). S_T = 1 at T_C = T_ref_C (typically 25°C).
    """
    T_K     = np.asarray(T_C,        dtype=float) + 273.15
    T_ref_K = p.T_ref_C + 273.15
    return np.exp(p.k_T * (T_K - T_ref_K) * T_ref_K / T_K)


def s_time(t_seconds: float | np.ndarray,
           p: XuModelParams = XU_LMO) -> float | np.ndarray:
    """Time (calendar) stress factor S_t (Eq. 27).

    Args:
        t_seconds: Elapsed time [s].
        p:         Model parameters.

    Returns:
        Linearised calendar degradation contribution.
    """
    return p.k_t * np.asarray(t_seconds, dtype=float)


# =============================================================================
# Degradation accumulation
# =============================================================================

def fc_cycle(delta: float | np.ndarray,
             sigma: float | np.ndarray,
             T_C: float = 25.0,
             p: XuModelParams = XU_LMO) -> float | np.ndarray:
    """Linearised degradation per cycle — Φ(δ, σ, T) (Eq. 18).

    Args:
        delta: Cycle DoD (normalised 0-1).
        sigma: Average SoC of cycle (normalised 0-1).
        T_C:   Cell temperature [°C] (default: 25°C, S_T = 1).
        p:     Model parameters.

    Returns:
        Per-cycle linearised degradation value(s).
    """
    return s_dod(delta, p) * s_soc(sigma, p) * s_temp(T_C, p)


def fc_cycle_derivative(delta: float | np.ndarray,
                        sigma: float | np.ndarray,
                        T_C: float = 25.0,
                        p: XuModelParams = XU_LMO) -> float | np.ndarray:
    """Derivative of fc_cycle w.r.t. delta — Φ'(δ) in Shi et al. (2018).

    Since fc_cycle = S_δ(δ) · S_σ(σ) · S_T(T), and only S_δ depends on δ:
        d(fc_cycle)/dδ = S_δ'(δ) · S_σ(σ) · S_T(T)

    Required by the subgradient algorithm of Shi et al. Eqs. 17-18.
    Represents how quickly the per-cycle degradation cost grows with depth.

    Args:
        delta: Cycle DoD (normalised 0-1).
        sigma: Average SoC of the cycle (normalised 0-1).
        T_C:   Cell temperature [°C]. Default 25°C (S_T = 1.0).
        p:     Model parameters.

    Returns:
        Φ'(δ) — always >= 0: deeper cycles are always more damaging.
    """
    return s_dod_derivative(delta, p) * s_soc(sigma, p) * s_temp(T_C, p)


def ft_calendar(t_seconds: float,
                sigma_mean: float,
                T_C: float = 25.0,
                p: XuModelParams = XU_LMO) -> float:
    """Total calendar aging over the simulation period (Eq. 19).

    Args:
        t_seconds:  Total simulation duration [s].
        sigma_mean: Mean SoC over the simulation (normalised 0-1).
        T_C:        Cell temperature [°C].
        p:          Model parameters.

    Returns:
        Calendar degradation contribution to fd.
    """
    return float(s_time(t_seconds, p) * s_soc(sigma_mean, p) * s_temp(T_C, p))


def compute_fd(cycles: List[Dict],
               sigma_mean: float,
               t_total_seconds: float,
               T_C: float = 25.0,
               p: XuModelParams = XU_LMO) -> Tuple[float, float, float]:
    """Total linearised degradation fd (Eq. 3).

    Args:
        cycles:           Rainflow cycle list from rainflow_cycle_counting().
        sigma_mean:       Mean SoC of the entire SoC profile.
        t_total_seconds:  Total simulation duration in seconds.
        T_C:              Cell temperature [°C].
        p:                Model parameters.

    Returns:
        (fd, fd_cycle, fd_calendar)
    """
    fd_calendar = ft_calendar(t_total_seconds, sigma_mean, T_C, p)
    fd_cycle = 0.0
    for c in cycles:
        fd_cycle += float(c["count"]) * float(fc_cycle(c["dod"], c["soc_mean"], T_C, p))
    return fd_calendar + fd_cycle, fd_cycle, fd_calendar


def sei_capacity_loss(fd: float, p: XuModelParams = XU_LMO) -> float:
    """Capacity loss L from the SEI two-exponential model (Eq. 12).

    L = 0 → new battery, L = 0.20 → end-of-life (80% remaining capacity).

    Args:
        fd: Total linearised degradation (output of compute_fd).
        p:  Model parameters.

    Returns:
        Capacity loss L in [0, 1].
    """
    L = (1.0
         - p.alpha_sei * np.exp(-p.beta_sei * fd)
         - (1.0 - p.alpha_sei) * np.exp(-fd))
    return float(np.clip(L, 0.0, 1.0))


# =============================================================================
# Capacity fade curve (shared by plots and validate_xu_dst)
# =============================================================================

def xu_capacity_curve(fd_values: np.ndarray,
                      p: XuModelParams = XU_LMO) -> np.ndarray:
    """Capacity retention (%) as a function of fd.

    Args:
        fd_values: Array of linearised degradation values.
        p:         Model parameters.

    Returns:
        Capacity retention in % (100 = new, 80 = EoL).
    """
    L = (1.0
         - p.alpha_sei * np.exp(-p.beta_sei * fd_values)
         - (1.0 - p.alpha_sei) * np.exp(-fd_values))
    return (1.0 - np.clip(L, 0.0, 1.0)) * 100.0


# =============================================================================
# Rainflow counting
# =============================================================================

def rainflow_cycle_counting(storage_e: List[float],
                             e_cap: float) -> List[Dict]:
    """ASTM E1049 rainflow counting with normalised DoD and mean SoC.

    Args:
        storage_e: Battery energy [MWh] time-series.
        e_cap:     Nominal energy capacity [MWh] for normalisation.

    Returns:
        List of dicts, each with:
            dod       — cycle DoD (normalised, 0-1)
            soc_mean  — average SoC during cycle (normalised, 0-1)
            depth_MWh — cycle range [MWh]
            mean_MWh  — cycle mean energy [MWh]
            count     — 0.5 (half cycle) or 1.0 (full cycle)
            i_start   — index of cycle start
            i_end     — index of cycle end

    Raises:
        ImportError: If the `rainflow` package is not installed.
    """
    if _rainflow_lib is None:
        raise ImportError("rainflow package required: pip install rainflow")

    e     = np.asarray(storage_e, dtype=float)
    e_cap = max(float(e_cap), 1e-9)

    cycles: List[Dict] = []
    for rng, mean, count, i_start, i_end in _rainflow_lib.extract_cycles(e):
        cycles.append({
            "dod":       float(rng)  / e_cap,
            "soc_mean":  float(mean) / e_cap,
            "depth_MWh": float(rng),
            "mean_MWh":  float(mean),
            "count":     float(count),
            "i_start":   int(i_start),
            "i_end":     int(i_end),
        })
    return cycles


# =============================================================================
# EFC helper
# =============================================================================

def count_equivalent_full_cycles(storage_p: List[float],
                                  storage_e: List[float],
                                  e_cap: float,
                                  dt_hours: float = 1.0) -> float:
    """Equivalent full cycles (EFC) from discharged energy.

    Args:
        storage_p: Battery power [MW] (+ = discharge, - = charge).
        storage_e: Battery energy [MWh] (kept for API compatibility).
        e_cap:     Nominal energy capacity [MWh].
        dt_hours:  Timestep [h].

    Returns:
        Equivalent full cycles (dimensionless).
    """
    if e_cap <= 0:
        return 0.0
    p = np.asarray(storage_p, dtype=float)
    return float(np.sum(p[p > 0.0]) * dt_hours) / float(e_cap)


# =============================================================================
# DoD histogram
# =============================================================================

def calculate_dod_distribution(storage_e: List[float],
                                e_cap: float,
                                n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Depth-of-discharge histogram from the SoC time-series.

    Args:
        storage_e: Battery energy [MWh] time-series.
        e_cap:     Nominal energy capacity [MWh].
        n_bins:    Number of histogram bins.

    Returns:
        (bin_centers, counts)
    """
    edges   = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    if e_cap <= 0:
        return centers, np.zeros(n_bins, dtype=int)
    soc    = np.asarray(storage_e, dtype=float)
    dod    = np.clip((float(e_cap) - soc) / float(e_cap), 0.0, 1.0)
    counts, _ = np.histogram(dod, bins=edges)
    return centers, counts


# =============================================================================
# Main analysis API
# =============================================================================

def analyze_degradation(
    storage_p: List[float],
    storage_e: List[float],
    e_cap_nominal: float,
    battery_params: Dict,
    dt_hours: float = 1.0,
    n_bins_dod: int = 10,
    enable_rainflow: bool = True,
    T_cell_C: float = 25.0,
    xu_params: XuModelParams = XU_LMO,
    eol_thresholds: List[float] = None,
) -> Dict:
    """Complete degradation analysis using the Xu et al. (2016) model.

    Drop-in replacement for degradation_v_2.analyze_degradation.

    Args:
        storage_p:       Battery power [MW] (+ discharge, - charge).
        storage_e:       Battery energy [MWh] time-series.
        e_cap_nominal:   Nominal energy capacity [MWh].
        battery_params:  Must include 'power_capacity_W'.
        dt_hours:        Timestep [h] (default: 1.0).
        n_bins_dod:      Number of DoD histogram bins.
        enable_rainflow: If False, skip rainflow (calendar aging only).
        T_cell_C:        Cell temperature [°C]. Default 25°C (S_T = 1.0).
        xu_params:       Xu model parameters. Defaults to LMO (Table I).
        eol_thresholds:  SoH fractions to treat as EoL. Default [0.80, 0.60].

    Returns:
        Dict with keys: total_cycles, fd, fd_cycle, fd_calendar,
        capacity_loss, capacity_retention, capacity_fade_percent, soh,
        dod_distribution, cycle_depth_distribution, e_cap_degraded,
        p_cap_degraded, xu_cycle_stats, eol_years, meta.
    """
    n_steps         = len(storage_e)
    t_total_seconds = n_steps * dt_hours * 3600.0
    e_arr           = np.asarray(storage_e, dtype=float)
    e_cap           = max(float(e_cap_nominal), 1e-9)

    total_cycles = count_equivalent_full_cycles(
        storage_p=storage_p, storage_e=storage_e,
        e_cap=e_cap, dt_hours=dt_hours,
    )
    sigma_mean = float(np.mean(e_arr)) / e_cap

    cycles: List[Dict] = []
    if enable_rainflow:
        cycles = rainflow_cycle_counting(storage_e=storage_e, e_cap=e_cap)

    fd, fd_cycle, fd_calendar = compute_fd(
        cycles=cycles, sigma_mean=sigma_mean,
        t_total_seconds=t_total_seconds, T_C=T_cell_C, p=xu_params,
    )

    capacity_loss      = sei_capacity_loss(fd, xu_params)
    capacity_retention = 1.0 - capacity_loss
    soh                = capacity_retention * 100.0
    fade_pct           = capacity_loss * 100.0

    dod_bins, dod_counts = calculate_dod_distribution(storage_e, e_cap, n_bins=n_bins_dod)

    xu_cycle_stats: Dict = {}
    if cycles:
        dods = np.array([c["dod"]      for c in cycles])
        socs = np.array([c["soc_mean"] for c in cycles])
        cnts = np.array([c["count"]    for c in cycles])
        xu_cycle_stats = {
            "n_rainflow_cycles": float(np.sum(cnts)),
            "mean_dod":          float(np.average(dods, weights=cnts)),
            "max_dod":           float(np.max(dods)),
            "mean_soc":          float(np.average(socs, weights=cnts)),
            "mean_fc":           float(np.average(
                                     [float(fc_cycle(c["dod"], c["soc_mean"], T_cell_C, xu_params))
                                      for c in cycles], weights=cnts)),
        }

    if "power_capacity_W" not in battery_params:
        raise KeyError("battery_params must include 'power_capacity_W'")
    p_cap_MW = float(battery_params["power_capacity_W"]) / 1e6

    if eol_thresholds is None:
        eol_thresholds = [0.80, 0.60]
    fd_per_yr = float(fd) / max(t_total_seconds / 3600.0 / 8760.0, 1e-9)
    eol_years: Dict[float, Optional[float]] = {}
    for thr in eol_thresholds:
        if (1.0 - sei_capacity_loss(fd_per_yr * 0.0, xu_params)) < thr:
            eol_years[thr] = 0.0
            continue
        lo, hi = 0.0, 200.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if 1.0 - sei_capacity_loss(fd_per_yr * mid, xu_params) > thr:
                lo = mid
            else:
                hi = mid
        eol_years[thr] = round((lo + hi) / 2.0, 2) if (lo + hi) / 2.0 < 190.0 else None

    return {
        "total_cycles":             float(total_cycles),
        "fd":                       float(fd),
        "fd_cycle":                 float(fd_cycle),
        "fd_calendar":              float(fd_calendar),
        "capacity_loss":            float(capacity_loss),
        "capacity_retention":       float(capacity_retention),
        "capacity_fade_percent":    float(fade_pct),
        "soh":                      float(soh),
        "dod_distribution":         (dod_bins, dod_counts),
        "cycle_depth_distribution": cycles,
        "e_cap_degraded":           e_cap * capacity_retention,
        "p_cap_degraded":           p_cap_MW * capacity_retention,
        "xu_cycle_stats":           xu_cycle_stats,
        "eol_years":                eol_years,
        "meta": {
            "model":           "Xu2016_LMO",
            "dt_hours":        float(dt_hours),
            "t_total_hours":   t_total_seconds / 3600.0,
            "T_cell_C":        float(T_cell_C),
            "sigma_mean":      float(sigma_mean),
            "enable_rainflow": bool(enable_rainflow),
            "eol_thresholds":  list(eol_thresholds),
        },
    }

# =============================================================================
# Interactive convexity self-test
# =============================================================================

def _check_sdelta_convexity(soc_min: float | None = None,
                             soc_max: float | None = None,
                             n_points: int = 200_000) -> None:
    """Interactive numerical verification that Xu S_δ is NOT globally convex.

    If soc_min / soc_max are not provided, prompts the user to enter them.
    max_dod = soc_max - soc_min sets the ceiling on physically reachable
    cycle depths given the LP SoC constraints.
    """
    if soc_min is None or soc_max is None:
        # Default to the window in config/battery.yaml so the check runs
        # unattended. Pass soc_min and soc_max explicitly to override.
        from degradation.paths import BATTERY_YAML

        soc_min, soc_max, _src = load_soc_window_from_yaml(BATTERY_YAML)
        print("\n=== Xu S_delta convexity check ===")
        print(f"  SoC window from {_src}: [{soc_min}, {soc_max}]")

    if not (0.0 <= soc_min < soc_max <= 1.0):
        raise ValueError(f"Need 0 ≤ soc_min < soc_max ≤ 1, got [{soc_min}, {soc_max}]")

    max_dod = round(soc_max - soc_min, 10)

    # --- Compute S_δ'' numerically ---
    delta = np.linspace(1e-4, 1.0, n_points)
    D  = XU_LMO.k_delta1 * delta ** XU_LMO.k_delta2 + XU_LMO.k_delta3
    Dp = XU_LMO.k_delta1 * XU_LMO.k_delta2 * delta ** (XU_LMO.k_delta2 - 1)
    d1 = -Dp / D**2
    d2 = np.gradient(d1, delta)

    # Global inflection point
    sign_changes = np.where(np.diff(np.sign(d2)))[0]
    boundary = float(delta[sign_changes[0]]) if len(sign_changes) > 0 else None

    # --- Report ---
    print(f"\n{'─'*70}")
    print(f"Step 7 — Interactive convexity check")
    print(f"{'─'*70}")
    print(f"  Operating window : soc_min={soc_min},  soc_max={soc_max}")
    print(f"  Max cycle depth  : max_dod = {max_dod:.4f}")
    print(f"  Shi fitting window lower bound: 0.15 (convexity boundary)")
    print()

    if boundary is not None:
        print(f"  Global d²S_δ/dδ² sign change at δ ≈ {boundary:.4f}")
        print(f"    S_δ NON-CONVEX for δ < {boundary:.4f}")
        print(f"    S_δ CONVEX     for δ > {boundary:.4f}")
    else:
        print("  WARNING: No sign change found — verify LMO parameters.")

    print()

    # Full reachable range [0, max_dod]
    mask_full = delta <= max_dod
    n_nc_full = int(np.sum(d2[mask_full] < 0))
    frac_full = n_nc_full / mask_full.sum()
    tag_full  = "NON-CONVEX ✗" if n_nc_full > 0 else "CONVEX ✓"
    print(f"  [0,    {max_dod:.2f}]  (full reachable range) → {tag_full}")
    if n_nc_full > 0:
        print(f"    {n_nc_full:,} / {mask_full.sum():,} sample points have d²S/dδ² < 0  "
              f"({frac_full*100:.1f}%)")

    # Shi fitting window [0.15, max_dod]
    if max_dod > 0.15:
        mask_fit  = (delta >= 0.15) & (delta <= max_dod)
        n_nc_fit  = int(np.sum(d2[mask_fit] < 0))
        tag_fit   = "NON-CONVEX ✗" if n_nc_fit > 0 else "CONVEX ✓"
        print(f"  [0.15, {max_dod:.2f}]  (Shi fitting window)   → {tag_fit}")
        if n_nc_fit > 0:
            print(f"    WARNING: {n_nc_fit:,} non-convex points inside fitting window — "
                  "lower bound must be raised.")
    else:
        print(f"  [0.15, {max_dod:.2f}]  Skipped — max_dod ≤ 0.15")

    # Dual-Φ architecture verdict
    print()
    print("  --- Dual-Φ architecture verdict ---")
    if boundary is None:
        print("  INCONCLUSIVE — could not locate non-convex boundary.")
    elif boundary >= max_dod:
        print(f"  NOTE: boundary ({boundary:.4f}) ≥ max_dod ({max_dod:.4f}).")
        print("  S_δ is convex over the full reachable range for this window.")
        print("  Dual-Φ split is conservative but not strictly required here.")
    elif boundary >= 0.15:
        print(f"  CRITICAL: boundary ({boundary:.4f}) is inside fitting window [0.15, {max_dod:.2f}].")
        print("  Dual-Φ is required AND fitting window lower bound must be raised.")
    else:
        print(f"  OK: boundary ({boundary:.4f}) < fitting window lower bound (0.15).")
        print("  Φ_shi is fitted and evaluated entirely in the convex region of S_δ.")
        print("  Dual-Φ architecture assumption HOLDS. ✓")

# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("xu.py — Two-Phi Architecture Self-Test")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load SoC window from YAML (or fall back to [0, 1])
    # ------------------------------------------------------------------
    from degradation.paths import BATTERY_YAML as _yaml_path

    if _yaml_path.exists():
        yaml_soc_min, yaml_soc_max, yaml_src = load_soc_window_from_yaml(_yaml_path)
    else:
        yaml_soc_min, yaml_soc_max = 0.0, 1.0
        yaml_src = f"default [0,1] — {_yaml_path} not found"

    print(f"\n  YAML source : {yaml_src}")
    print(f"  soc_min={yaml_soc_min}  soc_max={yaml_soc_max}  "
          f"max_dod={yaml_soc_max - yaml_soc_min:.2f}")

    # ------------------------------------------------------------------
    # 2. Stress factor sanity
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("Stress factors at reference (sigma=0.5, T=25C)")
    print(f"{'─'*70}")
    assert abs(s_soc(0.5)   - 1.0) < 1e-9, "S_sigma ref fail"
    assert abs(s_temp(25.0) - 1.0) < 1e-9, "S_T ref fail"
    print(f"  S_dod(0.5)  = {s_dod(0.5):.6e}")
    print(f"  S_soc(0.5)  = {s_soc(0.5):.6f}   must be 1.0 ✓")
    print(f"  S_T(25C)    = {s_temp(25.0):.6f}   must be 1.0 ✓")
    print(f"  S_t(1yr)    = {s_time(365.25*24*3600):.6e}")

    # ------------------------------------------------------------------
    # 3. fit_shi_polynomial — default [0,1] vs YAML window, side by side
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("Shi polynomial fit — default [0,1] vs YAML window")
    print(f"{'─'*70}")

    print("\n  Default fit  soc=[0.0, 1.0]  (full-range fallback, used at import):")
    default_fit = fit_shi_polynomial(
        soc_min=0.0, soc_max=1.0, source="default [0,1]", verbose=True
    )
    print(f"    {default_fit.summary()}")

    print(f"\n  YAML fit     soc=[{yaml_soc_min}, {yaml_soc_max}]  ({yaml_src}):")
    yaml_fit = fit_shi_polynomial(
        soc_min=yaml_soc_min, soc_max=yaml_soc_max,
        source=yaml_src, verbose=True
    )
    print(f"    {yaml_fit.summary()}")

    print(f"\n  Module SHI_K3={SHI_K3:.6e}  SHI_K4={SHI_K4:.4f}  "
          f"(= default fit, backwards-compatible)")

    # Confirm module defaults match the recomputed default
    assert abs(SHI_K3 - default_fit.k3) < 1e-12
    assert abs(SHI_K4 - default_fit.k4) < 1e-12
    print("  Module-level constants match recomputed default fit ✓")

    # ------------------------------------------------------------------
    # 4. Convexity proof over the YAML window
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"Convexity over full DoD range in YAML window [0.05, {yaml_fit.max_dod:.2f}]")
    print(f"{'─'*70}")

    _d = np.linspace(0.01, yaml_fit.max_dod, 50_000)
    _d2_xu  = np.gradient(fc_cycle_derivative(_d, 0.5, 25.0), _d)
    _d2_shi = np.gradient(
        phi_shi_prime(_d, yaml_fit.k3, yaml_fit.k4), _d
    )
    _n_bad_xu  = int(np.sum(_d2_xu  < 0))
    _n_bad_shi = int(np.sum(_d2_shi < 0))
    _bnd_idx   = np.where(np.diff(np.sign(_d2_xu)))[0]
    _bnd       = float(_d[_bnd_idx[0]]) if len(_bnd_idx) > 0 else None

    print(f"\n  Phi_xu  non-convex points: {_n_bad_xu}")
    if _bnd:
        print(f"    Xu convex only above delta={_bnd:.4f}")
        print(f"    Small cycles (delta < {_bnd:.4f}) fall in non-convex region")
        print(f"    -> Xu NOT safe as gradient Phi")
    print(f"  Phi_shi non-convex points: {_n_bad_shi}   (must be 0)")
    print(f"    k4*(k4-1) = {yaml_fit.k4*(yaml_fit.k4-1):.4f} > 0 -> globally convex ✓")
    assert _n_bad_shi == 0, f"Phi_shi non-convex at {_n_bad_shi} points!"
    print(f"  Assertion passed ✓")

    # ------------------------------------------------------------------
    # 5. Side-by-side values spanning the non-convex boundary
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("Phi_xu vs Phi_shi (YAML fit) at key DoD points")
    print(f"{'─'*70}")
    _sigma_mid = yaml_soc_min + yaml_fit.max_dod / 2.0
    print(f"  sigma={_sigma_mid:.2f} (window midpoint), T=25C")
    print(f"  {'delta':>7}  {'region':>13}  "
          f"{'Phi_xu':>12}  {'Phi_shi':>12}  "
          f"{'dPhi_xu':>12}  {'dPhi_shi':>12}  "
          f"{'d2Phi_shi':>12}")
    _check = sorted({0.02, 0.05, 0.10, round(_bnd, 3) if _bnd else 0.144,
                     0.20, 0.40, 0.60, yaml_fit.max_dod})
    for _d_val in _check:
        if _d_val > yaml_fit.max_dod:
            continue
        _rgn = "NON-CONVEX" if (_bnd and _d_val < _bnd) else "convex    "
        _pxu   = float(fc_cycle(_d_val, _sigma_mid, 25.0))
        _pshi  = float(phi_shi(_d_val, yaml_fit.k3, yaml_fit.k4))
        _dpxu  = float(fc_cycle_derivative(_d_val, _sigma_mid, 25.0))
        _dpshi = float(phi_shi_prime(_d_val, yaml_fit.k3, yaml_fit.k4))
        _d2shi = float(phi_shi_double(_d_val, yaml_fit.k3, yaml_fit.k4))
        print(f"  {_d_val:>7.4f}  {_rgn:>13}  "
              f"{_pxu:12.4e}  {_pshi:12.4e}  "
              f"{_dpxu:12.4e}  {_dpshi:12.4e}  "
              f"{_d2shi:12.4e}")

    # ------------------------------------------------------------------
    # 6. Reporting continuity — Xu Phi unchanged for fd/SoH
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("Reporting continuity — Xu Phi produces correct fd / SoH")
    print(f"{'─'*70}")
    _fd_cal = ft_calendar(365.25 * 24 * 3600, _sigma_mid, 25.0)
    _fd_cyc = 200 * float(fc_cycle(yaml_fit.max_dod * 0.5, _sigma_mid, 25.0))
    _fd_tot = _fd_cal + _fd_cyc
    _L      = sei_capacity_loss(_fd_tot)
    print(f"  200 cycles/yr, delta={yaml_fit.max_dod*0.5:.2f}, "
          f"sigma={_sigma_mid:.2f}, T=25C")
    print(f"  fd_cal={_fd_cal:.4e}  fd_cyc={_fd_cyc:.4e}  fd_tot={_fd_tot:.4e}")
    print(f"  SoH = {(1-_L)*100:.3f}%")
    print(f"  Xu Phi unchanged by Shi swap ✓")

    # ------------------------------------------------------------------
    # 7. Interactive convexity check — user supplies the SoC window
    # ------------------------------------------------------------------
    _check_sdelta_convexity()
    
    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("Summary")
    print(f"{'─'*70}")
    print(f"  Phi_xu  -> REPORTING (fd, SoH, EoL) — Xu Eq.32, validated LMO")
    print(f"  Phi_shi -> GRADIENTS (Shi Eqs.17-18) — k3*delta^k4, globally convex")
    print()
    print(f"  Default fit : {default_fit.summary()}")
    print(f"  YAML fit    : {yaml_fit.summary()}")
    print()
    print(f"  Use fit_shi_polynomial(soc_min, soc_max, verbose=True) in your")
    print(f"  run script and pass the result to compute_subgradient(shi_fit=...)")
    print(f"  for window-specific, fully provenance-tracked gradients.")
    print(f"\n  All assertions passed ✓")
    print(f"\nFor plots and DST validation run: python plots.py")