"""
WP2 Battery Degradation — Pure Shi (2018) Model  [Self-Contained]
==================================================================

Branch: feature/pure-shi-degradation

This file is intentionally self-contained.  It does NOT import from
xu.py so the Shi branch can be developed and run independently.
All utilities previously shared with xu (stress helpers, rainflow
counting, Shi polynomial fitting, subgradient engine) are included here.

Model summary
-------------
Accumulation (cycle-only, no calendar term):
    fd_shi = Σ_i  count_i · Φ(δ_i) · S_σ(σ_i) · S_T(T)
    Φ(δ)   = k3 · δ^k4          (Shi 2018 convex polynomial, k4 > 1)
    S_σ(σ) = exp(k_σ·(σ − σ_ref))   (Xu Eq. 25 form — kept for consistency)
    S_T(T) = exp(k_T·(T−T_ref)·T_ref/T)  (Xu Eq. 22 form — kept for consistency)

Capacity fade — Option B (Xu SEI two-exponential fed with fd_shi):
    L = 1 − α_sei·exp(−β_sei·fd_shi) − (1−α_sei)·exp(−fd_shi)
    capacity_retention = 1 − L

Gradient computation — Shi et al. (2018) Eqs. 17-18:
    Φ'(δ) = k3·k4·δ^(k4-1)  — globally convex (k4 > 1)
    ∂f/∂c_t = Φ'(v_i)·(B·τ·η_in)/2      charging timesteps
    ∂f/∂d_t = Φ'(w_j)·(B·τ)/(2·η_out)   discharging timesteps

Note on calendar aging
----------------------
Shi (2018) is cycle-only.  fd_calendar = 0.0 in all result dicts.
For WP2's continuous-operation scenario this means ~15-25% of total
Xu fd is structurally absent — flag this in the thesis comparison.

Note on Xu reference
--------------------
The Xu LMO S_δ(δ) is used ONLY when fitting Φ(δ) = k3·δ^k4 to a
reference curve.  It is NOT called anywhere in the reporting path.
The _XuRefParams dataclass here holds those coefficients privately.

Public API
----------
    ShiPolynomialFit                     dataclass — fit result + provenance
    ShiModelParams                       dataclass — all model coefficients
    load_soc_window_from_yaml(path)      read soc_min/soc_max from YAML
    fit_shi_polynomial(soc_min, soc_max) fit Φ to Xu S_δ reference
    s_soc(sigma, ...)                    SoC stress factor
    s_temp(T_C, ...)                     temperature stress factor
    rainflow_cycle_counting(...)         ASTM E1049 rainflow counting
    count_equivalent_full_cycles(...)    EFC from discharged energy
    calculate_dod_distribution(...)      DoD histogram
    phi_shi_cycle(delta, sigma, T_C, p)  per-cycle Shi cost Φ·S_σ·S_T
    compute_fd_shi(cycles, p, T_C)       accumulated fd_shi
    shi_capacity_loss(fd_shi, p)         SEI capacity loss (Option B)
    shi_capacity_curve(fd_values, p)     retention % curve for plotting
    build_half_cycle_map(cycles, e)      timestep → half-cycle attribution
    compute_subgradient(...)             Shi Eqs. 17-18 per-timestep gradient
    analyze_degradation_shi(...)         main analysis API (drop-in compatible)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rainflow as _rainflow_lib   # type: ignore
except ImportError:
    _rainflow_lib = None


# =============================================================================
# Xu LMO reference parameters — private, used only for Φ fitting
# =============================================================================

@dataclass
class _XuRefParams:
    """Xu LMO Table I coefficients needed only for fitting Φ to S_δ."""
    k_delta1:  float = 1.40e5
    k_delta2:  float = -5.01e-1
    k_delta3:  float = -1.23e5
    k_sigma:   float = 1.04
    sigma_ref: float = 0.50
    k_T:       float = 6.93e-2
    T_ref_C:   float = 25.0


_XU_REF = _XuRefParams()


def _s_dod_xu(delta: float | np.ndarray) -> float | np.ndarray:
    """Xu LMO DoD stress S_δ — used only for fitting Φ, not for reporting."""
    delta = np.clip(np.asarray(delta, dtype=float), 1e-6, 1.0)
    denom = _XU_REF.k_delta1 * delta ** _XU_REF.k_delta2 + _XU_REF.k_delta3
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    return 1.0 / denom


# =============================================================================
# Stress helpers — S_σ and S_T (same functional form as Xu Eqs. 25, 22)
# =============================================================================

def s_soc(
    sigma: float | np.ndarray,
    k_sigma:   float = _XU_REF.k_sigma,
    sigma_ref: float = _XU_REF.sigma_ref,
) -> float | np.ndarray:
    """SoC stress factor S_σ = exp(k_σ·(σ − σ_ref)).  S_σ = 1 at σ = σ_ref."""
    return np.exp(k_sigma * (np.asarray(sigma, dtype=float) - sigma_ref))


def s_temp(
    T_C: float | np.ndarray,
    k_T:    float = _XU_REF.k_T,
    T_ref_C: float = _XU_REF.T_ref_C,
) -> float | np.ndarray:
    """Temperature stress factor S_T (Arrhenius).  S_T = 1 at T_C = T_ref_C."""
    T_K     = np.asarray(T_C, dtype=float) + 273.15
    T_ref_K = T_ref_C + 273.15
    return np.exp(k_T * (T_K - T_ref_K) * T_ref_K / T_K)


# =============================================================================
# Shi polynomial Φ(δ) = k3·δ^k4  and  its derivative Φ'(δ)
# =============================================================================

@dataclass
class ShiPolynomialFit:
    """Result of fitting Φ(δ) = k3·δ^k4 to the Xu S_δ reference curve.

    k4 > 1 guarantees global convexity (Φ''(δ) > 0 everywhere),
    satisfying Shi Theorem 1 for all cycle depths including small
    arbitrage cycles that SoC bounds cannot exclude.
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
        return self.soc_max - self.soc_min

    @property
    def is_convex(self) -> bool:
        return self.k4 > 1.0

    def summary(self) -> str:
        return (
            f"k3={self.k3:.4e}  k4={self.k4:.4f}  R²={self.r2:.4f}  "
            f"fit=[{self.fit_lo:.2f},{self.fit_hi:.2f}]  "
            f"soc=[{self.soc_min},{self.soc_max}]  "
            f"convex={'YES' if self.is_convex else 'NO'}  "
            f"source={self.source}"
        )


def load_soc_window_from_yaml(yaml_path: "str | Path") -> Tuple[float, float, str]:
    """Read soc_min and soc_max from a battery YAML file.

    Tries PyYAML, then a minimal line parser, then falls back to (0.0, 1.0).
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return 0.0, 1.0, f"default [0,1] — {yaml_path.name} not found"
    try:
        import yaml as _yaml
        with open(yaml_path) as f:
            cfg = _yaml.safe_load(f)
        lim = cfg.get("operating_limits", {})
        return (float(lim.get("soc_min", 0.0)),
                float(lim.get("soc_max", 1.0)),
                f"{yaml_path.name} via PyYAML")
    except ImportError:
        pass
    except Exception as exc:
        return 0.0, 1.0, f"default [0,1] — YAML parse error: {exc}"
    try:
        in_limits = False
        lo, hi = 0.0, 1.0
        with open(yaml_path) as f:
            for raw in f:
                line = raw.strip()
                if line.startswith("operating_limits"):
                    in_limits = True; continue
                if in_limits:
                    if line and not line.startswith((" ", "\t")):
                        break
                    if "soc_min" in line:
                        lo = float(line.split(":")[1].strip().split()[0])
                    elif "soc_max" in line:
                        hi = float(line.split(":")[1].strip().split()[0])
        return lo, hi, f"{yaml_path.name} via line parser"
    except Exception as exc:
        return 0.0, 1.0, f"default [0,1] — parser error: {exc}"


def fit_shi_polynomial(
    soc_min:  float = 0.0,
    soc_max:  float = 1.0,
    n_points: int   = 500,
    source:   str   = "manual",
    verbose:  bool  = False,
) -> ShiPolynomialFit:
    """Fit Φ(δ) = k3·δ^k4 to the Xu LMO S_δ reference curve.

    Fitting range:
        fit_lo = 0.15  (just above Xu non-convex boundary ~0.1437)
        fit_hi = soc_max − soc_min  (max cycle depth in the SoC window)

    Log-log OLS gives equal relative weight across all DoD levels.
    k4 > 1 is verified; ValueError raised if fit is not convex.

    Args:
        soc_min:  Battery SoC lower limit (from YAML operating_limits).
        soc_max:  Battery SoC upper limit.
        n_points: Linear DoD points for fitting.
        source:   Provenance string stored in ShiPolynomialFit.
        verbose:  Print fit diagnostics.

    Returns:
        ShiPolynomialFit with k3, k4, R², and window metadata.
    """
    max_dod = float(soc_max) - float(soc_min)
    if max_dod < 0.15:
        raise ValueError(
            f"SoC window [{soc_min}, {soc_max}] too narrow "
            f"(max_dod={max_dod:.3f} < 0.15)."
        )
    fit_lo = 0.15
    fit_hi = min(max_dod, 1.0)
    if fit_hi <= fit_lo:
        raise ValueError(
            f"SoC window [{soc_min}, {soc_max}] → fit_hi={fit_hi} ≤ fit_lo={fit_lo}."
        )

    d_fit  = np.linspace(fit_lo, fit_hi, n_points)
    phi_xu = _s_dod_xu(d_fit)

    coeffs = np.polyfit(np.log(d_fit), np.log(phi_xu), 1)
    k4     = float(coeffs[0])
    k3     = float(np.exp(coeffs[1]))

    phi_fit = k3 * d_fit ** k4
    ss_res  = float(np.sum((phi_xu - phi_fit) ** 2))
    ss_tot  = float(np.sum((phi_xu - phi_xu.mean()) ** 2))
    r2      = 1.0 - ss_res / ss_tot

    if k4 <= 1.0:
        raise ValueError(
            f"Fitted k4={k4:.4f} ≤ 1 — Shi polynomial not globally convex."
        )

    fit = ShiPolynomialFit(
        k3=k3, k4=k4, r2=r2,
        fit_lo=fit_lo, fit_hi=fit_hi,
        soc_min=float(soc_min), soc_max=float(soc_max),
        source=source,
    )
    if verbose:
        print(f"  fit_shi_polynomial  [{soc_min}, {soc_max}]  (source: {source})")
        print(f"    max_dod={max_dod:.2f}  fit=[{fit_lo:.2f},{fit_hi:.2f}]  "
              f"{n_points} pts  log-log OLS")
        print(f"    k3={k3:.6e}  k4={k4:.4f}  R²={r2:.4f}  "
              f"{'globally convex ✓' if k4 > 1 else 'NOT convex ✗'}")
    return fit


# Module-level default (full [0,1] range) — available at import time
_DEFAULT_SHI_FIT: ShiPolynomialFit = fit_shi_polynomial(
    soc_min=0.0, soc_max=1.0, source="default [0,1]", verbose=False
)


def phi_shi(delta: float | np.ndarray, k3: float, k4: float) -> float | np.ndarray:
    """Φ(δ) = k3·δ^k4.  k4 > 1 → globally convex."""
    return k3 * np.clip(np.asarray(delta, dtype=float), 1e-9, 1.0) ** k4


def phi_shi_prime(delta: float | np.ndarray, k3: float, k4: float) -> float | np.ndarray:
    """Φ'(δ) = k3·k4·δ^(k4-1).  Monotonically increasing for k4 > 1."""
    return k3 * k4 * np.clip(np.asarray(delta, dtype=float), 1e-9, 1.0) ** (k4 - 1.0)


def phi_shi_prime_with_stress(
    delta: float | np.ndarray,
    sigma: float | np.ndarray,
    T_C:  float = 25.0,
    k3:   float = None,
    k4:   float = None,
) -> float | np.ndarray:
    """Φ'(δ)·S_σ(σ)·S_T(T) — gradient kernel for Shi Eqs. 17-18."""
    if k3 is None or k4 is None:
        raise ValueError("k3 and k4 must be provided.")
    return phi_shi_prime(delta, k3, k4) * s_soc(sigma) * s_temp(T_C)


# =============================================================================
# Rainflow counting and EFC utilities
# =============================================================================

def rainflow_cycle_counting(
    storage_e: List[float],
    e_cap: float,
) -> List[Dict]:
    """ASTM E1049 rainflow counting with normalised DoD and mean SoC.

    Returns:
        List of dicts: dod, soc_mean, depth_MWh, mean_MWh, count, i_start, i_end.
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


def count_equivalent_full_cycles(
    storage_p: List[float],
    storage_e: List[float],
    e_cap: float,
    dt_hours: float = 1.0,
) -> float:
    """Equivalent full cycles (EFC) from discharged energy."""
    if e_cap <= 0:
        return 0.0
    p = np.asarray(storage_p, dtype=float)
    return float(np.sum(p[p > 0.0]) * dt_hours) / float(e_cap)


def calculate_dod_distribution(
    storage_e: List[float],
    e_cap: float,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Depth-of-discharge histogram from the SoC time-series."""
    edges   = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    if e_cap <= 0:
        return centers, np.zeros(n_bins, dtype=int)
    dod    = np.clip((float(e_cap) - np.asarray(storage_e, dtype=float)) / float(e_cap), 0.0, 1.0)
    counts, _ = np.histogram(dod, bins=edges)
    return centers, counts


# =============================================================================
# ShiModelParams
# =============================================================================

@dataclass
class ShiModelParams:
    """All coefficients for the pure Shi degradation model (Option B).

    k3, k4 define Φ(δ) = k3·δ^k4 — always set via from_fit(), never hardcoded.
    SEI fade coefficients are Xu LMO Table I defaults, held constant so that
    only the accumulation rule changes between branches (controlled comparison).
    """
    k3:        float = field(default=None)
    k4:        float = field(default=None)
    alpha_sei: float = 5.75e-2
    beta_sei:  float = 121.0
    k_sigma:   float = _XU_REF.k_sigma
    sigma_ref: float = _XU_REF.sigma_ref
    k_T:       float = _XU_REF.k_T
    T_ref_C:   float = _XU_REF.T_ref_C

    @classmethod
    def from_fit(cls, shi_fit: ShiPolynomialFit) -> "ShiModelParams":
        """Construct ShiModelParams with k3/k4 from a ShiPolynomialFit.

        Recommended usage:
            soc_min, soc_max, _ = load_soc_window_from_yaml(yaml_path)
            shi_fit    = fit_shi_polynomial(soc_min, soc_max, verbose=True)
            shi_params = ShiModelParams.from_fit(shi_fit)
        """
        if not shi_fit.is_convex:
            raise ValueError(
                f"ShiPolynomialFit k4={shi_fit.k4:.4f} ≤ 1 — not globally convex."
            )
        return cls(k3=shi_fit.k3, k4=shi_fit.k4)


# =============================================================================
# Shi per-cycle cost and degradation accumulation
# =============================================================================

def phi_shi_cycle(
    delta: float | np.ndarray,
    sigma: float | np.ndarray,
    T_C:  float = 25.0,
    p:    ShiModelParams = None,
) -> float | np.ndarray:
    """Per-cycle Shi degradation cost: Φ(δ)·S_σ(σ)·S_T(T)."""
    delta = np.clip(np.asarray(delta, dtype=float), 1e-9, 1.0)
    return (p.k3 * delta ** p.k4
            * s_soc(sigma, p.k_sigma, p.sigma_ref)
            * s_temp(T_C, p.k_T, p.T_ref_C))


def compute_fd_shi(
    cycles: List[Dict],
    p:      ShiModelParams,
    T_C:    float = 25.0,
) -> Tuple[float, float]:
    """Accumulated Shi degradation — no calendar term.

    fd_shi = Σ_i  count_i · Φ_shi(δ_i, σ_i, T)

    Returns:
        (fd_shi, fd_shi_cycle)
        fd_calendar is always 0.0 — returned separately for API symmetry.
    """
    if p.k3 is None or p.k4 is None:
        raise ValueError("ShiModelParams.k3/k4 are None — use ShiModelParams.from_fit().")
    fd = sum(
        float(c["count"]) * float(phi_shi_cycle(c["dod"], c["soc_mean"], T_C, p))
        for c in cycles
    )
    return float(fd), float(fd)


# =============================================================================
# Capacity fade — Xu SEI two-exponential fed with fd_shi  (Option B)
# =============================================================================

def shi_capacity_loss(fd_shi: float, p: ShiModelParams) -> float:
    """Capacity loss L from the Xu SEI two-exponential fed with fd_shi.

    L = 1 − α_sei·exp(−β_sei·fd_shi) − (1−α_sei)·exp(−fd_shi)

    Option B: same functional form as Xu Eq. 12 — only fd source changes.
    """
    L = (1.0
         - p.alpha_sei * np.exp(-p.beta_sei * float(fd_shi))
         - (1.0 - p.alpha_sei) * np.exp(-float(fd_shi)))
    return float(np.clip(L, 0.0, 1.0))


def shi_capacity_curve(fd_values: np.ndarray, p: ShiModelParams) -> np.ndarray:
    """Capacity retention (%) as a function of fd_shi — for plotting."""
    L = (1.0
         - p.alpha_sei * np.exp(-p.beta_sei * fd_values)
         - (1.0 - p.alpha_sei) * np.exp(-fd_values))
    return (1.0 - np.clip(L, 0.0, 1.0)) * 100.0


# =============================================================================
# Subgradient engine — Shi et al. (2018) Eqs. 17-18
# =============================================================================

def build_half_cycle_map(
    cycles:    List[Dict],
    storage_e: List[float],
) -> Dict[str, np.ndarray]:
    """Map every timestep to its owning rainflow half-cycle.

    Iterates over cycles (~200), not timesteps (~8760), using numpy
    slice assignment.  Deeper cycles overwrite shallower at junctions
    (Shi footnote on subgradient non-uniqueness).

    Returns:
        Dict with arrays of shape (n,):
            cycle_owner  int   — index into cycles list (-1 = unattributed)
            dod_at_t     float — DoD of owning cycle
            soc_at_t     float — mean SoC of owning cycle
            direction    int   — +1 DHC (discharge), -1 CHC (charge), 0 flat
            is_junction  bool  — overwritten by a deeper cycle
    """
    e = np.asarray(storage_e, dtype=float)
    n = len(e)

    cycle_owner = np.full(n, -1, dtype=np.int32)
    dod_at_t    = np.zeros(n, dtype=np.float64)
    soc_at_t    = np.zeros(n, dtype=np.float64)
    is_junction = np.zeros(n, dtype=bool)

    for ci, c in sorted(enumerate(cycles), key=lambda x: x[1]["dod"]):
        i0 = max(int(min(c["i_start"], c["i_end"])), 0)
        i1 = min(int(max(c["i_start"], c["i_end"])), n - 1)
        if i0 > i1:
            continue
        already              = cycle_owner[i0:i1+1] != -1
        is_junction[i0:i1+1] = np.where(already, True, is_junction[i0:i1+1])
        cycle_owner[i0:i1+1] = ci
        dod_at_t[i0:i1+1]    = c["dod"]
        soc_at_t[i0:i1+1]    = c["soc_mean"]

    delta_e   = np.diff(e, append=e[-1])
    direction = np.sign(-delta_e).astype(np.int8)
    direction = np.where(cycle_owner == -1, np.int8(0), direction)

    return {
        "cycle_owner": cycle_owner,
        "dod_at_t":    dod_at_t,
        "soc_at_t":    soc_at_t,
        "direction":   direction,
        "is_junction": is_junction,
    }


def compute_subgradient(
    storage_e: List[float],
    cycles:    List[Dict],
    dt_hours:  float,
    battery_replacement_cost_per_MWh: float,
    eff_in:   float = 1.0,
    eff_out:  float = 0.85,
    T_C:      float = 25.0,
    shi_fit:  Optional[ShiPolynomialFit] = None,
) -> Dict:
    """Per-timestep subgradient of the rainflow degradation cost.

    Implements Shi et al. (2018) Eqs. 17-18:
        Charging  t ∈ T_{v_i}: ∂f/∂c_t = Φ'(v_i)·(B·τ·η_in)/2
        Discharge t ∈ T_{w_j}: ∂f/∂d_t = Φ'(w_j)·(B·τ)/(2·η_out)

    Args:
        storage_e:   Battery energy [MWh] time-series.
        cycles:      Output of rainflow_cycle_counting().
        dt_hours:    Timestep [h].
        battery_replacement_cost_per_MWh: B [currency/MWh].
        eff_in:      Charging efficiency η_in.
        eff_out:     Discharging efficiency η_out.
        T_C:         Cell temperature [°C].
        shi_fit:     ShiPolynomialFit — pass window-specific fit.
                     Falls back to module-level default [0,1] if None.

    Returns:
        Dict: subgrad_charge, subgrad_discharge, subgrad_combined,
              attribution, cycle_coverage, n_cycles, shi_fit.
    """
    _fit = shi_fit if shi_fit is not None else _DEFAULT_SHI_FIT
    n    = len(storage_e)
    B    = float(battery_replacement_cost_per_MWh)
    tau  = float(dt_hours)

    attribution = build_half_cycle_map(cycles, storage_e)
    dod_at_t    = attribution["dod_at_t"]
    soc_at_t    = attribution["soc_at_t"]
    direction   = attribution["direction"]

    phi_prime  = phi_shi_prime_with_stress(dod_at_t, soc_at_t, T_C, _fit.k3, _fit.k4)
    attributed = attribution["cycle_owner"] != -1
    phi_prime  = np.where(attributed, phi_prime, 0.0)

    is_charging    = direction == -1
    is_discharging = direction == +1

    subgrad_c = np.where(is_charging,    phi_prime * (B * tau * eff_in) / 2.0,      0.0)
    subgrad_d = np.where(is_discharging, phi_prime * (B * tau) / (2.0 * eff_out),   0.0)
    subgrad_combined = subgrad_d - subgrad_c

    return {
        "subgrad_charge":    subgrad_c,
        "subgrad_discharge": subgrad_d,
        "subgrad_combined":  subgrad_combined,
        "attribution":       attribution,
        "cycle_coverage":    float(np.sum(attributed)) / max(n, 1),
        "n_cycles":          len(cycles),
        "shi_fit":           _fit,
    }


# =============================================================================
# Main analysis API — drop-in replacement for analyze_degradation()
# =============================================================================

def analyze_degradation_shi(
    storage_p:      List[float],
    storage_e:      List[float],
    e_cap_nominal:  float,
    battery_params: Dict,
    dt_hours:       float = 1.0,
    n_bins_dod:     int   = 10,
    enable_rainflow: bool = True,
    T_cell_C:       float = 25.0,
    shi_fit:        Optional[ShiPolynomialFit] = None,
    shi_params:     Optional[ShiModelParams]   = None,
    eol_thresholds: Optional[List[float]]      = None,
) -> Dict:
    """Complete degradation analysis using the pure Shi (2018) model.

    Drop-in replacement for xu.analyze_degradation().
    Returns the same dict keys so run scripts, CSV logging, and plot
    functions work without changes.

    Accumulation : Σ count_i · Φ(δ_i) · S_σ(σ_i) · S_T(T)  (no calendar)
    Capacity fade: Xu SEI two-exponential fed with fd_shi     (Option B)

    Args:
        storage_p:       Battery power [MW] (+ discharge, − charge).
        storage_e:       Battery energy [MWh] time-series.
        e_cap_nominal:   Nominal energy capacity [MWh].
        battery_params:  Must include 'power_capacity_W'.
        dt_hours:        Timestep [h].
        enable_rainflow: If False, skip rainflow (zero cycle degradation).
        T_cell_C:        Cell temperature [°C]. Default 25°C (S_T = 1.0).
        shi_fit:         ShiPolynomialFit from fit_shi_polynomial(soc_min, soc_max).
                         Always pass the window-specific fit from the YAML.
        shi_params:      ShiModelParams — alternative to shi_fit.
        eol_thresholds:  SoH fractions to treat as EoL. Default [0.80, 0.60].

    Returns:
        Dict with same keys as analyze_degradation() plus:
            'fd_shi'         float — Shi accumulated degradation (= 'fd')
            'fd_calendar'    float — always 0.0
            'shi_polynomial' ShiPolynomialFit — k3/k4 provenance
    """
    if shi_params is None:
        if shi_fit is None:
            warnings.warn(
                "analyze_degradation_shi: no shi_fit provided. "
                "Falling back to default [0,1] fit — pass "
                "shi_fit=fit_shi_polynomial(soc_min, soc_max).",
                UserWarning, stacklevel=2,
            )
            shi_fit = _DEFAULT_SHI_FIT
        shi_params = ShiModelParams.from_fit(shi_fit)
    if shi_fit is None:
        shi_fit = _DEFAULT_SHI_FIT

    n_steps         = len(storage_e)
    t_total_seconds = n_steps * dt_hours * 3600.0
    e_arr           = np.asarray(storage_e, dtype=float)
    e_cap           = max(float(e_cap_nominal), 1e-9)
    sigma_mean      = float(np.mean(e_arr)) / e_cap

    total_cycles = count_equivalent_full_cycles(storage_p, storage_e, e_cap, dt_hours)

    cycles: List[Dict] = []
    if enable_rainflow:
        cycles = rainflow_cycle_counting(storage_e=storage_e, e_cap=e_cap)

    fd_shi, fd_cycle = compute_fd_shi(cycles, shi_params, T_cell_C)
    fd_calendar      = 0.0

    capacity_loss      = shi_capacity_loss(fd_shi, shi_params)
    capacity_retention = 1.0 - capacity_loss
    soh                = capacity_retention * 100.0
    fade_pct           = capacity_loss * 100.0

    dod_bins, dod_counts = calculate_dod_distribution(storage_e, e_cap, n_bins=n_bins_dod)

    shi_cycle_stats: Dict = {}
    if cycles:
        dods     = np.array([c["dod"]      for c in cycles])
        socs     = np.array([c["soc_mean"] for c in cycles])
        cnts     = np.array([c["count"]    for c in cycles])
        phi_vals = np.array([
            float(phi_shi_cycle(c["dod"], c["soc_mean"], T_cell_C, shi_params))
            for c in cycles
        ])
        shi_cycle_stats = {
            "n_rainflow_cycles": float(np.sum(cnts)),
            "mean_dod":          float(np.average(dods, weights=cnts)),
            "max_dod":           float(np.max(dods)),
            "mean_soc":          float(np.average(socs, weights=cnts)),
            "mean_phi":          float(np.average(phi_vals, weights=cnts)),
            "mean_fc":           float(np.average(phi_vals, weights=cnts)),  # compat alias
        }

    if "power_capacity_W" not in battery_params:
        raise KeyError("battery_params must include 'power_capacity_W'")
    p_cap_MW = float(battery_params["power_capacity_W"]) / 1e6

    if eol_thresholds is None:
        eol_thresholds = [0.80, 0.60]

    fd_per_yr = float(fd_shi) / max(t_total_seconds / 3600.0 / 8760.0, 1e-9)
    eol_years: Dict[float, Optional[float]] = {}
    for thr in eol_thresholds:
        if (1.0 - shi_capacity_loss(0.0, shi_params)) < thr:
            eol_years[thr] = 0.0
            continue
        lo, hi = 0.0, 200.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if 1.0 - shi_capacity_loss(fd_per_yr * mid, shi_params) > thr:
                lo = mid
            else:
                hi = mid
        eol_years[thr] = round((lo + hi) / 2.0, 2) if (lo + hi) / 2.0 < 190.0 else None

    return {
        "total_cycles":             float(total_cycles),
        "fd":                       float(fd_shi),
        "fd_cycle":                 float(fd_cycle),
        "fd_calendar":              float(fd_calendar),   # always 0.0
        "fd_shi":                   float(fd_shi),        # explicit alias
        "capacity_loss":            float(capacity_loss),
        "capacity_retention":       float(capacity_retention),
        "capacity_fade_percent":    float(fade_pct),
        "soh":                      float(soh),
        "dod_distribution":         (dod_bins, dod_counts),
        "cycle_depth_distribution": cycles,
        "xu_cycle_stats":           shi_cycle_stats,      # compat key
        "shi_cycle_stats":          shi_cycle_stats,
        "eol_years":                eol_years,
        "e_cap_degraded":           e_cap * capacity_retention,
        "p_cap_degraded":           p_cap_MW * capacity_retention,
        "shi_polynomial":           shi_fit,
        "meta": {
            "model":            "Shi2018_polynomial_SEI",
            "fade_mapping":     "Xu_SEI_Option_B",
            "k3":               float(shi_params.k3),
            "k4":               float(shi_params.k4),
            "fit_source":       shi_fit.source,
            "soc_window":       [shi_fit.soc_min, shi_fit.soc_max],
            "dt_hours":         float(dt_hours),
            "t_total_hours":    t_total_seconds / 3600.0,
            "T_cell_C":         float(T_cell_C),
            "sigma_mean":       float(sigma_mean),
            "enable_rainflow":  bool(enable_rainflow),
            "eol_thresholds":   list(eol_thresholds),
            "fd_calendar_note": "Shi model is cycle-only — fd_calendar is structurally 0.0.",
        },
    }


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    import time

    print("=" * 70)
    print("shi.py — Self-Contained Self-Test")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load all battery parameters from battery.yaml
    # ------------------------------------------------------------------
    def _load_battery_params_from_yaml(yaml_path):
        """Load full battery spec from battery.yaml.

        Reads: soc_min/soc_max/soc_initial, energy_capacity (Wh),
        power_capacity (W), round_trip_efficiency_nominal, temperature_C,
        capex_EUR_per_kWh.  Falls back to battery.yaml defaults if YAML is
        absent or a field is missing.
        """
        _defaults = {
            "soc_min":              0.10,
            "soc_max":              0.90,
            "soc_initial":          0.50,
            "e_cap_Wh":             300e6,        # 300 MWh
            "p_cap_W":              150e6,         # 150 MW
            "temperature_C":        25.0,
            "capex_EUR_per_kWh":    150.0,
            "eta_rt":               0.9025,        # 0.95 × 0.95
            "source":               "battery.yaml defaults (YAML not found)",
        }
        if yaml_path is None or not yaml_path.exists():
            return _defaults
        try:
            import yaml as _yaml
            with open(yaml_path) as _f:
                _cfg = _yaml.safe_load(_f)
            _bs  = _cfg.get("battery_systems", {})
            _lim = _cfg.get("operating_limits", {})
            _eco = _cfg.get("economics", {})
            _deg = _cfg.get("degradation", {})
            return {
                "soc_min":           float(_lim.get("soc_min",         _defaults["soc_min"])),
                "soc_max":           float(_lim.get("soc_max",         _defaults["soc_max"])),
                "soc_initial":       float(_lim.get("soc_initial",     _defaults["soc_initial"])),
                "e_cap_Wh":          float(_bs.get("energy_capacity",  _defaults["e_cap_Wh"])),
                "p_cap_W":           float(_bs.get("power_capacity",   _defaults["p_cap_W"])),
                "temperature_C":     float(_deg.get("temperature_C",   _defaults["temperature_C"])),
                "capex_EUR_per_kWh": float(_eco.get("capex_EUR_per_kWh", _defaults["capex_EUR_per_kWh"])),
                "eta_rt":            float(_bs.get("round_trip_efficiency_nominal", _defaults["eta_rt"])),
                "source":            f"{yaml_path.name} via PyYAML",
            }
        except ImportError:
            _defaults["source"] = "battery.yaml defaults (PyYAML not installed)"
            return _defaults
        except Exception as _exc:
            print(f"  WARNING: YAML parse error ({_exc}) — using battery.yaml defaults")
            _defaults["source"] = "battery.yaml defaults (YAML parse error)"
            return _defaults

    from degradation.paths import BATTERY_YAML as _yaml_path

    _bp = _load_battery_params_from_yaml(_yaml_path if _yaml_path.exists() else None)

    # Unpack into named locals for readability
    yaml_soc_min   = _bp["soc_min"]
    yaml_soc_max   = _bp["soc_max"]
    yaml_soc_init  = _bp["soc_initial"]
    yaml_src       = _bp["source"]
    _e_cap_MWh     = _bp["e_cap_Wh"] / 1e6           # Wh → MWh (model unit)
    _p_cap_W       = _bp["p_cap_W"]
    _T_cell_C      = _bp["temperature_C"]
    _capex_per_MWh = _bp["capex_EUR_per_kWh"] * 1e3   # EUR/kWh → EUR/MWh
    _eta_rt        = _bp["eta_rt"]
    _eta_out       = float(np.sqrt(_eta_rt))           # symmetric split: η_in = η_out

    print(f"\n  YAML source   : {yaml_src}")
    print(f"  soc=[{yaml_soc_min},{yaml_soc_max}]  soc_initial={yaml_soc_init}  "
          f"max_dod={yaml_soc_max - yaml_soc_min:.2f}")
    print(f"  e_cap={_e_cap_MWh:.0f} MWh  p_cap={_p_cap_W/1e6:.0f} MW  "
          f"T_cell={_T_cell_C}°C")
    print(f"  capex={_capex_per_MWh:.0f} EUR/MWh  eta_rt={_eta_rt:.4f}  "
          f"eta_out={_eta_out:.4f}")

    shi_fit    = fit_shi_polynomial(yaml_soc_min, yaml_soc_max, source=yaml_src, verbose=True)
    shi_params = ShiModelParams.from_fit(shi_fit)

    print(f"\n  ShiModelParams: k3={shi_params.k3:.4e}  k4={shi_params.k4:.4f}  "
          f"alpha_sei={shi_params.alpha_sei}  beta_sei={shi_params.beta_sei}")

    # Convexity check
    _d   = np.linspace(0.01, shi_fit.max_dod, 10_000)
    _d2  = np.gradient(phi_shi_prime(_d, shi_fit.k3, shi_fit.k4), _d)
    assert int(np.sum(_d2 < 0)) == 0, "Phi_shi has non-convex points!"
    print(f"\n  Global convexity: 0 non-convex points  "
          f"k4*(k4-1)={shi_fit.k4*(shi_fit.k4-1):.4f} ✓")

    # ------------------------------------------------------------------
    # 2. Representative SoC profile from YAML parameters
    #    Sinusoidal modulation centred on soc_initial, bounded by
    #    [soc_min, soc_max] from YAML operating_limits.
    # ------------------------------------------------------------------
    _t         = np.arange(8760)
    _soc       = yaml_soc_min + (yaml_soc_max - yaml_soc_min) * (
                    0.5 + 0.4 * np.sin(2 * np.pi * _t / 24.0)
                    * np.sin(2 * np.pi * _t / (24 * 365)))
    _storage_e = (np.clip(_soc, yaml_soc_min, yaml_soc_max) * _e_cap_MWh).tolist()
    _storage_p = list(np.diff(_storage_e, prepend=_storage_e[0]))
    _bat       = {"power_capacity_W": _p_cap_W}

    t0    = time.perf_counter()
    _degr = analyze_degradation_shi(
        _storage_p, _storage_e, _e_cap_MWh, _bat,
        shi_fit=shi_fit, T_cell_C=_T_cell_C, eol_thresholds=[0.80, 0.60],
    )
    print(f"\n  analyze_degradation_shi: {(time.perf_counter()-t0)*1000:.1f} ms")
    print(f"    fd_shi      = {_degr['fd_shi']:.6f}")
    print(f"    fd_calendar = {_degr['fd_calendar']}  (must be 0.0)")
    print(f"    SoH         = {_degr['soh']:.3f}%")
    print(f"    EoL @ 80%   = {_degr['eol_years'].get(0.80)} yr")
    print(f"    EoL @ 60%   = {_degr['eol_years'].get(0.60)} yr")
    print(f"    model tag   = {_degr['meta']['model']}")

    assert _degr["fd_calendar"] == 0.0
    assert _degr["fd_shi"] == _degr["fd"]
    assert _degr["meta"]["model"] == "Shi2018_polynomial_SEI"
    assert _degr["xu_cycle_stats"] is _degr["shi_cycle_stats"]

    # ------------------------------------------------------------------
    # 3. Subgradient sanity — ratio discharge/charge = 1/eta_out
    # ------------------------------------------------------------------
    _cycles = rainflow_cycle_counting(_storage_e, _e_cap_MWh)
    _sg = compute_subgradient(
        _storage_e, _cycles, dt_hours=1.0,
        battery_replacement_cost_per_MWh=_capex_per_MWh,
        eff_in=_eta_out, eff_out=_eta_out, shi_fit=shi_fit,
    )
    _d_arr = _sg["subgrad_discharge"][_sg["subgrad_discharge"] > 0]
    _c_arr = _sg["subgrad_charge"][_sg["subgrad_charge"] > 0]
    _ratio = _d_arr.mean() / _c_arr.mean()
    _expected_ratio = 1.0 / (_eta_out * _eta_out)   # = 1/eta_rt; ratio = (1/eta_out)/eta_in
    assert abs(_ratio - _expected_ratio) < 0.02, \
        f"ratio {_ratio:.4f} != {_expected_ratio:.4f}"
    assert float(_sg["cycle_coverage"]) > 0.7
    print(f"\n  Subgradient discharge/charge = {_ratio:.4f}  "
          f"(expected {_expected_ratio:.4f} = 1/eta_rt,  "
          f"eta_in=eta_out={_eta_out:.4f} from YAML) ✓")
    print(f"  Cycle coverage = {_sg['cycle_coverage']*100:.1f}% ✓")
    print(f"\n  Battery spec used in self-test  ({yaml_src}):")
    print(f"    e_cap={_e_cap_MWh:.0f} MWh  |  p_cap={_p_cap_W/1e6:.0f} MW  |  "
          f"T={_T_cell_C}°C  |  capex={_capex_per_MWh:.0f} EUR/MWh  |  "
          f"eta_out={_eta_out:.4f}")
    print("\n  All assertions passed ✓")
    print("  No imports from xu ✓")