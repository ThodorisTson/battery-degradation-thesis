"""Single-window baseline: 20-year dispatch with degradation and replacement.

Fork origin: run_battery_xu_shi_degradation_v5_6_RTE_test.py

Solves the wind-battery dispatch problem for one fixed SoC window over a 20-year horizon, re-solving each year at the degraded capacity, and reports
capacity fade, state of health, end of life, and net present value. No parameter sweep; use scripts/run_sweep_2d.py for sizing.

Degradation model
    DEG_MODEL selects the reporting path. "xu" runs the full Xu et al. (2016) model; "shi" runs the fitted polynomial with the Xu calendar term, at
    roughly half the runtime. Both write to separate output directories, since a Shi run and an Xu run of the same case produce identical filenames.
    Gradients always use the Shi polynomial, which is convex; see degradation/xu.py for why the two differ.

Round-trip efficiency
    Efficiency is YAML-driven, never overridden in code. The AC round trip is round_trip_efficiency_nominal (DC) times pcu_efficiency squared. To change
    it, edit both fields in config/battery.yaml together: setting DC alone to 0.95 gives AC 0.9236, not the DEA catalogue value of 0.910. Every output
    filename carries an _rte### tag from the AC round trip actually loaded, and the npv_summary row records rte_dc, pcu, rte_ac and eta.

Economics
    Revenue is reported on three bases (battery-only marginal, arbitrage, and total plant) and discounted per year rather than annuitised. Initial and
    replacement capex are separate, following the Danish Technology Catalogue rationale documented in config/battery.yaml. See degradation/economics.py
    for the discounting horizon convention.

Known limitations
    - config/battery.yaml describes a generic utility-scale Li-ion cost basis while the Xu degradation parameters are calibrated for LMO chemistry.
    - The single-year SHIPP sizing solve still runs, for the equivalent full cycles, dispatch, and single-year degradation report, but its NPV is
      superseded by the multi-year assembly.

Author: Thodoris Tsonopoulos, MSc thesis, TU Delft Wind Energy
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy_financial as npf
import pandas as pd
import matplotlib.pyplot as plt
import time

from shipp.kernel import solve_lp_sparse
from shipp.kernel_pyomo import solve_lp_pyomo
from shipp.components import Storage, Production, TimeSeries

from degradation.site import quick_setup, get_wake_model

# Xu — reporting baseline + calendar correction helpers
from degradation.xu import (
    analyze_degradation,
    count_equivalent_full_cycles,
    rainflow_cycle_counting,
    ft_calendar,
    sei_capacity_loss,
    compute_fd,
    fc_cycle,
    phi_shi_with_stress,
    XU_LMO,
)

# Shi — cycle accumulation + gradient
from degradation.shi import analyze_degradation_shi, phi_shi_prime_with_stress, s_soc, s_temp
from degradation.subgradient import compute_subgradient
from degradation.xu import fit_shi_polynomial

from degradation.plots import (
    plot_degradation_analysis,
    print_degradation_report,
)

import xarray as xr
from py_wake.site import XRSite

from degradation.plots_multiyear import (
    plot_gradient_analysis,
    plot_subgradient_timeseries,
    plot_multiyear_trajectory,
)

# Single source of truth for all economic primitives
from degradation.economics import (
    eta_symmetric,
    capex,
    replacement_cost,
    annuity_factor,
    discount_weights,
    degradation_cost,
    revenue_annual,
    HEADLINE_BASIS,
)

# Single source of truth for every path in the repository
from degradation.paths import HPP_YAML, PRICE_CSV_2019, results_dir

# =============================================================================
# CONFIG
# =============================================================================

PRICE_CSV = PRICE_CSV_2019          # switch to PRICE_CSV_2022 for the 2022 price year

# Degradation model: "shi" (fitted Phi + Xu calendar, fast) or "xu" (full Xu, about twice the runtime). Declared here because the output folder is keyed
# on it: a Shi run and an Xu run of the same case produce files with identical names, so they must not share a directory.
DEG_MODEL = "xu"

BRANCH_DIR_NAME = {"shi": "shi", "xu": "xu"}
if DEG_MODEL not in BRANCH_DIR_NAME:
    raise ValueError(
        f"DEG_MODEL must be one of {sorted(BRANCH_DIR_NAME)}, got {DEG_MODEL!r}."
    )

RESULTS_DIR = results_dir(f"baseline/{BRANCH_DIR_NAME[DEG_MODEL]}")
PLOTS_DIR   = results_dir(f"baseline/{BRANCH_DIR_NAME[DEG_MODEL]}/plots")

discount_rate = 0.03   # real discount rate, 3% (see Section 3.2.2)
dt            = 1.0    # hours

# Solver
pyo_solver = "gurobi"   # 'gurobi' recommended; 'none' = scipy sparse (no DoD, ≤6 mo)

# Horizon
RUN_FULL_YEAR    = True
N_DAYS_TEST      = 30
MAX_HOURS_SPARSE = 180 * 24

# Problem settings
p_min      = 0.0
WAKE_MODEL = "Bastankhah"

# EoL thresholds for reporting (SoH fractions, e.g. 0.80 = 80% capacity remaining)
eol_thresholds = [0.80, 0.70, 0.60]

# Project lifetime and replacement
N_YEARS              = 20
EOL_REPLACEMENT      = 0.70
EOL_REPLACEMENT_TOL  = 0.005   # triggers at EOL_REPLACEMENT + TOL (e.g. 70.5%)

# Output toggles
print_baseline_table = True
print_degr_reports   = True
SAVE_CSV             = True
SAVE_REPORT          = True
MAKE_PLOT            = True
show_plots           = False

run_ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
FILE_TAG = "v56_rtetest"   # RTE sensitivity variant; rte value appended per run in _build_run_label


def _build_run_label(
    ts: str, price_csv: Path, p_cap: float, e_cap: float,
    soc_min: float = 0.10, soc_max: float = 0.90,
    rte_ac: Optional[float] = None,
) -> str:
    stem    = price_csv.stem.lower()
    year    = "".join(filter(str.isdigit, stem))[-4:]
    dataset = f"dk{year}"
    bat     = f"{int(round(p_cap))}mw_{int(round(e_cap))}mwh"
    soc_tag = f"soc{int(soc_min*100)}_{int(soc_max*100)}"
    rte_tag = f"_rte{int(round(rte_ac * 1000)):03d}" if rte_ac is not None else ""
    return f"{ts}_{dataset}_{bat}_{soc_tag}_{FILE_TAG}{rte_tag}"


# =============================================================================
# Small structs
# =============================================================================

@dataclass
class HorizonData:
    n:         int
    ws:        np.ndarray
    wd:        np.ndarray
    ti:        Optional[np.ndarray]
    price_eur: np.ndarray


# =============================================================================
# Data loading helpers (unchanged from v5.1)
# =============================================================================

def _find_price_column(df: pd.DataFrame) -> str:
    for c in ("price_eur_mwh", "price_eur_per_mwh", "Price", "price"):
        if c in df.columns:
            return c
    raise KeyError(f"No recognized price column. Columns: {df.columns.tolist()}")


def _choose_horizon(n_wind: int, n_price: int) -> int:
    requested = n_wind if RUN_FULL_YEAR else int(N_DAYS_TEST * 24)
    n = min(requested, n_wind, n_price)
    if pyo_solver == "none":
        n = min(n, MAX_HOURS_SPARSE)
    return int(n)


def _load_inputs(hpp: dict) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    ts = hpp["site"]["energy_resource"]["time_series"]["wind_resource"]
    ws = np.asarray(ts["wind_speed"],     dtype=float)
    wd = np.asarray(ts["wind_direction"], dtype=float)
    ti = None
    ti_dat = ts.get("turbulence_intensity")
    if isinstance(ti_dat, dict) and "data" in ti_dat:
        ti = np.asarray(ti_dat["data"], dtype=float)
    if ws.shape != wd.shape:
        raise ValueError("Wind speed and wind direction arrays have different lengths.")
    if ti is not None and ti.shape != ws.shape:
        ti = None
    return ws, wd, ti


def _load_prices() -> np.ndarray:
    df  = pd.read_csv(PRICE_CSV)
    col = _find_price_column(df)
    return df[col].astype(float).to_numpy()


def _run_pywake_power_MW(
    setup: dict,
    wd:    np.ndarray,
    ws:    np.ndarray,
    ti:    Optional[np.ndarray],
) -> np.ndarray:
    n    = len(ws)
    site = XRSite(ds=xr.Dataset(data_vars=dict(P=1)))
    wf_model = get_wake_model(WAKE_MODEL, site, setup["windturbine"])
    time_days = np.arange(n) / 24.0
    kwargs = {"x": setup["x"], "y": setup["y"], "wd": wd, "ws": ws, "time": time_days}
    if ti is not None:
        kwargs["TI"] = ti
    sim_res = wf_model(**kwargs)
    return sim_res.Power.sum(["wt"]).values / 1e6


# =============================================================================
# SHIPP component builders  (v5.6: EUR-native, symmetric efficiency)
# =============================================================================

def _build_shipp_components(
    setup:    dict,
    p_max_MW: float,
) -> Tuple[Storage, Storage, float, float, float, float, float, float, float, float, float]:
    """Build nominal-capacity Storage and null Storage from YAML battery params.

    Returns
    -------
    (stor, stor_null, e_cap_MWh, p_cap_MW, rte_ac, eta, 
     e_cost_EUR_per_MWh, p_cost_EUR_per_MW,
     repl_e_EUR_per_MWh, repl_p_EUR_per_MW,
     soc_min, soc_max)

    Changes vs v5.1
    ---------------
    - Costs in EUR (not USD).  eur_to_usd removed.
    - eff_in = eff_out = eta = sqrt(rte_ac) via eta_symmetric() instead ofeff_in=1.0 / eff_out=rte_ac.
    - Separate repl_e / repl_p read from YAML for replacement cost anddegradation valuation (72 / 96 EUR/kWh, kW).
    """
    bat = setup["battery"]

    e_cap_MWh = float(bat["energy_capacity_Wh"]) / 1e6
    p_cap_MW  = float(bat["power_capacity_W"])   / 1e6

    rte_dc   = float(bat["rte_nominal"])
    pcu_eff  = float(bat["pcu_efficiency"])
    rte_ac   = rte_dc * (pcu_eff ** 2)
    eta      = eta_symmetric(rte_ac)   # symmetric split: eta^2 == rte_ac

    e_cost_EUR_per_MWh = float(bat["capex_EUR_per_kWh"]) * 1000.0   # 245 EUR/kWh → EUR/MWh
    p_cost_EUR_per_MW  = float(bat["capex_EUR_per_kW"])  * 1000.0   # 86  EUR/kW  → EUR/MW
    repl_e_EUR_per_MWh = float(bat["repl_energy_EUR_per_kWh"]) * 1000.0  # 72 EUR/kWh
    repl_p_EUR_per_MW  = float(bat["repl_power_EUR_per_kW"])   * 1000.0  # 96 EUR/kW

    soc_min = float(bat.get("soc_min", 0.10))
    soc_max = float(bat.get("soc_max", 0.90))

    dod_yaml = 1.0 - soc_min
    if pyo_solver == "none":
        dod_eff = 1.0
        print(f"  ⚠ DoD ({dod_yaml:.0%}) ignored, scipy sparse requires dod=1.0.")
    else:
        dod_eff = dod_yaml

    print(f"  SoC window : {soc_min*100:.0f}% – {soc_max*100:.0f}%  "
          f"(DoD={dod_eff:.0%}, enforced by {pyo_solver})")
    print(f"  eta        : {eta:.4f}  (eff_in = eff_out = sqrt(rte_ac={rte_ac:.4f}))")

    stor = Storage(
        e_cap=e_cap_MWh,
        p_cap=p_cap_MW,
        eff_in=eta,
        eff_out=eta,
        e_cost=e_cost_EUR_per_MWh,
        p_cost=p_cost_EUR_per_MW,
        dod=dod_eff,
    )
    stor_null = Storage(e_cap=0.0, p_cap=0.0, eff_in=1.0, eff_out=1.0,
                        e_cost=0.0, p_cost=0.0)

    return (stor, stor_null, e_cap_MWh, p_cap_MW, rte_ac, eta,
            e_cost_EUR_per_MWh, p_cost_EUR_per_MW,
            repl_e_EUR_per_MWh, repl_p_EUR_per_MW,
            soc_min, soc_max)


def _solve_shipp(
    price_eur:     np.ndarray,
    power_wind_MW: np.ndarray,
    stor:          Storage,
    stor_null:     Storage,
    p_max_MW:      float,
    n:             int,
    soc_max:       float = 1.0,
):
    """Single-year SHIPP solve (sizing + dispatch-fixed).

    v5.6: price passed directly in EUR (no USD conversion).
    """
    price_dam  = TimeSeries(price_eur.tolist(), dt)   # EUR — no eur_to_usd
    prod       = Production(TimeSeries(power_wind_MW.tolist(), dt), p_cost=0.0)
    prod_null  = Production(TimeSeries([0.0] * n, dt), p_cost=0.0)

    if pyo_solver == "none":
        os       = solve_lp_sparse(price_dam, prod, prod_null, stor, stor_null,
                                   discount_rate, N_YEARS, p_min, p_max_MW, n)
        os_fixed = solve_lp_sparse(price_dam, prod, prod_null, stor, stor_null,
                                   discount_rate, N_YEARS, p_min, p_max_MW, n,
                                   fixed_cap=True)
    else:
        os       = solve_lp_pyomo(price_dam, prod, prod_null, stor, stor_null,
                                  discount_rate, N_YEARS, p_min, p_max_MW, n,
                                  pyo_solver, soc_max1=soc_max)
        os_fixed = solve_lp_pyomo(price_dam, prod, prod_null, stor, stor_null,
                                  discount_rate, N_YEARS, p_min, p_max_MW, n,
                                  pyo_solver, fixed_cap=True, return_duals=True,
                                  soc_max1=soc_max)
    return os, os_fixed


# =============================================================================
# Year-level helpers
# =============================================================================

def _build_storage_year(
    e_cap_eff:          float,
    p_cap_MW:           float,
    rte_ac:             float,
    e_cost_EUR_per_MWh: float,
    soc_min:            float,
    soc_max:            float,
) -> Storage:
    """Per-year Storage with degraded capacity.  v5.6: symmetric efficiency, EUR."""
    dod_eff = 1.0 - soc_min if pyo_solver != "none" else 1.0
    eta     = eta_symmetric(rte_ac)
    return Storage(
        e_cap=e_cap_eff,
        p_cap=p_cap_MW,
        eff_in=eta,
        eff_out=eta,
        e_cost=e_cost_EUR_per_MWh,
        p_cost=0.0,
        dod=dod_eff,
    )


def _xu_full_degradation(
    storage_p: list, storage_e: list,
    e_cap_eff: float, T_cell_C: float, dt_hours: float,
) -> Dict:
    """Full Xu model (Xu cycling S_delta + Xu calendar) on one year's dispatch.

    Mirrors _xu_full_degradation from v5.4.  Used when DEG_MODEL = 'xu'.
    Returns keys 'fd', 'fd_cycle', 'fd_calendar' (period values; loop annualises).
    """
    e_arr           = np.asarray(storage_e, dtype=float)
    n_steps         = len(e_arr)
    t_total_seconds = n_steps * dt_hours * 3600.0
    sigma_mean      = float(np.mean(e_arr)) / max(e_cap_eff, 1e-9)
    cycles          = rainflow_cycle_counting(storage_e, e_cap_eff)
    fd, fd_cycle, fd_cal = compute_fd(cycles, sigma_mean, t_total_seconds, T_cell_C)
    return {"fd": float(fd), "fd_cycle": float(fd_cycle), "fd_calendar": float(fd_cal)}


def _xu_full_degradation_report(
    storage_p:      List[float],
    storage_e:      List[float],
    e_cap_eff:      float,
    T_cell_C:       float,
    dt_hours:       float,
    eol_thresholds: List[float],
    bat_params:     Dict,
) -> Dict:
    """Full Xu model on one year's dispatch, in the reporting dict format.

    _xu_full_degradation above returns only the three degradation terms, which is all the multi-year loop needs. The single-year reporting path also needs
    the state of health, the end-of-life years and the cycle statistics, so those are assembled here.

    The post-processing mirrors the tail of _shi_with_calendar_correction. It is duplicated rather than factored out because that function is validated
    and used for every result produced so far; the two must be kept in step ifeither is edited.
    """
    e_arr           = np.asarray(storage_e, dtype=float)
    n_steps         = len(e_arr)
    t_total_seconds = n_steps * dt_hours * 3600.0
    t_total_hours   = t_total_seconds / 3600.0
    sigma_mean      = float(np.mean(e_arr)) / max(e_cap_eff, 1e-9)

    cycles = rainflow_cycle_counting(storage_e, e_cap_eff)
    fd_total, fd_cycle, fd_cal = compute_fd(
        cycles, sigma_mean, t_total_seconds, T_cell_C
    )

    dods = np.array([c["dod"]      for c in cycles], dtype=float)
    cnts = np.array([c["count"]    for c in cycles], dtype=float)
    socm = np.array([c["soc_mean"] for c in cycles], dtype=float)

    L_corr   = sei_capacity_loss(float(fd_total))
    cap_ret  = 1.0 - L_corr

    fd_per_yr = float(fd_total) / max(t_total_hours / 8760.0, 1e-9)
    eol: Dict[float, Optional[float]] = {}
    for thr in eol_thresholds:
        if (1.0 - sei_capacity_loss(0.0)) < thr:
            eol[thr] = 0.0
            continue
        lo, hi = 0.0, 200.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if 1.0 - sei_capacity_loss(fd_per_yr * mid) > thr:
                lo = mid
            else:
                hi = mid
        val = (lo + hi) / 2.0
        eol[thr] = round(val, 2) if val < 190.0 else None

    return {
        "meta": {
            "model": "Xu2016_semiempirical",
            "calendar_correction":
                "Xu ft_calendar is native to this branch, no correction applied",
        },
        "xu_cycle_stats": {
            "n_rainflow_cycles": float(np.sum(cnts)),
            "mean_dod": float(np.average(dods, weights=cnts)) if len(dods) else 0.0,
            "mean_soc": float(np.average(socm, weights=cnts)) if len(socm) else 0.0,
            "mean_soc_time_avg": sigma_mean,
        },
        "fd":                    float(fd_total),
        "fd_cycle":              float(fd_cycle),
        "fd_calendar":           float(fd_cal),
        "capacity_retention":    float(cap_ret),
        "capacity_loss":         float(L_corr),
        "soh":                   float(cap_ret * 100.0),
        "capacity_fade_percent": float(L_corr * 100.0),
        "eol_years":             eol,
        "e_cap_degraded":        e_cap_eff * cap_ret,
        "p_cap_degraded":        float(bat_params["power_capacity_W"]) / 1e6 * cap_ret,
        "total_cycles":          count_equivalent_full_cycles(
                                     storage_p, storage_e, e_cap_eff,
                                     dt_hours=dt_hours),
    }


def _fd_validate_degradation_term(
    dods:               np.ndarray,
    counts:             np.ndarray,
    soc_means:          np.ndarray,
    e_cap:              float,
    e_cost_EUR_per_MWh: float,
    factor:             float,
    shi_fit,
    T_cell_C:           float = 25.0,
    eps_list:           Tuple[float, ...] = (0.01, 0.005),
    scale:              float = 1.0,
) -> None:
    """Frozen-dispatch finite-difference check of the analytic dDegCost/dE_cap.

    Unchanged from v5.1 except variable renamed e_cost_EUR_per_MWh (was USD).
    """
    if len(dods) == 0:
        print("[1.5] no cycles — degradation FD skipped.")
        return

    k3, k4 = shi_fit.k3, shi_fit.k4
    stress  = s_soc(soc_means) * s_temp(T_cell_C)
    B_fac   = e_cost_EUR_per_MWh * factor * scale

    def deg_cost(e_prime: float) -> float:
        delta_p = dods * (e_cap / e_prime)
        delta_p = np.clip(delta_p, 1e-9, 1.0)
        return B_fac * float(np.sum(counts * k3 * delta_p ** k4 * stress))

    phi_prime = phi_shi_prime_with_stress(dods, soc_means, T_cell_C, k3, k4)
    analytic  = B_fac * float(np.sum(counts * phi_prime * (-dods / e_cap)))

    print(f"[1.5] degradation-term FD check at E_cap={e_cap:.2f} MWh")
    print(f"      analytic dDegCost/dE = {analytic:+.6f}")
    for eps in eps_list:
        dE  = eps * e_cap
        cp  = deg_cost(e_cap + dE)
        cm  = deg_cost(e_cap - dE)
        fd  = (cp - cm) / (2.0 * dE)
        rel = (fd - analytic) / analytic if analytic != 0 else float("nan")
        print(f"      eps={eps:6.3f}  dE={dE:7.3f}  FD={fd:+.6f}  "
              f"rel.err={rel:+.3%}")


def _shi_with_calendar_correction(
    storage_p:      List[float],
    storage_e:      List[float],
    e_cap_eff:      float,
    bat_params:     Dict,
    shi_fit,
    T_cell_C:       float,
    dt_hours:       float,
    eol_thresholds: List[float],
) -> Dict:
    """Shi cycle degradation + Xu calendar correction on the reporting path.

    Unchanged from v5.1 (the dual-Phi separation is unaltered).
    """
    n_steps         = len(storage_e)
    t_total_seconds = n_steps * dt_hours * 3600.0
    t_total_hours   = t_total_seconds / 3600.0
    e_arr           = np.asarray(storage_e, dtype=float)
    sigma_mean      = float(np.mean(e_arr)) / max(e_cap_eff, 1e-9)

    degr = analyze_degradation_shi(
        storage_p, storage_e, e_cap_eff, bat_params,
        shi_fit=shi_fit,
        T_cell_C=T_cell_C,
        dt_hours=dt_hours,
        eol_thresholds=eol_thresholds,
    )

    fd_cal       = ft_calendar(t_total_seconds, sigma_mean, T_cell_C)
    fd_corrected = degr["fd_shi"] + fd_cal
    cal_frac_pct = 100.0 * fd_cal / max(fd_corrected, 1e-30)

    L_corr       = sei_capacity_loss(fd_corrected)
    cap_ret_corr = 1.0 - L_corr
    soh_corr     = cap_ret_corr * 100.0

    fd_per_yr = fd_corrected / max(t_total_hours / 8760.0, 1e-9)
    eol_corr: Dict[float, Optional[float]] = {}
    for thr in eol_thresholds:
        if (1.0 - sei_capacity_loss(0.0)) < thr:
            eol_corr[thr] = 0.0
            continue
        lo, hi = 0.0, 200.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if 1.0 - sei_capacity_loss(fd_per_yr * mid) > thr:
                lo = mid
            else:
                hi = mid
        val = (lo + hi) / 2.0
        eol_corr[thr] = round(val, 2) if val < 190.0 else None

    degr["mean_soc_time_avg"]  = float(sigma_mean)
    degr["fd_calendar"]        = float(fd_cal)
    degr["fd"]                 = float(fd_corrected)
    degr["fd_cycle"]           = degr["fd_shi"]
    degr["capacity_retention"] = float(cap_ret_corr)
    degr["capacity_loss"]      = float(L_corr)
    degr["soh"]                = float(soh_corr)
    degr["capacity_fade_percent"] = float(L_corr * 100.0)
    degr["eol_years"]          = eol_corr
    degr["e_cap_degraded"]     = e_cap_eff * cap_ret_corr
    degr["p_cap_degraded"]     = (
        float(bat_params["power_capacity_W"]) / 1e6 * cap_ret_corr
    )
    degr["meta"]["calendar_correction"] = (
        "Xu ft_calendar added to Shi reporting path (dual-Phi: gradient unchanged)"
    )
    degr["meta"]["fd_calendar_note"] = (
        f"fd_cal={fd_cal:.4e}  ({cal_frac_pct:.1f}% of corrected total)  "
        f"sigma_mean={sigma_mean:.3f}"
    )
    return degr


# =============================================================================
# Multi-year loop  (v5.6: per-year discounting + dual-basis NPV from v5.4)
# =============================================================================

def _run_multiyear(
    wind_8760:          np.ndarray,
    price_8760:         np.ndarray,
    stor_null:          Storage,
    p_max_MW:           float,
    e_cap_nominal:      float,
    p_cap_MW:           float,
    rte_ac:             float,
    e_cost_EUR_per_MWh: float,
    p_cost_EUR_per_MW:  float,
    repl_e_EUR_per_MWh: float,
    repl_p_EUR_per_MW:  float,
    bat_params:         Dict,
    shi_fit,
    soc_min:            float,
    soc_max:            float,
    T_cell_C:           float = 25.0,
    deg_model:          str   = "shi",
) -> Dict:
    """Year-by-year degradation loop with capacity fade feedback and replacement.

    v5.6 changes vs v5.1
    --------------------
    - Price in EUR (no eur_to_usd).
    - Per-year undiscounted revenues tracked on three bases (marginal, arbitrage,
      total plant) via revenue_annual from wp2_econ.
    - Lifetime NPV assembled per-year via discount_weights, not via single-year
      LP NPV minus PV of replacements.
    - Replacement cost uses repl_e / repl_p (72 / 96 EUR/kWh, kW), not initial
      capex, matching the Danish Technology Catalogue scope.
    - deg_model = 'xu' runs the full Xu model instead of Shi + Xu calendar.
    - Gradient computation: e_cost argument to compute_subgradient updated to
      repl_e_EUR_per_MWh (degradation valuation uses replacement cost, not
      initial capex — consistent with dDegCost/dEcap in v5.4).

    Structure and all v5.1 loop mechanics are unchanged.
    """
    n           = len(wind_8760)
    scale       = 1.0 / max(n / 8760.0, 1e-9)
    price_eur   = price_8760

    # Wind-only baseline for marginal revenue computation
    wind_export_no_bat = np.minimum(wind_8760, p_max_MW)
    rev_wind_only = (365.0 * 24.0 / n) * float(np.dot(price_eur, wind_export_no_bat)) * dt

    fd_cumulative    = 0.0
    soh              = 1.0
    n_replacements   = 0
    replacement_years: List[int]   = []
    soh_trajectory:    List[tuple] = []
    annual_fd:         List[tuple] = []
    annual_gradient:   List        = []
    annual_stats:      List        = []   # (mean_dod, e_cap_eff) per year, both branches
    annual_soc:        List        = []
    annual_revenue_bat:   List[float] = []   # battery-only marginal per year
    annual_revenue_total: List[float] = []   # total plant per year
    annual_revenue_arb:   List[float] = []   # arbitrage per year
    soc_frac_fixed:       float       = None

    period_days = n * dt / 24.0
    if n != 8760:
        print(f"  Note: annual tile is {n} h ({period_days:.1f} days), "
              f"fd annualised by x{scale:.5f} each year.")

    print(f"\n{'─'*70}")
    print(f"Multi-year loop: {N_YEARS} yr | tile: {n} h ({period_days:.1f} d) | "
          f"E_cap={e_cap_nominal:.0f} MWh | P_cap={p_cap_MW:.0f} MW | "
          f"model={deg_model} | replacement SoH<"f"{(EOL_REPLACEMENT + EOL_REPLACEMENT_TOL)*100:.1f}%")
    print(f"{'─'*70}")
    print(f"  {'Year':>4}  {'SoH_%':>7}  {'fd_yr':>9}  {'fd_cum':>9}  "
          f"{'rev_kEUR':>10}  {'t_lp_s':>7}  {'Replaced':>8}")
    print(f"  {'────':>4}  {'──────':>7}  {'──────':>9}  {'──────':>9}  "
          f"{'────────':>10}  {'───────':>7}  {'────────':>8}")

    t_loop_start = time.perf_counter()
    t_lp_total   = 0.0
    t_degr_total = 0.0

    for year in range(1, N_YEARS + 1):  # 20 SoH boundary points; revenue discounted over 19 yrs (weights slice below)
        t_yr_start = time.perf_counter()

        # ── Effective capacity this year ──────────────────────────────────
        e_cap_eff   = e_cap_nominal * soh
        e_start1_yr = soc_frac_fixed * e_cap_eff if soc_frac_fixed is not None else None

        # ── Re-solve LP at degraded capacity ──────────────────────────────
        stor_yr   = _build_storage_year(
            e_cap_eff, p_cap_MW, rte_ac, e_cost_EUR_per_MWh, soc_min, soc_max
        )
        price_dam = TimeSeries(price_eur.tolist(), dt)   # EUR directly
        prod_yr   = Production(TimeSeries(wind_8760.tolist(), dt), p_cost=0.0)
        prod_null = Production(TimeSeries([0.0] * n, dt), p_cost=0.0)

        t_lp_start = time.perf_counter()

        if pyo_solver == "none":
            os_yr = solve_lp_sparse(
                price_dam, prod_yr, prod_null, stor_yr, stor_null,
                discount_rate, N_YEARS, p_min, p_max_MW, n, fixed_cap=True,
            )
        else:
            os_yr = solve_lp_pyomo(
                price_dam, prod_yr, prod_null, stor_yr, stor_null,
                discount_rate, N_YEARS, p_min, p_max_MW, n,
                pyo_solver, fixed_cap=True, soc_max1=soc_max,
                return_duals=True, e_start1=e_start1_yr,
            )

        t_lp_yr = time.perf_counter() - t_lp_start
        t_lp_total += t_lp_yr

        # ── Per-year revenues on three bases (via wp2_econ) ───────────────
        p_prod_yr = np.array(os_yr.production_p[0].data, dtype=float)
        p_bat_yr  = np.array(os_yr.storage_p[0].data,    dtype=float)
        rev_yr      = revenue_annual(price_eur, p_bat_yr, p_prod_yr,
                                     wind_8760, p_max_MW, n, dt)
        rev_arb_yr  = rev_yr["arbitrage"]
        rev_bat_yr  = rev_yr["marginal"]               # battery-only (marginal)
        rev_total_yr = rev_bat_yr + rev_wind_only       # total plant
        annual_revenue_arb.append(rev_arb_yr)
        annual_revenue_bat.append(rev_bat_yr)
        annual_revenue_total.append(rev_total_yr)

        # ── Annual degradation ────────────────────────────────────────────
        storage_e_yr = os_yr.storage_e[0].data
        if soc_frac_fixed is None:
            soc_frac_fixed = os_yr.soc_final / e_cap_eff
        annual_soc.append(np.array(storage_e_yr, dtype=float).copy())
        storage_p_yr = os_yr.storage_p[0].data

        t_deg_start = time.perf_counter()

        if deg_model == "xu":
            degr_yr = _xu_full_degradation(
                storage_p_yr, storage_e_yr, e_cap_eff, T_cell_C, dt,
            )
        else:
            degr_yr = _shi_with_calendar_correction(
                storage_p_yr, storage_e_yr, e_cap_eff,
                bat_params, shi_fit, T_cell_C,
                dt, eol_thresholds,
            )

        t_deg_yr = time.perf_counter() - t_deg_start
        t_degr_total += t_deg_yr

        fd_yr          = degr_yr["fd"]          * scale
        fd_yr_cycle    = degr_yr["fd_cycle"]    * scale
        fd_yr_calendar = degr_yr["fd_calendar"] * scale
        fd_cumulative += fd_yr
        annual_fd.append((fd_yr, fd_yr_cycle, fd_yr_calendar))

        # ── Dispatch statistics (both branches) ───────────────────────────
        # Rainflow counting and the cycle-weighted mean amplitude are functionals of the SoC trajectory alone and do not involve Phi, so
        # they are computed on both branches. Only the sub-gradient below is specific to the Shi polynomial.
        cycles_yr   = rainflow_cycle_counting(storage_e_yr, e_cap_eff)
        dods_yr     = np.array([c["dod"]      for c in cycles_yr], dtype=float)
        cnts_yr     = np.array([c["count"]    for c in cycles_yr], dtype=float)
        socm_yr     = np.array([c["soc_mean"] for c in cycles_yr], dtype=float)
        mean_dod_yr = (float(np.average(dods_yr, weights=cnts_yr))
                       if len(dods_yr) > 0 else 0.0)
        annual_stats.append((mean_dod_yr, float(e_cap_eff)))

        # ── Per-year Castillo gradient (Shi branch only) ──────────────────
        grad_yr = None
        if deg_model != "xu" and os_yr.dual_prices is not None:
            dual_yr   = os_yr.dual_prices["dual_e_min1"]
            e_cap_yr  = os_yr.dual_prices["e_cap1"]
            if abs(e_cap_yr - e_cap_eff) > 1e-6 * max(e_cap_eff, 1.0):
                print(f"  WARNING yr {year}: e_cap from duals ({e_cap_yr:.6f} MWh) "
                      f"differs from e_cap_eff ({e_cap_eff:.6f} MWh); the cycles "
                      f"above were counted against e_cap_eff.")
            sg_yr     = compute_subgradient(
                storage_e=storage_e_yr,
                cycles=cycles_yr,
                dt_hours=dt,
                battery_replacement_cost_per_MWh=repl_e_EUR_per_MWh,  # valuation at replacement cost
                eff_in=eta_symmetric(rte_ac),
                eff_out=eta_symmetric(rte_ac),
                shi_fit=shi_fit,
            )

            factor_yr        = npf.npv(discount_rate, np.ones(N_YEARS)) - 1
            dual_per_year_yr = dual_yr / factor_yr

            # [0] OLD fused scalar — DEPRECATED, kept for continuity
            dDeg_dDoD_yr = -e_cap_yr * float(
                np.dot(sg_yr["subgrad_combined"], dual_per_year_yr)
            )

            # TERM 1: dRev/dE_cap (Castillo Theorem 1 / Example 5)
            dual_max_yr   = os_yr.dual_prices["dual_e_max1"]
            dRev_dEcap_yr = (
                soc_max * float(np.sum(dual_max_yr))
                - soc_min * float(np.sum(dual_yr))
            )

            # TERM 2: dDegCost/dE_cap  (frozen-dispatch capacity gradient)
            # Raising E_cap at a fixed energy dispatch rescales the whole normalized SoC trajectory: every cycle becomes a smaller fraction of the pack and sits lower 
            # in it, and the year's mean SoC drops as well. Degradation cost responds through both cycle wear and calendar wear, so this gradient has a cycle part and a calendar part.
            if len(dods_yr) > 0:
                # ---- CYCLE PART (the finite-difference figure, Test 1, validates this) ----
                # per-cycle degradation  Phi_i = k3 * delta^k4 * S_sigma(sigma) * S_T
                phi_full_cyc  = (shi_fit.k3 * dods_yr ** shi_fit.k4
                                 * s_soc(socm_yr) * s_temp(T_cell_C))
                # depth kernel  dPhi/d(delta) * S_sigma * S_T = k3*k4*delta^(k4-1)*S_sigma*S_T
                phi_prime_cyc = phi_shi_prime_with_stress(
                    dods_yr, socm_yr, T_cell_C, shi_fit.k3, shi_fit.k4
                )
                # frozen dispatch: amplitude and cycle mean both scale as 1/E_cap
                ddod_dEcap = -dods_yr / e_cap_yr          # d(delta_i)/dE_cap : cycle gets shallower
                dsig_dEcap = -socm_yr / e_cap_yr          # d(sigma_i)/dE_cap : cycle mean drops
                # per-cycle sensitivity = depth response + mean-SoC (stress) coupling
                dphi_dEcap = (phi_prime_cyc * ddod_dEcap                        # depth term
                              + phi_full_cyc * XU_LMO.k_sigma * dsig_dEcap)     # mean-SoC coupling
                cycle_sum  = float(np.sum(cnts_yr * dphi_dEcap))               # sum over rainflow cycles

                # ---- CALENDAR PART (elementary; no rainflow, no finite difference needed) ----
                # calendar wear = S_t(t) * S_sigma(sigma_bar); only sigma_bar depends on E_cap. Use the SAME mean SoC the calendar function uses: mean(SoC) / e_cap_eff.
                sigma_bar   = float(np.mean(np.asarray(storage_e_yr, dtype=float))) / e_cap_eff
                # d(f_cal)/dE_cap = k_sigma * f_cal * d(sigma_bar)/dE_cap,
                # with d(sigma_bar)/dE_cap = -sigma_bar / E_cap  (mean falls as pack grows)
                calendar_gr = -XU_LMO.k_sigma * sigma_bar * degr_yr["fd_calendar"] / e_cap_eff

                # ---- ASSEMBLE ----
                # Same cost/discount prefactor as before (replacement cost, discount, period-to-year scaling). Both parts are per-period, so 'scale' annualizes them together.
                dDegCost_dEcap_yr = repl_e_EUR_per_MWh * factor_yr * scale * (cycle_sum + calendar_gr)
            else:
                dDegCost_dEcap_yr = 0.0

            # TERM 3: lambda_E (marginal initial capex, not replacement)
            lambda_E_yr   = e_cost_EUR_per_MWh

            # Assembled outer gradient
            dNPV_dEcap_yr = dRev_dEcap_yr - dDegCost_dEcap_yr - lambda_E_yr
            rc_e_cap_yr   = float(os_yr.dual_prices.get("rc_e_cap1", 0.0))

            # [1.3] basis diagnostic
            _ratio = (rc_e_cap_yr / dNPV_dEcap_yr) if dNPV_dEcap_yr != 0.0 else float("nan")
            print(f"[1.3] yr={year:>2d}  n={n}  scale={scale:.3f}  e_cap={e_cap_yr:6.1f}  "
                  f"rc={rc_e_cap_yr:+12.1f}  dNPV={dNPV_dEcap_yr:+12.1f}  "
                  f"ratio(rc/dNPV)={_ratio:+7.3f}")
            print(f"      dRev={dRev_dEcap_yr:+12.1f}  "
                  f"dDeg={dDegCost_dEcap_yr:+12.1f}  lambdaE={lambda_E_yr:+10.1f}  "
                  f"factor={factor_yr:.3f}  scale={scale:.3f}")

            # Year-1 FD validation of the degradation term
            if year == 1:
                _fd_validate_degradation_term(
                    dods=dods_yr, counts=cnts_yr, soc_means=socm_yr,
                    e_cap=e_cap_yr, e_cost_EUR_per_MWh=repl_e_EUR_per_MWh,
                    factor=factor_yr, shi_fit=shi_fit, T_cell_C=T_cell_C,
                    scale=scale,
                )

            grad_yr = (
                dDeg_dDoD_yr,                                        # [0] OLD fused (kept)
                float(np.mean(np.abs(sg_yr["subgrad_combined"]))),   # [1] mean_abs_subgrad
                mean_dod_yr,                                         # [2] mean_dod (weighted)
                float(np.mean(np.abs(dual_per_year_yr))),            # [3] mean_abs_dual
                float(e_cap_yr),                                     # [4] e_cap_eff
                float(np.mean(sg_yr["n_straddled"] == 1)),           # [5] frac_single_straddle
                sg_yr["subgrad_combined"].copy(),                    # [6] full time series
                dRev_dEcap_yr,                                       # [7] dRev/dE
                dDegCost_dEcap_yr,                                   # [8] dDegCost/dE
                lambda_E_yr,                                         # [9] lambda_E
                dNPV_dEcap_yr,                                       # [10] full gradient
                rc_e_cap_yr,                                         # [11] rc cross-check
            )
        annual_gradient.append(grad_yr)

        # ── Update SoH ────────────────────────────────────────────────────
        L_cum = sei_capacity_loss(fd_cumulative)
        soh   = 1.0 - L_cum
        soh_trajectory.append((year, soh * 100.0, fd_cumulative, n_replacements))

        eol_trigger  = EOL_REPLACEMENT + EOL_REPLACEMENT_TOL
        replaced_tag = "YES" if soh < eol_trigger else ""
        t_yr = time.perf_counter() - t_yr_start

        print(f"  {year:>4d}  {soh*100:>7.3f}  {fd_yr:>9.5f}  "
              f"{fd_cumulative:>9.5f}  {rev_bat_yr*1e-3:>10.1f}  "
              f"{t_lp_yr:>7.2f}  {replaced_tag:>8}")

        # ── Battery replacement ────────────────────────────────────────────
        if soh < eol_trigger:
            n_replacements += 1
            replacement_years.append(year)
            fd_cumulative  = 0.0
            soh            = 1.0
            soc_frac_fixed = None
            print(f"  *** Battery #{n_replacements} replaced at end of year {year} "
                  f"(SoH < {eol_trigger*100:.1f}% [{EOL_REPLACEMENT*100:.0f}% + "
                  f"{EOL_REPLACEMENT_TOL*100:.1f}% tol]) ***")

    t_loop_total = time.perf_counter() - t_loop_start
    pct_lp   = 100.0 * t_lp_total   / max(t_loop_total, 1e-9)
    pct_degr = 100.0 * t_degr_total / max(t_loop_total, 1e-9)

    # ── Discounted lifetime NPV — triple basis via wp2_econ ───────────────
    weights           = discount_weights(discount_rate, N_YEARS)        # 19 factors (SHIPP convention)
    factor            = annuity_factor(discount_rate, N_YEARS)
    capex_initial     = capex(e_cap_nominal, p_cap_MW,
                              e_cost_EUR_per_MWh, p_cost_EUR_per_MW)
    capex_replacement = replacement_cost(e_cap_nominal, p_cap_MW,
                                         repl_e_EUR_per_MWh, repl_p_EUR_per_MW)

    pv_replacements = sum(capex_replacement * weights[yr - 1]
                          for yr in replacement_years)

    # Battery-only (marginal) basis
    pv_rev_bat        = float(np.dot(annual_revenue_bat[:len(weights)], weights))   # discount 19 operating years
    npv_bat_multiyear = -capex_initial + pv_rev_bat - pv_replacements
    npv_bat_no_deg    = -capex_initial + (annual_revenue_bat[0] if annual_revenue_bat else 0.0) * factor

    fd_yr1_total   = annual_fd[0][0] if annual_fd else 0.0
    deg_cost_planB = degradation_cost(fd_yr1_total, e_cap_nominal,
                                      repl_e_EUR_per_MWh, factor)
    npv_bat_planB  = npv_bat_no_deg - deg_cost_planB

    # Arbitrage basis
    pv_rev_arb        = float(np.dot(annual_revenue_arb[:len(weights)], weights))   # discount 19 operating years
    npv_arb_multiyear = -capex_initial + pv_rev_arb - pv_replacements
    npv_arb_no_deg    = -capex_initial + (annual_revenue_arb[0] if annual_revenue_arb else 0.0) * factor
    npv_arb_planB     = npv_arb_no_deg - deg_cost_planB

    # Total-plant basis
    pv_rev_total        = float(np.dot(annual_revenue_total[:len(weights)], weights))   # discount 19 operating years
    npv_total_multiyear = -capex_initial + pv_rev_total - pv_replacements
    npv_total_no_deg    = -capex_initial + (annual_revenue_total[0] if annual_revenue_total else 0.0) * factor

    print(f"{'─'*70}")
    print(f"  Final SoH: {soh*100:.2f}%  |  Replacements: {n_replacements}  |  "
          f"Replacement years: {replacement_years}")
    print(f"\n  ── Timing summary ────────────────────────────────────────────")
    print(f"  Total loop          : {t_loop_total:>8.1f} s  ({t_loop_total/60:.1f} min)")
    print(f"  LP solves (total)   : {t_lp_total:>8.1f} s  ({pct_lp:.0f}% of loop)")
    print(f"  Degradation (total) : {t_degr_total:>8.1f} s  ({pct_degr:.0f}% of loop)")
    print(f"  Per-year average    : {t_loop_total/max(N_YEARS,1):>8.1f} s/yr")
    print(f"\n  ── NPV summary (wp2_econ, EUR, {N_YEARS}-yr horizon, {discount_rate*100:.0f}% discount) ──")
    print(f"  Battery capex (initial)    : {capex_initial*1e-6:>8.2f} MEUR")
    print(f"  Replacement capex          : {capex_replacement*1e-6:>8.2f} MEUR")
    print(f"  PV(replacements)           : {pv_replacements*1e-6:>8.2f} MEUR  "
          f"(yr {replacement_years})")
    print(f"\n  BATTERY-ONLY (marginal) basis:")
    print(f"    PV(battery rev)          : {pv_rev_bat*1e-6:>8.2f} MEUR")
    print(f"    NPV multi-year           : {npv_bat_multiyear*1e-6:>8.2f} MEUR  ← HEADLINE")
    print(f"    NPV no-degradation       : {npv_bat_no_deg*1e-6:>8.2f} MEUR")
    print(f"    NPV Plan B proxy         : {npv_bat_planB*1e-6:>8.2f} MEUR")
    print(f"\n  ARBITRAGE basis (price . storage_p):")
    print(f"    NPV multi-year           : {npv_arb_multiyear*1e-6:>8.2f} MEUR")
    print(f"\n  TOTAL PLANT basis (wind + battery):")
    print(f"    PV(total plant rev)      : {pv_rev_total*1e-6:>8.2f} MEUR")
    print(f"    NPV multi-year           : {npv_total_multiyear*1e-6:>8.2f} MEUR")
    print(f"  ─────────────────────────────────────────────────────────────")

    return {
        # ── Trajectory / degradation ──
        "soh_trajectory":       soh_trajectory,
        "annual_fd":            annual_fd,
        "n_replacements":       n_replacements,
        "replacement_years":    replacement_years,
        "replacement_cost_EUR": n_replacements * capex_replacement,
        "final_soh":            soh,
        "final_soh_pct":        soh * 100.0,
        "e_cap_nominal":        e_cap_nominal,
        "annual_gradient":      annual_gradient,
        "annual_stats":         annual_stats,
        "annual_soc":           annual_soc,
        # ── Per-year revenues ──
        "annual_revenue_bat_eur":   annual_revenue_bat,
        "annual_revenue_total_eur": annual_revenue_total,
        "annual_revenue_arb_eur":   annual_revenue_arb,
        "rev_wind_only_eur":        rev_wind_only,
        # ── Battery-only NPV ──
        "capex_initial_EUR":        capex_initial,
        "pv_rev_bat_EUR":           pv_rev_bat,
        "pv_replacements_EUR":      pv_replacements,
        "npv_bat_multiyear_EUR":    npv_bat_multiyear,
        "npv_bat_no_deg_EUR":       npv_bat_no_deg,
        "npv_bat_planB_EUR":        npv_bat_planB,
        # ── Arbitrage NPV ──
        "pv_rev_arb_EUR":           pv_rev_arb,
        "npv_arb_multiyear_EUR":    npv_arb_multiyear,
        "npv_arb_no_deg_EUR":       npv_arb_no_deg,
        # ── Total-plant NPV ──
        "pv_rev_total_EUR":         pv_rev_total,
        "npv_total_multiyear_EUR":  npv_total_multiyear,
        "npv_total_no_deg_EUR":     npv_total_no_deg,
    }


# =============================================================================
# Console + CSV helpers (updated for EUR and v5.4 NPV fields)
# =============================================================================

def _print_degr_block(degr: Dict, label: str, period_days: float) -> None:
    content = f"  DEGRADATION,  {label}"
    W = max(72, len(content) + 2)
    print("\n" + "╔" + "═" * W + "╗")
    print(f"║{content:<{W}}║")
    print("╚" + "═" * W + "╝")
    print_degradation_report(degr, period_days=period_days, enabled=True)


def _save_multiyear_csv(
    multiyear: Dict,
    run_label: str,
    timestamp: str,
) -> None:
    traj     = multiyear["soh_trajectory"]
    ann      = multiyear["annual_fd"]
    ann_grad = multiyear.get("annual_gradient", [None] * len(ann))
    ann_stat = multiyear.get("annual_stats",    [None] * len(ann))
    ann_bat  = multiyear.get("annual_revenue_bat_eur", [None] * len(ann))
    rows = []
    for i, ((year, soh_pct, fd_cum, n_rep), (fd_yr, fd_cyc, fd_cal)) in enumerate(zip(traj, ann)):
        grad = ann_grad[i]
        stat = ann_stat[i] if i < len(ann_stat) else None
        rows.append({
            "timestamp":             timestamp,
            "run_label":             run_label,
            "year":                  year,
            "soh_pct":               round(soh_pct, 4),
            "fd_annual":             round(fd_yr, 6),
            "fd_cycle":              round(fd_cyc, 6),
            "fd_calendar":           round(fd_cal, 6),
            "fd_calendar_pct":       round(100.0 * fd_cal / max(fd_yr, 1e-30), 1),
            "fd_cumulative":         round(fd_cum, 6),
            "n_replacements":        n_rep,
            "replacement_this_year": year in multiyear["replacement_years"],
            "rev_bat_kEUR":          round(ann_bat[i] * 1e-3, 2) if ann_bat[i] is not None else None,
            "dDeg_dDoD_DEPRECATED":  round(grad[0], 6) if grad else None,
            "mean_abs_subgrad":      round(grad[1], 6) if grad else None,
            "mean_dod":              round(stat[0], 4) if stat else (round(grad[2], 4) if grad else None),
            "mean_abs_dual":         round(grad[3], 6) if grad else None,
            "e_cap_eff":             round(stat[1], 4) if stat else (round(grad[4], 4) if grad else None),
            "frac_single_straddle":  round(grad[5], 4) if grad else None,
            "dRev_dEcap":            round(grad[7], 4) if grad else None,
            "dDegCost_dEcap":        round(grad[8], 6) if grad else None,
            "lambda_E":              round(grad[9], 4) if grad else None,
            "dNPV_dEcap":            round(grad[10], 4) if grad else None,
            "rc_e_cap1":             round(grad[11], 4) if grad else None,
        })

    csv_path = RESULTS_DIR / f"multiyear_trajectory_{run_label}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  ✓ CSV: {csv_path.name}")


def _save_gradient_timeseries_csv(
    multiyear: Dict,
    run_label: str,
    timestamp: str,
) -> None:
    ann_grad = multiyear.get("annual_gradient")
    if not ann_grad or all(g is None for g in ann_grad):
        print("  ⚠ No gradient time series to save, duals unavailable.")
        return

    traj = multiyear["soh_trajectory"]
    rows = []
    for (year, soh_pct, fd_cum, _n_rep), grad in zip(traj, ann_grad):
        if grad is None:
            continue
        subgrad = grad[6]
        for t, val in enumerate(subgrad):
            rows.append({
                "timestamp":        timestamp,
                "run_label":        run_label,
                "year":             year,
                "timestep":         t,
                "soh_pct":          round(soh_pct, 4),
                "subgrad_combined": round(float(val), 8),
            })

    csv_path = RESULTS_DIR / f"gradient_timeseries_{run_label}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  ✓ CSV: {csv_path.name}  "
          f"({len(rows):,} rows, {len(ann_grad)} yr x {len(ann_grad[0][6])} timesteps)")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 80)
    print(f"WP2 BATTERY OPTIMIZATION + DEGRADATION  {FILE_TAG} (single window, wp2_econ)")
    print(f"cycle branch: {DEG_MODEL} | Xu calendar | multi-year | replacement | per-year NPV")
    print(f"DEG_MODEL={DEG_MODEL}  |  N_YEARS={N_YEARS}  |  r={discount_rate*100:.0f}%")
    print(f"Output folder: {RESULTS_DIR}")
    print("=" * 80)

    # ── 1. Load config + raw series ───────────────────────────────────────
    print("\n[1/6] Loading WP2 configuration...")
    setup = quick_setup(HPP_YAML, config={"interp_n": 2000}, verbose=False)
    hpp   = setup["hpp"]

    ws_all, wd_all, ti_all = _load_inputs(hpp)
    price_all               = _load_prices()

    # ── 2. Choose horizon ─────────────────────────────────────────────────
    print("\n[2/6] Loading electricity prices...")
    n = _choose_horizon(len(ws_all), len(price_all))
    ws        = ws_all[:n]
    wd        = wd_all[:n]
    ti        = ti_all[:n] if ti_all is not None else None
    price_eur = price_all[:n]

    print(f"  Horizon: {n:,} h ({n/24:.1f} days)")
    print(f"  Mean price: {float(np.mean(price_eur)):.2f} EUR/MWh")

    if n != 8760:
        print(f"  ℹ  Window is {n} h ({n/24:.0f} days), not a full year. "
              f"Degradation & revenue will be annualised x{8760/n:.3f} per project year.")

    # ── 3. PyWake ─────────────────────────────────────────────────────────
    print("\n[3/6] Running PyWake time-series simulation...")
    power_wind_MW = _run_pywake_power_MW(setup, wd, ws, ti)
    print(f"  Wind mean: {float(np.mean(power_wind_MW)):.1f} MW | "
          f"peak: {float(np.max(power_wind_MW)):.1f} MW")

    # ── 4. SHIPP (single-year baseline) ───────────────────────────────────
    print("\n[4/6] Running SHIPP optimization (single-year baseline)...")
    p_max_MW = float(hpp["grid_connection_capacity"]) / 1e6

    (stor, stor_null, e_cap_MWh, p_cap_MW, rte_ac, eta,
     e_cost_EUR_per_MWh, p_cost_EUR_per_MW,
     repl_e_EUR_per_MWh, repl_p_EUR_per_MW,
     soc_min, soc_max) = _build_shipp_components(setup, p_max_MW)

    shi_fit = fit_shi_polynomial(soc_min, soc_max, verbose=True)

    print(f"  Battery: {p_cap_MW:.0f} MW / {e_cap_MWh:.0f} MWh | "
          f"Grid: {p_max_MW:.0f} MW | RTE(ac): {rte_ac*100:.1f}% | eta: {eta:.4f}")
    print(f"  Capex: {e_cost_EUR_per_MWh/1000:.0f} EUR/kWh + {p_cost_EUR_per_MW/1000:.0f} EUR/kW")
    print(f"  Repl : {repl_e_EUR_per_MWh/1000:.0f} EUR/kWh + {repl_p_EUR_per_MW/1000:.0f} EUR/kW")
    print(f"  Solver: {pyo_solver}")

    run_label = _build_run_label(run_ts, PRICE_CSV, p_cap_MW, e_cap_MWh, soc_min, soc_max, rte_ac=rte_ac)

    os, os_fixed = _solve_shipp(price_eur, power_wind_MW, stor, stor_null, p_max_MW, n, soc_max)

    np.save(RESULTS_DIR / "storage_e_fixed.npy",
            np.array(os_fixed.storage_e[0].data))
    np.save(RESULTS_DIR / "e_cap_fixed.npy",
            np.array([os_fixed.storage_list[0].e_cap]))

    # Baseline metrics (single-year, informational; NPV superseded by multi-year)
    wind_no_bat       = np.minimum(power_wind_MW, p_max_MW)
    revenues_res_only = (365.0 * 24.0 / n * np.dot(price_eur, wind_no_bat) * dt)
    os.get_added_npv(discount_rate, N_YEARS)
    os_fixed.get_added_npv(discount_rate, N_YEARS)
    period_days = n * dt / 24.0

    def _rev_inc_pct(rev: float) -> float:
        return 100.0 * (rev / revenues_res_only - 1.0) if revenues_res_only > 0 else 0.0

    cycles_opt   = count_equivalent_full_cycles(
        os.storage_p[0].data, os.storage_e[0].data,
        os.storage_list[0].e_cap, dt_hours=dt
    )
    cycles_fixed = count_equivalent_full_cycles(
        os_fixed.storage_p[0].data, os_fixed.storage_e[0].data,
        os_fixed.storage_list[0].e_cap, dt_hours=dt
    )

    if print_baseline_table:
        print("\n" + "=" * 80)
        print("SINGLE-YEAR BASELINE (LP solve; NPV superseded by multi-year below)")
        print("=" * 80)
        print("                Revenue [kEUR]  Rev.inc%    p/e_cap [MW/MWh]    "
              "NPV [MEUR]  Cycles/yr")
        print("-" * 88)
        print(
            f"Sizing Opt.   {os.annual_revenue*1e-3:>10.1f}  "
            f"{_rev_inc_pct(os.annual_revenue):>8.2f}%  "
            f"{os.storage_list[0].p_cap:>7.1f}/{os.storage_list[0].e_cap:<7.1f}  "
            f"{os.npv:>10.2f}  {cycles_opt/period_days*365:>9.0f}"
        )
        print(
            f"Dispatch-fix  {os_fixed.annual_revenue*1e-3:>10.1f}  "
            f"{_rev_inc_pct(os_fixed.annual_revenue):>8.2f}%  "
            f"{os_fixed.storage_list[0].p_cap:>7.1f}/{os_fixed.storage_list[0].e_cap:<7.1f}  "
            f"{os_fixed.npv:>10.2f}  {cycles_fixed/period_days*365:>9.0f}"
        )
        print("-" * 88)
        print("  Note: os.npv / os_fixed.npv above use the SHIPP kernel's own NPV method.")
        print("  The multi-year section below replaces these with per-year discounting.")

    # ── 5. Single-year degradation ────────────────────────────────────────
    print(f"\n[5/6] Single-year degradation analysis (cycle branch: {DEG_MODEL})...")
    bat_params = setup["battery"]

    def _degr_single_year(storage_p, storage_e, e_cap):
        """Return (active_branch_dict, shi_dict) for one year's dispatch.

        The Shi dict is computed on both branches. It is the input the single-year diagnostic plot accepts, and keeping it makes the
        Shi-against-Xu cycle difference available without a second run.
        """
        d_shi = _shi_with_calendar_correction(
            storage_p, storage_e, e_cap,
            bat_params, shi_fit, 25.0, dt, eol_thresholds,
        )
        if DEG_MODEL != "xu":
            return d_shi, d_shi
        d_xu = _xu_full_degradation_report(
            storage_p, storage_e, e_cap, 25.0, dt, eol_thresholds, bat_params,
        )
        return d_xu, d_shi

    degr_fixed, degr_fixed_shi = _degr_single_year(
        os_fixed.storage_p[0].data,
        os_fixed.storage_e[0].data,
        float(os_fixed.storage_list[0].e_cap),
    )

    opt_e_cap  = float(os.storage_list[0].e_cap)
    no_battery = opt_e_cap < 1.0
    if no_battery:
        print(
            f"\n  ⚠  Sizing optimizer chose no battery "
            f"(e_cap = {opt_e_cap:.3f} MWh < 1 MWh).\n"
            f"     Not economically viable at this price level. Degradation skipped."
        )
        degr_opt     = None
        degr_opt_shi = None
    else:
        degr_opt, degr_opt_shi = _degr_single_year(
            os.storage_p[0].data,
            os.storage_e[0].data,
            opt_e_cap,
        )

    if print_degr_reports:
        # print_degradation_report expects the Shi dict, so the console block
        # stays on that branch. The Xu numbers are printed separately below.
        _print_degr_block(degr_fixed_shi, "DISPATCH-FIXED,  150 MW / 300 MWh", period_days)
        if degr_opt_shi is not None:
            _print_degr_block(degr_opt_shi, "SIZING OPT,  optimal capacity", period_days)
        else:
            print("\n  [Sizing-opt degradation skipped, no battery installed]")

        if DEG_MODEL == "xu":
            print("\n" + "=" * 60)
            print("XU BRANCH,  single-year reporting values")
            print("=" * 60)
            for name, d_xu, d_shi in [("dispatch_fixed", degr_fixed, degr_fixed_shi),
                                      ("sizing_opt",     degr_opt,   degr_opt_shi)]:
                if d_xu is None:
                    continue
                under = 100.0 * (1.0 - d_shi["fd_cycle"] / max(d_xu["fd_cycle"], 1e-30))
                print(f"  {name}: fd={d_xu['fd']:.6f}  cycle={d_xu['fd_cycle']:.6f}  "
                      f"calendar={d_xu['fd_calendar']:.6f}  SoH={d_xu['soh']:.3f}%")
                print(f"    Shi under-reports the cycle term by {under:.1f}% "
                      f"({d_shi['fd_cycle']:.6f} against {d_xu['fd_cycle']:.6f}); "
                      f"the calendar term differs by "
                      f"{abs(d_shi['fd_calendar'] - d_xu['fd_calendar']):.2e}.")
            print("=" * 60)

    # Gap D verification (dual extraction check)
    if os_fixed.dual_prices is not None:
        print("\n" + "=" * 60)
        print("GAP D VERIFICATION,  dual prices + gradient signal")
        print("=" * 60)
        dual  = os_fixed.dual_prices["dual_e_min1"]
        e_cap = os_fixed.dual_prices["e_cap1"]
        print(f"  dual_e_min1: min={dual.min():.4e}  max={dual.max():.4e}  "
              f"mean={dual.mean():.4e}")
        print(f"  n_nonzero: {np.sum(dual != 0)} / {len(dual)}")
        cycles = rainflow_cycle_counting(os_fixed.storage_e[0].data, e_cap)
        sg = compute_subgradient(
            storage_e=os_fixed.storage_e[0].data,
            cycles=cycles,
            dt_hours=dt,
            battery_replacement_cost_per_MWh=repl_e_EUR_per_MWh,
            eff_in=eta,
            eff_out=eta,
            shi_fit=shi_fit,
        )
        factor        = npf.npv(discount_rate, np.ones(N_YEARS)) - 1
        dual_per_year = dual / factor
        grad_deg_dod  = -e_cap * float(np.dot(sg["subgrad_combined"], dual_per_year))
        print(f"  subgrad_combined: [{sg['subgrad_combined'].min():.4e}, "
              f"{sg['subgrad_combined'].max():.4e}]")
        print(f"  dDeg/dDoD (chain rule): {grad_deg_dod:.6e}")
        print("=" * 60)
    else:
        print("\n  WARNING: dual_prices is None, dual extraction failed.")

    # ── 6. Multi-year loop ────────────────────────────────────────────────
    print(f"\n[6/6] Lifetime degradation loop "
          f"({N_YEARS} yr x {n/24:.0f}-day window, annualised x{8760/n:.3f})...")

    wind_tile  = power_wind_MW[:8760] if n > 8760 else power_wind_MW
    price_tile = price_eur[:8760]     if n > 8760 else price_eur

    multiyear = _run_multiyear(
        wind_8760=wind_tile,
        price_8760=price_tile,
        stor_null=stor_null,
        p_max_MW=p_max_MW,
        e_cap_nominal=e_cap_MWh,
        p_cap_MW=p_cap_MW,
        rte_ac=rte_ac,
        e_cost_EUR_per_MWh=e_cost_EUR_per_MWh,
        p_cost_EUR_per_MW=p_cost_EUR_per_MW,
        repl_e_EUR_per_MWh=repl_e_EUR_per_MWh,
        repl_p_EUR_per_MW=repl_p_EUR_per_MW,
        bat_params=bat_params,
        shi_fit=shi_fit,
        soc_min=soc_min,
        soc_max=soc_max,
        T_cell_C=25.0,
        deg_model=DEG_MODEL,
    )

    # ── Save CSV ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if SAVE_CSV:
        # Single-year baseline
        base_csv  = RESULTS_DIR / f"battery_optimization_results_{run_label}.csv"
        base_rows = [
            {
                "timestamp":            timestamp,
                "hours_simulated":      n,
                "days_simulated":       period_days,
                "optimization_type":    "sizing",
                "revenue_kEUR":         os.annual_revenue * 1e-3,
                "revenue_increase_pct": _rev_inc_pct(os.annual_revenue),
                "p_cap_MW":             os.storage_list[0].p_cap,
                "e_cap_MWh":            os.storage_list[0].e_cap,
                "npv_MEUR":             os.npv,
                "cycles_per_year":      cycles_opt / period_days * 365.0,
                "solver":               pyo_solver,
                "no_battery":           no_battery,
            },
            {
                "timestamp":            timestamp,
                "hours_simulated":      n,
                "days_simulated":       period_days,
                "optimization_type":    "dispatch_fixed",
                "revenue_kEUR":         os_fixed.annual_revenue * 1e-3,
                "revenue_increase_pct": _rev_inc_pct(os_fixed.annual_revenue),
                "p_cap_MW":             os_fixed.storage_list[0].p_cap,
                "e_cap_MWh":            os_fixed.storage_list[0].e_cap,
                "npv_MEUR":             os_fixed.npv,
                "cycles_per_year":      cycles_fixed / period_days * 365.0,
                "solver":               pyo_solver,
                "no_battery":           False,
            },
        ]
        pd.DataFrame(base_rows).to_csv(
            base_csv, mode="a", header=not base_csv.exists(), index=False
        )

        # Single-year degradation
        degr_csv = RESULTS_DIR / f"battery_degradation_results_{run_label}.csv"

        def _degr_row(d: Dict, case: str) -> Dict:
            stats = d.get("shi_cycle_stats", d.get("xu_cycle_stats", {}))
            return {
                "timestamp":           timestamp,
                "case":                case,
                "model":               d["meta"].get("model", "Shi2018"),
                "calendar_correction": d["meta"].get("calendar_correction", ""),
                "fd_total":            d["fd"],
                "fd_cycle":            d["fd_cycle"],
                "fd_calendar":         d["fd_calendar"],
                "fd_calendar_pct":     100.0 * d["fd_calendar"] / max(d["fd"], 1e-30),
                "soh_pct":             d["soh"],
                "capacity_fade_pct":   d["capacity_fade_percent"],
                "e_cap_degraded_MWh":  d["e_cap_degraded"],
                "total_cycles_efc":    d["total_cycles"],
                "n_rainflow_cycles":   stats.get("n_rainflow_cycles", 0),
                "mean_dod_pct":        stats.get("mean_dod", 0) * 100,
                "mean_soc_pct":        stats.get("mean_soc", 0) * 100,
                "mean_soc_time_avg_pct": 100.0 * ( d.get("mean_soc_time_avg", stats.get("mean_soc_time_avg", 0.0))),                
                "eol_80_yr":           d["eol_years"].get(0.80),
                "eol_70_yr":           d["eol_years"].get(0.70),
                "eol_60_yr":           d["eol_years"].get(0.60),
            }

        degr_rows = [_degr_row(degr_fixed, "dispatch_fixed")]
        if degr_opt is not None:
            degr_rows.append(_degr_row(degr_opt, "sizing_opt"))
        if DEG_MODEL == "xu":
            # Keep the Shi values alongside, so the surrogate error is archived
            # with the run rather than recovered from a separate Shi run.
            degr_rows.append(_degr_row(degr_fixed_shi, "dispatch_fixed_shi_reference"))
            if degr_opt_shi is not None:
                degr_rows.append(_degr_row(degr_opt_shi, "sizing_opt_shi_reference"))
        pd.DataFrame(degr_rows).to_csv(
            degr_csv, mode="a", header=not degr_csv.exists(), index=False
        )

        # Multi-year NPV summary row
        npv_csv  = RESULTS_DIR / f"npv_summary_{run_label}.csv"
        npv_rows = [{
            "timestamp":                  timestamp,
            "run_label":                  run_label,
            "deg_model":                  DEG_MODEL,
            "N_YEARS":                    N_YEARS,
            "discount_rate":              discount_rate,
            "rte_dc_yaml":                bat_params["rte_nominal"],
            "pcu_yaml":                   bat_params["pcu_efficiency"],
            "rte_ac":                     rte_ac,
            "eta_oneway":                 eta,
            "e_cap_MWh":                  e_cap_MWh,
            "p_cap_MW":                   p_cap_MW,
            "soc_min":                    soc_min,
            "soc_max":                    soc_max,
            "capex_initial_MEUR":         multiyear["capex_initial_EUR"] * 1e-6,
            "pv_replacements_MEUR":       multiyear["pv_replacements_EUR"] * 1e-6,
            "n_replacements":             multiyear["n_replacements"],
            "replacement_years":          str(multiyear["replacement_years"]),
            "npv_bat_multiyear_MEUR":     multiyear["npv_bat_multiyear_EUR"] * 1e-6,
            "npv_bat_no_deg_MEUR":        multiyear["npv_bat_no_deg_EUR"] * 1e-6,
            "npv_bat_planB_MEUR":         multiyear["npv_bat_planB_EUR"] * 1e-6,
            "npv_arb_multiyear_MEUR":     multiyear["npv_arb_multiyear_EUR"] * 1e-6,
            "npv_total_multiyear_MEUR":   multiyear["npv_total_multiyear_EUR"] * 1e-6,
            "final_soh_pct":              multiyear["final_soh_pct"],
            "headline_basis":             HEADLINE_BASIS,
        }]
        pd.DataFrame(npv_rows).to_csv(
            npv_csv, mode="a", header=not npv_csv.exists(), index=False
        )

        print(f"\n  ✓ CSV: {base_csv.name}")
        print(f"  ✓ CSV: {degr_csv.name}")
        print(f"  ✓ CSV: {npv_csv.name}")

        _save_multiyear_csv(multiyear, run_label, timestamp)
        _save_gradient_timeseries_csv(multiyear, run_label, timestamp)

    # ── Save text report ──────────────────────────────────────────────────
    if SAVE_REPORT:
        report_path = RESULTS_DIR / f"degradation_report_{run_label}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"WP2 BATTERY — {FILE_TAG} | cycle branch: {DEG_MODEL} | Xu calendar | multi-year NPV (EUR)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated       : {timestamp}\n")
            f.write(f"Dataset         : {PRICE_CSV.name}\n")
            f.write(f"Horizon         : {n:,} h  ({period_days:.1f} days)\n")
            f.write(f"Solver          : {pyo_solver}\n")
            f.write(f"deg_model       : {DEG_MODEL}\n")
            f.write(f"N_YEARS         : {N_YEARS}\n")
            f.write(f"discount_rate   : {discount_rate*100:.0f}%\n")
            f.write(f"eta (symmetric) : {eta:.4f}  (rte_ac={rte_ac:.4f})\n")
            f.write(f"SoC window      : {soc_min*100:.0f}–{soc_max*100:.0f}%\n")
            f.write(f"Shi polynomial  : k3={shi_fit.k3:.4e}  k4={shi_fit.k4:.4f}  "
                    f"R2={shi_fit.r2:.4f}\n\n")

            for case_name, d in [("dispatch_fixed", degr_fixed),
                                  ("sizing_opt",     degr_opt)]:
                if d is None:
                    continue
                f.write(f"DEGRADATION ({case_name})\n")
                f.write("-" * 60 + "\n")
                f.write(f"fd_total      : {d['fd']:.6f}\n")
                f.write(f"  fd_cycle    : {d['fd_cycle']:.6f}  "
                        f"({100*d['fd_cycle']/max(d['fd'],1e-30):.0f}%)\n")
                f.write(f"  fd_calendar : {d['fd_calendar']:.6f}  "
                        f"({100*d['fd_calendar']/max(d['fd'],1e-30):.0f}%)\n")
                f.write(f"SoH           : {d['soh']:.3f}%\n")
                for thr, yr in d["eol_years"].items():
                    f.write(f"EoL {thr*100:.0f}%      : {yr} yr\n")
                f.write("\n")

            f.write("MULTI-YEAR NPV SUMMARY\n" + "-" * 60 + "\n")
            f.write(f"N_years              : {N_YEARS}\n")
            f.write(f"EoL replacement      : SoH < " f"{(EOL_REPLACEMENT + EOL_REPLACEMENT_TOL)*100:.1f}%  "f"({EOL_REPLACEMENT*100:.0f}% + {EOL_REPLACEMENT_TOL*100:.1f}% tolerance)\n")
            f.write(f"Replacements         : {multiyear['n_replacements']}\n")
            f.write(f"Replacement years    : {multiyear['replacement_years']}\n")
            f.write(f"Capex (initial)      : "
                    f"{multiyear['capex_initial_EUR']*1e-6:.2f} MEUR\n")
            f.write(f"PV(replacements)     : "
                    f"{multiyear['pv_replacements_EUR']*1e-6:.2f} MEUR\n")
            f.write(f"NPV bat (multi-year) : "
                    f"{multiyear['npv_bat_multiyear_EUR']*1e-6:.2f} MEUR  ← HEADLINE\n")
            f.write(f"NPV bat (no-deg)     : "
                    f"{multiyear['npv_bat_no_deg_EUR']*1e-6:.2f} MEUR\n")
            f.write(f"NPV bat (Plan B)     : "
                    f"{multiyear['npv_bat_planB_EUR']*1e-6:.2f} MEUR\n")
            f.write(f"NPV arb (multi-year) : "
                    f"{multiyear['npv_arb_multiyear_EUR']*1e-6:.2f} MEUR\n")
            f.write(f"NPV total (multi-yr) : "
                    f"{multiyear['npv_total_multiyear_EUR']*1e-6:.2f} MEUR\n")
            f.write(f"Final SoH            : {multiyear['final_soh_pct']:.2f}%\n\n")

            f.write(f"{'Year':>5}  {'SoH_%':>8}  {'fd_annual':>10}  "
                    f"{'fd_cumul':>10}  {'dNPV/dEcap':>14}  {'mean_DoD%':>10}  "
                    f"{'rev_kEUR':>10}  {'replaced':>8}\n")
            f.write("-" * 80 + "\n")
            ann_fd   = multiyear["annual_fd"]
            ann_grad = multiyear.get("annual_gradient", [None] * len(ann_fd))
            ann_stat = multiyear.get("annual_stats",    [None] * len(ann_fd))
            ann_bat  = multiyear.get("annual_revenue_bat_eur", [None] * len(ann_fd))
            for (yr, soh_p, fd_c, _n_rep), (fd_y, fd_cyc, fd_cal), grad, stat, rev in zip(
                multiyear["soh_trajectory"], ann_fd, ann_grad, ann_stat, ann_bat
            ):
                replaced = "YES" if yr in multiyear["replacement_years"] else ""
                grad_str = f"{grad[10]:+.4e}" if grad is not None else "      N/A"
                dod_str  = f"{stat[0]*100:.1f}" if stat is not None else "   N/A"
                rev_str  = f"{rev*1e-3:.1f}"      if rev  is not None else "   N/A"
                f.write(f"{yr:>5d}  {soh_p:>8.3f}  {fd_y:>10.6f}  "
                        f"{fd_c:>10.6f}  {grad_str:>12}  {dod_str:>10}  "
                        f"{rev_str:>10}  {replaced:>8}\n")

            if DEG_MODEL == "xu":
                f.write("\nNote: the outer gradient is assembled from the Shi sub-gradient, so dNPV/dEcap is not computed on this "
                        "branch. The mean_DoD column is a rainflow statistic and is valid on both branches.\n")

        print(f"  ✓ Report: {report_path.name}")

    # ── Save slim multiyear dict for thesis_plots_results.py ──────────────
    _slim = {k: v for k, v in multiyear.items() if k != 'annual_gradient'}
    _slim['annual_gradient'] = [
        tuple(g[:6]) if (g is not None and len(g) >= 6) else g
        for g in multiyear.get('annual_gradient', [])
    ]
    np.save(RESULTS_DIR / f'multiyear_{run_label}.npy', _slim, allow_pickle=True)
    print(f'  ✓ multiyear_{run_label}.npy saved')
    # ── Plots ─────────────────────────────────────────────────────────────
    if MAKE_PLOT:
        time_vec = np.arange(n) * dt / 24.0

        # Power export + SoC overview
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(time_vec, power_wind_MW + os_fixed.storage_p[0].data,
                   linewidth=0.7, label="Wind + battery export")
        ax[0].plot(time_vec, power_wind_MW,
                   linewidth=0.7, alpha=0.6, label="Wind only")
        ax[0].axhline(p_max_MW, linestyle="--", alpha=0.5,
                      label=f"Grid limit ({p_max_MW:.0f} MW)")
        ax[0].set_xlabel("Time [days]"); ax[0].set_ylabel("Power [MW]")
        ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)
        ax[1].plot(time_vec, os_fixed.storage_e[0].data, linewidth=0.7)
        ax[1].set_xlabel("Time [days]"); ax[1].set_ylabel("SoC [MWh]")
        ax[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"battery_baseline_results_{run_label}.png", dpi=200)

        # Single-year degradation detail
        plot_degradation_analysis(
            degr_fixed_shi,
            storage_e=os_fixed.storage_e[0].data,
            time_vec=time_vec,
            save_path=str(PLOTS_DIR / f"battery_degradation_analysis_fixed_{run_label}.png"),
            show=False, verbose=True, eol_thresholds=eol_thresholds,
        )
        if degr_opt is not None:
            plot_degradation_analysis(
                degr_opt,
                storage_e=os.storage_e[0].data,
                time_vec=time_vec,
                save_path=str(PLOTS_DIR / f"battery_degradation_analysis_sizing_{run_label}.png"),
                show=False, verbose=True, eol_thresholds=eol_thresholds,
            )

        # Multi-year trajectory + gradient plots
        plot_multiyear_trajectory(multiyear, run_label,
            plots_dir=PLOTS_DIR, n_years=N_YEARS,
            eol_replacement=EOL_REPLACEMENT, show=show_plots)
        plot_gradient_analysis(multiyear, run_label,
            plots_dir=PLOTS_DIR, n_years=N_YEARS, show=show_plots)
        plot_subgradient_timeseries(multiyear, run_label,
            plots_dir=PLOTS_DIR, show=show_plots)

        if show_plots:
            plt.show()
        else:
            plt.close("all")

    print("\n" + "=" * 80)
    print(f"✓ COMPLETE — {FILE_TAG} (single window | wp2_econ | EUR | per-year NPV)")
    print("=" * 80)


if __name__ == "__main__":
    main()
