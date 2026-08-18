"""Window sweep: net present value over the SoC operating window at fixed size.

Fork origin: run_battery_xu_shi_degradation_v5.4 (+Xu comp_short).py

Holds battery energy and power capacity fixed and sweeps the state-of-charge operating window, parametrised by centre and width:
    soc_min = centre - width/2,   soc_max = centre + width/2

Two series are run. The width series, at fixed centre, isolates the cycling term, which depends on cycle depth. The centre series, at fixed width, isolates
the calendar term, which depends on mean state of charge. Capacity sizing is handled separately by scripts/run_sizing_sweep.py.

Degradation
    Reported degradation is Shi cycling with the Xu calendar term. The convex polynomial is refitted for each window, since the fit range depends on the
    reachable cycle depth. Per window the script records the fit R-squared and the weighted fraction of observed cycles below delta = 0.1437, which is
    where the Xu stress function stops being convex and the polynomial is extrapolating. That fraction is the fit-trust diagnostic for narrow windows.

Kernel coupling
    The energy floor is e_cap * (1 - dod) with dod = 1 - soc_min; the ceiling is soc_max1 = soc_max. Width and (1 - soc_min) are distinct quantities: width
    is the reported usable depth, dod is the floor parameter.

Multi-year NPV
    Each year re-solves the LP at capacity faded to e_cap_nominal * SoH. Per-year undiscounted revenue is discounted individually, and results are
    reported on battery-only (marginal), arbitrage, and total-plant bases.

Usage:
    python scripts/run_window_sweep.py

Author: Thodoris Tsonopoulos, MSc thesis, TU Delft Wind Energy
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import time
import json

import numpy as np
import pandas as pd

from shipp.kernel_pyomo import solve_lp_pyomo
from shipp.components import Storage, Production, TimeSeries

from degradation.xu import (
    count_equivalent_full_cycles,
    rainflow_cycle_counting,
    ft_calendar,
    sei_capacity_loss,
    fc_cycle,                 # Xu per-cycle stress  S_delta * S_soc * S_temp
    phi_shi_with_stress,      # fitted Shi per-cycle  k3*delta^k4 * S_soc * S_temp
    compute_fd,               # full Xu fd: (fd, fd_cycle, fd_calendar)
    fit_shi_polynomial,       # was re-exported by degradation_subgradient before
                              # that module was rewritten; imported from its
                              # defining module now. degradation_shi has an
                              # identical copy; this one matches the
                              # ShiPolynomialFit that compute_subgradient expects.
)
from degradation.shi import analyze_degradation_shi, phi_shi_prime_with_stress
from degradation.subgradient import compute_subgradient
from degradation.economics import (eta_symmetric, capex, replacement_cost, annuity_factor,
                      discount_weights, degradation_cost, revenue_annual, HEADLINE_BASIS)

import xarray as xr
from py_wake.site import XRSite
import numpy_financial as npf

from degradation.site import quick_setup, get_wake_model

# =============================================================================
# CONFIG
# =============================================================================

from degradation.paths import (
    HPP_YAML,
    PRICE_CSV_2022,
    results_dir as make_results_dir,
)

PRICE_CSV   = PRICE_CSV_2022        # switch to PRICE_CSV_2019 for the 2019 price year
RESULTS_DIR = make_results_dir("window_sweep")

discount_rate = 0.03
dt            = 1.0

# Solver
pyo_solver = "gurobi"

# Horizon
RUN_FULL_YEAR     = True
N_DAYS_TEST       = 30
MAX_HOURS_SPARSE  = 180 * 24

# Problem
p_min      = 0.0
WAKE_MODEL = "Bastankhah"

# EoL
eol_thresholds   = [0.80, 0.70, 0.60]
N_YEARS          = 20
EOL_REPLACEMENT  = 0.70
EOL_REPLACEMENT_TOL = 0.005

# ── Sweep verbosity ───────────────────────────────────────
# False -> one-line summary per DoD window; True -> full per-year detail. True is required to capture the per-year SoH trajectory in the log, which is
# what the replacement-year threshold check in the results chapter needs.
SWEEP_VERBOSE = True

run_ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
FILE_TAG = "window"

# ── v5.4 DoD sweep configuration ─────────────────────────────
# (E, P) FIXED for the DoD study, as an OVERWRITE of the YAML read. None on either falls back to the YAML value. All (E, P) exploration happens in Plan B.
E_CAP_FIXED_MWh = 550.0     # Xu grid optimum from the refined Plan B pass (sweep 20260704_014612); quadratic vertex E*=560
P_CAP_FIXED_MW  = 175.0     # same sweep; quadratic vertex P*=182 (Xu) and 183 (Shi)

# Two controlled series, parametrized by (center, width): soc_min = center - width/2 ; soc_max = center + width/2
# WIDTH series  isolates the CYCLING term (Shi/Phi, convex in delta).
# CENTER series moves the mean-SoC stress S_sigma, which multiplies BOTH the cycle and the calendar term, so it scales the two together rather than
# isolating the calendar term.
DOD_WIDTH_SERIES_CENTER = 0.50
DOD_WIDTH_SERIES        = [0.40, 0.60, 0.80, 1.00]        # 30-70, 20-80, 10-90, 0-100
DOD_CENTER_SERIES_WIDTH = 0.80
DOD_CENTER_SERIES       = [0.60, 0.55, 0.50, 0.45, 0.40]  # 20-100,15-95,10-90,5-85,0-80
# Width floored at 0.40 (0.00 raises ValueError in the fit; 0.20 is suspect).

# Xu LMO S_delta loses convexity below this; fit_lo=0.15 sits just above it.
# Used to measure how much observed cycle weight is valued by an extrapolated Phi.
LMO_NONCONVEX_DELTA = 0.1437

# Named reference windows, always solved and labeled for presentation; they bracket the
# study (manufacturer cell-protection envelope and the unconstrained extreme). Overlaps
# with the parametric series are de-duped (solved once) and inherit the label.
DOD_REFERENCE_WINDOWS = [
    ("manufacturer", 0.10, 0.90),   # 10-90 envelope
    ("extreme",      0.00, 1.00),   # 0-100 upper bound
]

# Full-model comparison. When True, every window is ALSO evaluated end-to-end under the
# pure Xu model (Xu cycling + Xu calendar driving SoH, replacement, NPV), alongside the
# fitted-Shi result. This re-solves the 20-year dispatch under the Xu fade, so it roughly
# DOUBLES runtime. False = Shi-only fast run. Mirrors the Plan B Xu-vs-Shi comparison.
COMPARE_XU_MULTIYEAR = True

# =============================================================================
# Console logging — tee all output to a file
# =============================================================================

class TeeLogger:
    """Duplicates stdout/stderr to a log file while preserving console output."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._file = open(log_path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def start(self):
        sys.stdout = self
        sys.stderr = _TeeStream(self._stderr, self._file)
        return self

    def write(self, text):
        self._stdout.write(text)
        self._file.write(text)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def stop(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._file.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()


class _TeeStream:
    """Helper for stderr tee."""
    def __init__(self, original, log_file):
        self._orig = original
        self._file = log_file

    def write(self, text):
        self._orig.write(text)
        self._file.write(text)

    def flush(self):
        self._orig.flush()
        self._file.flush()


# =============================================================================
# Data loading helpers  (from v5.1, unchanged)
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


def _run_pywake_power_MW(setup, wd, ws, ti) -> np.ndarray:
    n    = len(ws)
    site = XRSite(ds=xr.Dataset(data_vars=dict(P=1)))
    wf_model = get_wake_model(WAKE_MODEL, site, setup["windturbine"])
    time_days = np.arange(n) / 24.0
    kwargs = {"x": setup["x"], "y": setup["y"], "wd": wd, "ws": ws, "time": time_days}
    if ti is not None:
        kwargs["TI"] = ti
    return wf_model(**kwargs).Power.sum(["wt"]).values / 1e6


# =============================================================================
# Core helpers  (from v5.1)
# =============================================================================

def _build_storage_year(
    e_cap_eff, p_cap_MW, rte_ac, e_cost_EUR_per_MWh, soc_min, soc_max
) -> Storage:
    dod_eff = 1.0 - soc_min if pyo_solver != "none" else 1.0
    eta = eta_symmetric(rte_ac)   # symmetric split: eff_in = eff_out = sqrt(rte_ac) = 0.9367
    return Storage(
        e_cap=e_cap_eff, p_cap=p_cap_MW, eff_in=eta, eff_out=eta,
        e_cost=e_cost_EUR_per_MWh, p_cost=0.0, dod=dod_eff,
    )


def _shi_with_calendar_correction(
    storage_p, storage_e, e_cap_eff, bat_params, shi_fit,
    T_cell_C, dt_hours, eol_thresholds,
) -> Dict:
    """Shi cycle degradation + Xu calendar term (reporting path only)."""
    n_steps         = len(storage_e)
    t_total_seconds = n_steps * dt_hours * 3600.0
    t_total_hours   = t_total_seconds / 3600.0
    e_arr           = np.asarray(storage_e, dtype=float)
    sigma_mean      = float(np.mean(e_arr)) / max(e_cap_eff, 1e-9)

    degr = analyze_degradation_shi(
        storage_p, storage_e, e_cap_eff, bat_params,
        shi_fit=shi_fit, T_cell_C=T_cell_C,
        dt_hours=dt_hours, eol_thresholds=eol_thresholds,
    )

    fd_cal       = ft_calendar(t_total_seconds, sigma_mean, T_cell_C)
    fd_corrected = degr["fd_shi"] + fd_cal
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

    degr["fd_calendar"]        = float(fd_cal)
    degr["fd"]                 = float(fd_corrected)
    degr["fd_cycle"]           = degr["fd_shi"]
    degr["capacity_retention"] = float(cap_ret_corr)
    degr["capacity_loss"]      = float(L_corr)
    degr["soh"]                = float(soh_corr)
    degr["capacity_fade_percent"] = float(L_corr * 100.0)
    degr["eol_years"]          = eol_corr
    degr["e_cap_degraded"]     = e_cap_eff * cap_ret_corr
    degr["p_cap_degraded"]     = float(bat_params["power_capacity_W"]) / 1e6 * cap_ret_corr
    degr["meta"]["calendar_correction"] = "Xu ft_calendar, reporting only"
    return degr


def _xu_full_degradation(storage_p, storage_e, e_cap_eff, T_cell_C, dt_hours) -> Dict:
    """Full Xu model (Xu cycling S_delta + Xu calendar) on a given year's dispatch.

    Same dict interface the multiyear loop consumes from the Shi path: keys "fd", 
    "fd_cycle", "fd_calendar" (period values; the loop annualises with *scale). Uses the
    SAME rainflow cycles and the SAME e_cap_eff as the Shi path, so the only difference is
    the cycling stress (Xu S_delta vs fitted Shi k3*delta^k4). The calendar term is Xu in
    both. Lets _run_multiyear(deg_model="xu") drive SoH/replacement/NPV entirely from Xu.
    """
    e_arr           = np.asarray(storage_e, dtype=float)
    n_steps         = len(e_arr)
    t_total_seconds = n_steps * dt_hours * 3600.0
    sigma_mean      = float(np.mean(e_arr)) / max(e_cap_eff, 1e-9)
    cycles          = rainflow_cycle_counting(storage_e, e_cap_eff)
    fd, fd_cycle, fd_cal = compute_fd(cycles, sigma_mean, t_total_seconds, T_cell_C)
    return {"fd": float(fd), "fd_cycle": float(fd_cycle), "fd_calendar": float(fd_cal)}


# =============================================================================
# Multi-year loop  (v5.3: MODIFIED for proper discounted NPV)
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
    verbose:            bool  = True,
    deg_model:          str   = "shi",   # "shi" (fitted Phi) or "xu" (full Xu model)
) -> Dict:
    
    """Year-by-year degradation loop with capacity fade feedback and replacement.

    Each year:
      - re-solves the LP at capacity faded to e_cap_nominal * SoH,
      - extracts per-year UNDISCOUNTED revenue on two bases:
          battery-only (marginal) = total plant − wind-only baseline, total plant = wind + battery (SHIPP-compatible),
      - accumulates Shi cycle + Xu calendar degradation, updates SoH, and triggers a replacement when SoH falls below the threshold.

    Lifetime NPV is assembled through wp2_econ (capex, replacement_cost, discount_weights, annuity_factor) on the SHIPP year-1..N-1 convention:
          NPV = -capex_initial + Σ rev_t * weights[t-1] - PV(replacements)
    A no-degradation NPV (year-1 revenue × annuity) and a Plan B single-year extrapolation NPV are also returned for comparison.
    """

    n           = len(wind_8760)
    scale       = 1.0 / max(n / 8760.0, 1e-9)     # annualisation factor
    price_eur   = price_8760          # prices in EUR for revenue

    # Wind-only baseline: what the wind farm earns with NO battery (curtailed at p_max)
    wind_export_no_bat = np.minimum(wind_8760, p_max_MW)
    rev_wind_only = (365.0 * 24.0 / n) * float(np.dot(price_eur, wind_export_no_bat)) * dt

    fd_cumulative    = 0.0
    soh              = 1.0
    n_replacements   = 0
    replacement_years: List[int]   = []
    soh_trajectory:    List[tuple] = []
    annual_fd:         List[tuple] = []
    annual_gradient:   List        = []
    annual_soc:        List        = []
    annual_revenue:       List[float] = []   # battery-only marginal revenue per year
    annual_revenue_total: List[float] = []   # total plant revenue per year (wind+bat)
    annual_revenue_arb:   List[float] = []   # battery arbitrage revenue per year (price . storage_p)
    soc_frac_fixed:       float       = None

    if verbose:
        period_days = n * dt / 24.0
        print(f"\n{'─'*70}")
        print(f"Multi-year loop: {N_YEARS} yr | tile: {n} h ({period_days:.1f} d) | "
              f"E_cap={e_cap_nominal:.0f} MWh | P_cap={p_cap_MW:.0f} MW | "
              f"SoH threshold: {EOL_REPLACEMENT*100:.0f}%")
        print(f"{'─'*70}")
        print(f"  {'Year':>4}  {'SoH_%':>7}  {'fd_yr':>9}  {'fd_cum':>9}  "
              f"{'rev_kEUR':>10}  {'t_lp_s':>7}  {'Replaced':>8}")

    t_loop_start = time.perf_counter()
    t_lp_total   = 0.0

    for year in range(1, N_YEARS): # 19 revenue years (SHIPP convention)
        t_yr_start = time.perf_counter()

        # ── Effective capacity this year ──
        e_cap_eff = e_cap_nominal * soh
        e_start1_yr = soc_frac_fixed * e_cap_eff if soc_frac_fixed is not None else None

        # ── Re-solve LP with degraded capacity ──
        stor_yr  = _build_storage_year(
            e_cap_eff, p_cap_MW, rte_ac, e_cost_EUR_per_MWh, soc_min, soc_max
        )
        price_dam = TimeSeries((price_8760).tolist(), dt)
        prod_yr   = Production(TimeSeries(wind_8760.tolist(), dt), p_cost=0.0)
        prod_null = Production(TimeSeries([0.0] * n, dt), p_cost=0.0)

        t_lp_start = time.perf_counter()
        os_yr = solve_lp_pyomo(
            price_dam, prod_yr, prod_null, stor_yr, stor_null,
            discount_rate, N_YEARS, p_min, p_max_MW, n,
            pyo_solver, fixed_cap=True, soc_max1=soc_max,
            return_duals=True, e_start1=e_start1_yr,
        )
        t_lp_yr = time.perf_counter() - t_lp_start
        t_lp_total += t_lp_yr

        # ── Extract per-year revenues on three bases (via wp2_econ) ──────
        #   arbitrage : price . storage_p                 (battery-as-asset; SHIPP a_npv)
        #   marginal  : total plant - wind-only baseline  (+ curtailment recovery)
        #   total     : price . (wind_after_curtailment + battery)
        # All three come from the same dispatch at no extra cost. revenue_annual
        # returns arbitrage + marginal; total is reconstructed as marginal + wind-only.
        p_prod_yr = np.array(os_yr.production_p[0].data, dtype=float)
        p_bat_yr  = np.array(os_yr.storage_p[0].data,    dtype=float)
        rev_yr      = revenue_annual(price_eur, p_bat_yr, p_prod_yr,
                                     wind_8760, p_max_MW, n, dt)
        rev_arb_yr  = rev_yr["arbitrage"]
        rev_bat_yr  = rev_yr["marginal"]             # marginal == battery-only basis
        rev_total_yr = rev_bat_yr + rev_wind_only    # total = marginal + wind-only baseline
        annual_revenue_arb.append(rev_arb_yr)
        annual_revenue.append(rev_bat_yr)
        annual_revenue_total.append(rev_total_yr)

        # ── Annual degradation with calendar correction ──
        storage_e_yr = os_yr.storage_e[0].data
        if soc_frac_fixed is None:
            soc_frac_fixed = os_yr.soc_final / e_cap_eff
        annual_soc.append(np.array(storage_e_yr, dtype=float).copy())
        storage_p_yr = os_yr.storage_p[0].data

        if deg_model == "xu":
            degr_yr = _xu_full_degradation(
                storage_p_yr, storage_e_yr, e_cap_eff, T_cell_C, dt,
            )
        else:
            degr_yr = _shi_with_calendar_correction(
                storage_p_yr, storage_e_yr, e_cap_eff,
                bat_params, shi_fit, T_cell_C, dt, eol_thresholds,
            )

        fd_yr          = degr_yr["fd"]          * scale
        fd_yr_cycle    = degr_yr["fd_cycle"]    * scale
        fd_yr_calendar = degr_yr["fd_calendar"] * scale
        fd_cumulative += fd_yr
        annual_fd.append((fd_yr, fd_yr_cycle, fd_yr_calendar))

        # ── Per-year gradient (Castillo, unchanged from v5.1) ──
        grad_yr = None
        if deg_model != "xu" and os_yr.dual_prices is not None:
            dual_yr   = os_yr.dual_prices["dual_e_min1"]
            e_cap_yr  = os_yr.dual_prices["e_cap1"]
            cycles_yr = rainflow_cycle_counting(storage_e_yr, e_cap_yr)
            sg_yr     = compute_subgradient(
                storage_e=storage_e_yr, cycles=cycles_yr, dt_hours=dt,
                battery_replacement_cost_per_MWh=repl_e_EUR_per_MWh,
                eff_in=eta_symmetric(rte_ac), eff_out=eta_symmetric(rte_ac), shi_fit=shi_fit,
            )

            factor_yr        = npf.npv(discount_rate, np.ones(N_YEARS)) - 1
            dual_per_year_yr = dual_yr / factor_yr

            # [0] OLD fused scalar (deprecated, kept for continuity)
            dDeg_dDoD_yr = -e_cap_yr * float(np.dot(sg_yr["subgrad_combined"], dual_per_year_yr))

            dods_yr     = np.array([c["dod"]      for c in cycles_yr])
            cnts_yr     = np.array([c["count"]    for c in cycles_yr])
            socm_yr     = np.array([c["soc_mean"] for c in cycles_yr])
            mean_dod_yr = float(np.average(dods_yr, weights=cnts_yr)) if len(dods_yr) > 0 else 0.0

            # TERM 1: dRev/dE_cap (exact, Castillo Theorem 1)
            dual_max_yr   = os_yr.dual_prices["dual_e_max1"]
            dRev_dEcap_yr = (
                soc_max * float(np.sum(dual_max_yr))
                - soc_min * float(np.sum(dual_yr))
            )

            # TERM 2: dDegCost/dE_cap (frozen-dispatch)
            if len(dods_yr) > 0:
                phi_prime_cyc = phi_shi_prime_with_stress(
                    dods_yr, socm_yr, T_cell_C, shi_fit.k3, shi_fit.k4
                )
                ddod_dEcap_yr     = -dods_yr / e_cap_yr
                dDegCost_dEcap_yr = repl_e_EUR_per_MWh * factor_yr * scale * float(
                    np.sum(cnts_yr * phi_prime_cyc * ddod_dEcap_yr)
                )
            else:
                dDegCost_dEcap_yr = 0.0

            # TERM 3: lambda_E
            lambda_E_yr   = e_cost_EUR_per_MWh # dCapex/dE: marginal capacity added at INITIAL capex (245), not replacement

            # Assembled gradient
            dNPV_dEcap_yr = dRev_dEcap_yr - dDegCost_dEcap_yr - lambda_E_yr
            rc_e_cap_yr   = float(os_yr.dual_prices.get("rc_e_cap1", 0.0))

            grad_yr = (
                dDeg_dDoD_yr, float(np.mean(np.abs(sg_yr["subgrad_combined"]))),
                mean_dod_yr, float(np.mean(np.abs(dual_per_year_yr))),
                float(e_cap_yr), float(np.mean(sg_yr["n_straddled"] > 0)),
                # Was cycle_coverage from the attribution map, which the rewritten degradation_subgradient removed. The equivalent diagnostic in the
                # straddle construction is the share of timesteps at which at least one cycle straddles, i.e. at which the gradient is non-zero.
                sg_yr["subgrad_combined"].copy(),
                dRev_dEcap_yr, dDegCost_dEcap_yr, lambda_E_yr,
                dNPV_dEcap_yr, rc_e_cap_yr,
            )
        annual_gradient.append(grad_yr)

        # ── EFC for this year ──
        efc_yr = count_equivalent_full_cycles(
            storage_p_yr, storage_e_yr, e_cap_eff, dt_hours=dt
        )

        # ── Update SoH ──
        L_cum = sei_capacity_loss(fd_cumulative)
        soh   = 1.0 - L_cum
        soh_trajectory.append((year, soh * 100.0, fd_cumulative, n_replacements))

        eol_trigger = EOL_REPLACEMENT + EOL_REPLACEMENT_TOL

        if verbose:
            replaced_tag = "YES" if soh < eol_trigger else ""
            print(f"  {year:>4d}  {soh*100:>7.3f}  {fd_yr:>9.5f}  "
                  f"{fd_cumulative:>9.5f}  {rev_bat_yr*1e-3:>10.1f}  "
                  f"{t_lp_yr:>7.2f}  {replaced_tag:>8}")

        # ── Battery replacement ──
        if soh < eol_trigger:
            n_replacements += 1
            replacement_years.append(year)
            fd_cumulative = 0.0
            soh           = 1.0
            soc_frac_fixed = None
            if verbose:
                print(f"  *** Battery #{n_replacements} replaced at end of year {year} ***")

    t_loop_total = time.perf_counter() - t_loop_start

    # ═════════════════════════════════════════════════════════════════════════
    # DISCOUNTED LIFETIME NPV — DUAL BASIS (battery-only + total plant)
    # Economics via wp2_econ; SHIPP year-1..N-1 discount convention.
    # ═════════════════════════════════════════════════════════════════════════
    weights           = discount_weights(discount_rate, N_YEARS)   # 19 per-year factors
    factor            = annuity_factor(discount_rate, N_YEARS)      # = sum(weights)
    capex_initial     = capex(e_cap_nominal, p_cap_MW,
                              e_cost_EUR_per_MWh, p_cost_EUR_per_MW)
    capex_replacement = replacement_cost(e_cap_nominal, p_cap_MW,
                                         repl_e_EUR_per_MWh, repl_p_EUR_per_MW)

    # PV of replacement costs (shared by both bases)
    pv_replacements = sum(capex_replacement * weights[yr - 1]
                          for yr in replacement_years)

    # ── A. BATTERY-ONLY basis ────────────────────────────────────────────
    #   rev_battery = rev_total_plant − rev_wind_only
    #   Isolates the battery investment decision from the wind farm.

    pv_rev_bat        = float(np.dot(annual_revenue, weights))
    npv_bat_multiyear = -capex_initial + pv_rev_bat - pv_replacements
    npv_bat_no_deg    = -capex_initial + annual_revenue[0] * factor

    # Plan B style (single-year extrapolation)
    fd_yr1_total   = annual_fd[0][0]
    deg_cost_planB = degradation_cost(fd_yr1_total, e_cap_nominal,
                                      repl_e_EUR_per_MWh, factor)
    npv_bat_planB  = npv_bat_no_deg - deg_cost_planB

    # ── A2. ARBITRAGE basis (battery-as-asset: price . storage_p) ─────────
    #   SHIPP a_npv definition. Differs from marginal only by curtailment recovery (~0.02% at this site). HEADLINE_BASIS selects which is primary.
    pv_rev_arb        = float(np.dot(annual_revenue_arb, weights))
    npv_arb_multiyear = -capex_initial + pv_rev_arb - pv_replacements
    npv_arb_no_deg    = -capex_initial + annual_revenue_arb[0] * factor
    npv_arb_planB     = npv_arb_no_deg - deg_cost_planB   # same deg penalty (capacity-based)

    # ── B. TOTAL PLANT basis (wind + battery, SHIPP-compatible) ──────────
    #   Same as kernel's own NPV, but with proper per-year discounting.
    #   The gradient (rc_e_cap1) validates against this basis.
    pv_rev_total        = float(np.dot(annual_revenue_total, weights))

    npv_total_multiyear = -capex_initial + pv_rev_total - pv_replacements
    npv_total_no_deg    = -capex_initial + annual_revenue_total[0] * factor

    if verbose:
        print(f"{'─'*70}")
        print(f"  Loop time: {t_loop_total:.1f}s ({t_lp_total:.1f}s LP)")
        print(f"  Final SoH: {soh*100:.2f}% | Replacements: {n_replacements}")
        print(f"\n  ── NPV comparison (v5.3) ─────────────────────────────────")
        print(f"  Wind-only baseline : {rev_wind_only*1e-3:>9.0f} kEUR/yr  (constant)")
        print(f"  Battery capex      : {capex_initial*1e-6:>9.2f} MEUR")
        print(f"  PV(replacements)   : {pv_replacements*1e-6:>9.2f} MEUR")
        print(f"")
        print(f"  BATTERY-ONLY basis:")
        print(f"    PV(bat. rev)     : {pv_rev_bat*1e-6:>9.2f} MEUR")
        print(f"    NPV (multi-year) : {npv_bat_multiyear*1e-6:>9.2f} MEUR  ← corrected")
        print(f"    NPV (no-deg)     : {npv_bat_no_deg*1e-6:>9.2f} MEUR")
        print(f"    NPV (Plan B)     : {npv_bat_planB*1e-6:>9.2f} MEUR")
        print(f"    Bat rev yr1→yr{N_YEARS} : {annual_revenue[0]*1e-3:.0f} → "
              f"{annual_revenue[-1]*1e-3:.0f} kEUR  "
              f"({(annual_revenue[-1]/annual_revenue[0] - 1)*100:+.1f}%)")
        print(f"")
        print(f"  ARBITRAGE basis (battery-as-asset; SHIPP a_npv):")
        print(f"    PV(arb. rev)     : {pv_rev_arb*1e-6:>9.2f} MEUR")
        print(f"    NPV (multi-year) : {npv_arb_multiyear*1e-6:>9.2f} MEUR")
        print(f"    NPV (no-deg)     : {npv_arb_no_deg*1e-6:>9.2f} MEUR")
        print(f"    NPV (Plan B)     : {npv_arb_planB*1e-6:>9.2f} MEUR")
        print(f"")
        print(f"  TOTAL PLANT basis (SHIPP-compatible):")
        print(f"    PV(total rev)    : {pv_rev_total*1e-6:>9.2f} MEUR")
        print(f"    NPV (multi-year) : {npv_total_multiyear*1e-6:>9.2f} MEUR")
        print(f"    NPV (no-deg)     : {npv_total_no_deg*1e-6:>9.2f} MEUR")
        print(f"    Total rev yr1→yr{N_YEARS}: {annual_revenue_total[0]*1e-3:.0f} → "
              f"{annual_revenue_total[-1]*1e-3:.0f} kEUR  "
              f"({(annual_revenue_total[-1]/annual_revenue_total[0] - 1)*100:+.1f}%)")
        print(f"  ──────────────────────────────────────────────────────────")

    return {
        # ── Legacy fields (v5.1 compatible) ──
        "soh_trajectory":       soh_trajectory,
        "annual_fd":            annual_fd,
        "n_replacements":       n_replacements,
        "replacement_years":    replacement_years,
        "replacement_cost_EUR": n_replacements * capex_replacement,
        "final_soh":            soh,
        "final_soh_pct":        soh * 100.0,
        "e_cap_nominal":        e_cap_nominal,
        "annual_gradient":      annual_gradient,
        "annual_soc":           annual_soc,
        # ── v5.2: per-year revenues ──
        "annual_revenue_bat_eur":   annual_revenue,         # battery-only per year
        "annual_revenue_total_eur": annual_revenue_total,   # total plant per year
        "rev_wind_only_eur":        rev_wind_only,          # wind baseline (constant)
        # ── v5.2: battery-only NPV ──
        "capex_initial_EUR":        capex_initial,
        "pv_rev_bat_EUR":           pv_rev_bat,
        "pv_replacements_EUR":      pv_replacements,
        "npv_bat_multiyear_EUR":    npv_bat_multiyear,
        "npv_bat_no_deg_EUR":       npv_bat_no_deg,
        "npv_bat_planB_EUR":        npv_bat_planB,
        # ── arbitrage basis (battery-as-asset; SHIPP a_npv) ──
        "annual_revenue_arb_eur":   annual_revenue_arb,
        "pv_rev_arb_EUR":           pv_rev_arb,
        "npv_arb_multiyear_EUR":    npv_arb_multiyear,
        "npv_arb_no_deg_EUR":       npv_arb_no_deg,
        "npv_arb_planB_EUR":        npv_arb_planB,
        # ── v5.2: total-plant NPV (SHIPP-compatible, gradient-compatible) ──
        "pv_rev_total_EUR":         pv_rev_total,
        "npv_total_multiyear_EUR":  npv_total_multiyear,
        "npv_total_no_deg_EUR":     npv_total_no_deg,
    }

# =============================================================================
# Phase 3 (v5.4): DoD parameter sweep at fixed (E, P)
# =============================================================================

def _window_from(center: float, width: float) -> Tuple[float, float]:
    """(center, width) -> (soc_min, soc_max), clamped to the physical [0,1] box."""
    return max(0.0, center - width / 2.0), min(1.0, center + width / 2.0)


def _dod_points() -> List[Tuple[str, float, float, str]]:
    """(series_tag, center, width, label) for both series plus the named reference
    windows, de-duped. The shared 10-90 point is tagged "both" and the CSV writer emits
    it into each series. Reference windows are always present and pass their label to any
    coincident parametric point."""
    cw = round(DOD_WIDTH_SERIES_CENTER, 6)
    ww = round(DOD_CENTER_SERIES_WIDTH, 6)
    width_pts  = [(cw, round(w, 6)) for w in DOD_WIDTH_SERIES]
    center_pts = [(round(c, 6), ww) for c in DOD_CENTER_SERIES]
    ref_label, ref_pts = {}, []
    for label, lo, hi in DOD_REFERENCE_WINDOWS:
        c = round((lo + hi) / 2.0, 6); w = round(hi - lo, 6)
        ref_pts.append((c, w)); ref_label[(c, w)] = label
    width_set, center_set = set(width_pts), set(center_pts)
    seen, uniq = set(), []
    for c, w in width_pts + center_pts + ref_pts:
        if (c, w) in seen:
            continue
        seen.add((c, w))
        in_w, in_c = (c, w) in width_set, (c, w) in center_set
        if   in_w and in_c: tag = "both"
        elif in_w:          tag = "width"
        elif in_c:          tag = "center"
        else:               tag = "ref"
        uniq.append((tag, c, w, ref_label.get((c, w), "")))
    return uniq


def _window_diagnostics(soc_year1, e_cap_nominal: float, k3: float, k4: float,
                        T_C: float
                        ) -> Tuple[float, float, int, float, float, float, float, float]:
    """Year-1 cycle-depth diagnostics + Shi-vs-Xu cycling fd for one DoD window.

    Returns (mean_dod_wt, frac_count_below_convex, n_cycles, max_dod_obs, frac_fd_below_convex, fd_cycle_shi, fd_cycle_xu, frac_fd_below_convex_xu).
    frac_fd_below_convex_xu : same share measured on the Xu branch. The Shi version understates it, because the surrogate
                              discounts shallow cycles more than deep ones. The reported degradation path is Xu, so this is
                              the figure that supports any claim about how much damage the extrapolation region actually carries.
    frac_count_below_convex : share of cycle COUNT with delta < 0.1437 (Xu non-convex).
    frac_fd_below_convex    : share of Shi cycling FD from those cycles. k4 > 1 down-weights shallow cycles, so this is far below the count fraction.
    fd_cycle_shi / fd_cycle_xu : year-1 cycling fd from the fitted Shi Phi and from the pure Xu S_delta, on the SAME cycles and SAME S_soc*S_temp stress.
                              Ratio = derivative-free fit-error measure; Xu is valid for reporting at all delta (only the gradient needs convexity).
    """
    cyc = rainflow_cycle_counting(np.asarray(soc_year1, dtype=float), e_cap_nominal)
    if not cyc:
        return 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0
    dods = np.array([c["dod"]      for c in cyc], dtype=float)
    cnts = np.array([c["count"]    for c in cyc], dtype=float)
    socm = np.array([c["soc_mean"] for c in cyc], dtype=float)
    wsum = float(cnts.sum())
    if wsum <= 0.0:
        return 0.0, 0.0, len(cyc), float(dods.max()), 0.0, 0.0, 0.0, 0.0
    below      = dods < LMO_NONCONVEX_DELTA
    mean_dod   = float(np.average(dods, weights=cnts))
    frac_below = float(cnts[below].sum() / wsum)
    shi_contrib = cnts * np.asarray(phi_shi_with_stress(dods, socm, T_C, k3, k4), dtype=float)
    xu_contrib  = cnts * np.asarray(fc_cycle(dods, socm, T_C), dtype=float)
    fd_cycle_shi = float(shi_contrib.sum())
    fd_cycle_xu  = float(xu_contrib.sum())
    frac_fd_below = float(shi_contrib[below].sum() / fd_cycle_shi) if fd_cycle_shi > 0.0 else 0.0
    frac_fd_below_xu = float(xu_contrib[below].sum() / fd_cycle_xu) if fd_cycle_xu > 0.0 else 0.0
    return (mean_dod, frac_below, len(cyc), float(dods.max()),
            frac_fd_below, fd_cycle_shi, fd_cycle_xu, frac_fd_below_xu)


def _run_dod_sweep(
    power_wind_MW:      np.ndarray,
    price_eur:          np.ndarray,
    p_max_MW:           float,
    e_cap_fixed:        float,
    p_cap_fixed:        float,
    rte_ac:             float,
    e_cost_EUR_per_MWh: float,
    p_cost_EUR_per_MW:  float,
    repl_e_EUR_per_MWh: float,
    repl_p_EUR_per_MW:  float,
    bat_params:         Dict,
    T_cell_C:           float = 25.0,
) -> List[Dict]:
    """Sweep the SoC operating window at fixed (E, P).

    Per window: derive (soc_min, soc_max), refit the convex Phi for that width, run the unchanged multi-year loop, and capture DoD -> revenue -> EoL plus the fit/extrapolation diagnostics. dod (kernel floor)
    stays 1 - soc_min; soc_max1 stays soc_max (kernel lines 168/171).
    """
    n = len(price_eur)
    stor_null = Storage(e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0, e_cost=0, p_cost=0)
    points = _dod_points()

    print(f"\n{'='*70}")
    print(f"DoD SWEEP at fixed E={e_cap_fixed:.0f} MWh, P={p_cap_fixed:.0f} MW "
          f"(E/P={e_cap_fixed/p_cap_fixed:.1f}h)")
    print(f"{len(points)} windows | LP solves: {len(points)} x {N_YEARS} yr "
          f"= {len(points)*N_YEARS}")
    print(f"{'='*70}")

    results = []
    t_sweep_start = time.perf_counter()

    for idx, (tag, center, width, label) in enumerate(points):
        soc_min, soc_max = _window_from(center, width)
        dod_kernel = 1.0 - soc_min

        try:
            shi_fit_w = fit_shi_polynomial(
                soc_min, soc_max,
                source=f"dod {soc_min:.2f}-{soc_max:.2f}", verbose=False,
            )
        except ValueError as exc:
            print(f"  [skip] {tag:>6} window {soc_min:.2f}-{soc_max:.2f}: {exc}")
            continue

        print(f"\n[{idx+1}/{len(points)}] {tag:>6} | window {soc_min*100:.0f}-{soc_max*100:.0f}% "
              f"| center={center:.2f} width={width:.2f} | k4={shi_fit_w.k4:.3f} "
              f"R2={shi_fit_w.r2:.3f}")

        t0 = time.perf_counter()
        my_shi = _run_multiyear(
            wind_8760=power_wind_MW, price_8760=price_eur, stor_null=stor_null,
            p_max_MW=p_max_MW, e_cap_nominal=e_cap_fixed, p_cap_MW=p_cap_fixed,
            rte_ac=rte_ac, e_cost_EUR_per_MWh=e_cost_EUR_per_MWh,
            p_cost_EUR_per_MW=p_cost_EUR_per_MW,
            repl_e_EUR_per_MWh=repl_e_EUR_per_MWh,
            repl_p_EUR_per_MW=repl_p_EUR_per_MW,
            bat_params=bat_params, shi_fit=shi_fit_w,
            soc_min=soc_min, soc_max=soc_max, T_cell_C=T_cell_C,
            verbose=SWEEP_VERBOSE, deg_model="shi",
        )
        # ── Save Year-1 SoC for sigma_effect_real.py (Fig 2.3) ──────────────
        # Runs only for the 10-90 % reference window.
        # Filename encodes tile length: _8760h = full year, _720h = test run.
        # sigma_effect_real.py requires _8760h; it will refuse _720h explicitly.
        if abs(soc_min - 0.10) < 0.005 and abs(soc_max - 0.90) < 0.005:
            _n   = len(my_shi["annual_soc"][0])          # 8760 or N_DAYS_TEST*24
            _nyr = len(my_shi["annual_soc"])              # N_YEARS-1 rows
            # Year 1 only (shape: (_n,))  — used by sigma_effect_real.py
            _e_yr1 = np.array(my_shi["annual_soc"][0], dtype=float)
            np.save(RESULTS_DIR / f"storage_e_yr01_{_n}h.npy", _e_yr1)
            # All years stacked (shape: (_nyr, _n)) — one row per sim year
            _e_all = np.stack([np.array(s, dtype=float)
                                for s in my_shi["annual_soc"]])
            np.save(RESULTS_DIR / f"storage_e_{N_YEARS}yr_{_n}h_stack.npy", _e_all)
            # Scalar capacity (unchanged across runs, kept for sigma_effect_real.py)
            np.save(RESULTS_DIR / "e_cap_fixed.npy", np.array([e_cap_fixed]))
            print(f"  ✓ storage_e_yr01_{_n}h.npy          ({_n} steps, Year 1)")
            print(f"  ✓ storage_e_{N_YEARS}yr_{_n}h_stack.npy  ({_nyr}×{_n}, all years)")

        my_xu = None
        if COMPARE_XU_MULTIYEAR:
            my_xu = _run_multiyear(
                wind_8760=power_wind_MW, price_8760=price_eur, stor_null=stor_null,
                p_max_MW=p_max_MW, e_cap_nominal=e_cap_fixed, p_cap_MW=p_cap_fixed,
                rte_ac=rte_ac, e_cost_EUR_per_MWh=e_cost_EUR_per_MWh,
                p_cost_EUR_per_MW=p_cost_EUR_per_MW,
                repl_e_EUR_per_MWh=repl_e_EUR_per_MWh,
                repl_p_EUR_per_MW=repl_p_EUR_per_MW,
                bat_params=bat_params, shi_fit=shi_fit_w,
                soc_min=soc_min, soc_max=soc_max, T_cell_C=T_cell_C,
                verbose=SWEEP_VERBOSE, deg_model="xu",
            )
        elapsed = time.perf_counter() - t0

        (mean_dod_obs, frac_below, n_cyc, max_dod_obs,
         frac_fd_below, fd_cycle_shi, fd_cycle_xu,
         frac_fd_below_xu) = _window_diagnostics(
            my_shi["annual_soc"][0], e_cap_fixed, shi_fit_w.k3, shi_fit_w.k4, T_cell_C
        )
        fd_yr1     = my_shi["annual_fd"][0][0]
        first_repl = my_shi["replacement_years"][0] if my_shi["replacement_years"] else None
        rev_fade   = (my_shi["annual_revenue_bat_eur"][-1] /
                      my_shi["annual_revenue_bat_eur"][0] - 1) * 100 \
                     if my_shi["annual_revenue_bat_eur"][0] != 0 else 0.0

        # ── Full-model Xu results (nan/None when COMPARE_XU_MULTIYEAR is off) ──
        nanv     = float("nan")
        npv_shi  = my_shi["npv_bat_multiyear_EUR"]
        if my_xu is not None:
            npv_xu        = my_xu["npv_bat_multiyear_EUR"]
            fd_yr1_xu     = my_xu["annual_fd"][0][0]
            soh_end_xu    = my_xu["final_soh_pct"]
            nrepl_xu      = my_xu["n_replacements"]
            first_repl_xu = my_xu["replacement_years"][0] if my_xu["replacement_years"] else None
            npv_xu_ratio  = (npv_xu / npv_shi) if npv_shi != 0 else nanv
        else:
            npv_xu = fd_yr1_xu = soh_end_xu = npv_xu_ratio = nanv
            nrepl_xu = nanv
            first_repl_xu = None

        results.append({
            "series": tag, "center": center, "width": width, "label": label,
            "soc_min": soc_min, "soc_max": soc_max, "dod_kernel": dod_kernel,
            "e_cap": e_cap_fixed, "p_cap": p_cap_fixed,
            "npv_bat_no_deg":    my_shi["npv_bat_no_deg_EUR"],
            "npv_bat_multiyear": npv_shi,
            "npv_bat_planB":     my_shi["npv_bat_planB_EUR"],
            "npv_bat_multiyear_xu": npv_xu,
            "npv_xu_over_shi":      npv_xu_ratio,
            "rev_bat_yr1_eur":   my_shi["annual_revenue_bat_eur"][0],
            "rev_bat_yrN_eur":   my_shi["annual_revenue_bat_eur"][-1],
            "rev_fade_pct":      rev_fade,
            "fd_yr1": fd_yr1, "fd_yr1_xu": fd_yr1_xu,
            "n_replacements": my_shi["n_replacements"], "n_replacements_xu": nrepl_xu,
            "first_repl_yr": first_repl, "first_repl_yr_xu": first_repl_xu,
            "final_soh_pct": my_shi["final_soh_pct"], "final_soh_pct_xu": soh_end_xu,
            "shi_k3": shi_fit_w.k3, "shi_k4": shi_fit_w.k4,
            "shi_r2": shi_fit_w.r2, "fit_hi": shi_fit_w.fit_hi,
            "mean_dod_obs": mean_dod_obs, "max_dod_obs": max_dod_obs,
            "frac_below_convex": frac_below, "frac_fd_below_convex": frac_fd_below,
            "frac_fd_below_convex_xu": frac_fd_below_xu,
            "fd_cycle_shi_yr1": fd_cycle_shi, "fd_cycle_xu_yr1": fd_cycle_xu,
            "fd_cycle_xu_over_shi": (fd_cycle_xu / fd_cycle_shi) if fd_cycle_shi > 0 else nanv,
            "n_cycles_yr1": n_cyc,
            "elapsed_s": elapsed,
        })

        cyc_ratio = (fd_cycle_xu / fd_cycle_shi) if fd_cycle_shi > 0 else float("nan")
        flag  = "  <-- check: material fd extrapolation" if frac_fd_below_xu > 0.10 else "" # Flag on the Xu branch: it is the reporting path and the larger of the two.

        print(f"  [Shi] NPV={npv_shi*1e-6:>7.1f}M  fd={fd_yr1:.5f}  "
              f"repl={my_shi['n_replacements']}  SoH_end={my_shi['final_soh_pct']:.1f}%  "
              f"mean_dod={mean_dod_obs:.3f}  cyc<0.15={frac_below:.2f} "
              f"fd<0.15={frac_fd_below:.4f} (xu {frac_fd_below_xu:.4f})  "
              f"yr1Xu/Shi={cyc_ratio:.2f}{flag}")              
        if my_xu is not None:
            print(f"  [Xu ] NPV={npv_xu*1e-6:>7.1f}M  fd={fd_yr1_xu:.5f}  "
                  f"repl={int(nrepl_xu)}  SoH_end={soh_end_xu:.1f}%  "
                  f"full-model NPV Xu/Shi={npv_xu_ratio:.3f}")
        print(f"        t={elapsed:.1f}s")

    total = time.perf_counter() - t_sweep_start
    print(f"\n{'-'*70}\nDoD sweep complete: {len(results)} windows in "
          f"{total:.0f}s ({total/60:.1f}m)")
    return results

# =============================================================================
# DoD sweep output (CSV)
# =============================================================================

def _save_dod_sweep_csv(results: List[Dict], e_cap: float, p_cap: float) -> Path:
    """Save the DoD sweep, one row per window, with fit + extrapolation diagnostics."""
    rows = []
    for r in results:
        base = {
            "center":               r["center"],
            "width":                r["width"],
            "label":                r["label"],
            "soc_min":              r["soc_min"],
            "soc_max":              r["soc_max"],
            "dod_kernel":           r["dod_kernel"],          # 1 - soc_min (floor param)
            "e_cap_MWh":            r["e_cap"],
            "p_cap_MW":             r["p_cap"],
            "npv_bat_noDeg_MEUR":     r["npv_bat_no_deg"]    * 1e-6,
            "npv_bat_multiyear_MEUR": r["npv_bat_multiyear"] * 1e-6,
            "npv_bat_planB_MEUR":     r["npv_bat_planB"]     * 1e-6,
            "npv_bat_multiyear_xu_MEUR": (r["npv_bat_multiyear_xu"] * 1e-6),
            "npv_xu_over_shi":           r["npv_xu_over_shi"],
            "rev_bat_yr1_kEUR":     r["rev_bat_yr1_eur"] * 1e-3,
            "rev_bat_yrN_kEUR":     r["rev_bat_yrN_eur"] * 1e-3,
            "rev_fade_pct":         r["rev_fade_pct"],
            "fd_yr1":               r["fd_yr1"],
            "fd_yr1_xu":            r["fd_yr1_xu"],
            "fd_cycle_shi_yr1":     r["fd_cycle_shi_yr1"],
            "fd_cycle_xu_yr1":      r["fd_cycle_xu_yr1"],
            "fd_cycle_xu_over_shi": r["fd_cycle_xu_over_shi"],
            "n_replacements":       r["n_replacements"],
            "n_replacements_xu":    r["n_replacements_xu"],
            "first_repl_yr":        r["first_repl_yr"],
            "first_repl_yr_xu":     r["first_repl_yr_xu"],
            "final_soh_pct":        r["final_soh_pct"],
            "final_soh_pct_xu":     r["final_soh_pct_xu"],
            "shi_k3":               r["shi_k3"],
            "shi_k4":               r["shi_k4"],
            "shi_r2":               r["shi_r2"],
            "fit_hi":               r["fit_hi"],
            "mean_dod_obs":         r["mean_dod_obs"],
            "max_dod_obs":          r["max_dod_obs"],
            "frac_below_convex":    r["frac_below_convex"],
            "frac_fd_below_convex":    r["frac_fd_below_convex"],
            "frac_fd_below_convex_xu": r["frac_fd_below_convex_xu"],
            "n_cycles_yr1":         r["n_cycles_yr1"],
            "elapsed_s":            r["elapsed_s"],
        }
        # Shared 10-90 anchor (tag "both") written into each series; solved once.
        series_tags = ["width", "center"] if r["series"] == "both" else [r["series"]]
        for st in series_tags:
            rows.append({"series": st, **base})
    path = RESULTS_DIR / f"{FILE_TAG}_dodsweep_{run_ts}_E{int(e_cap)}_P{int(p_cap)}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  ✓ CSV: {path.name}")
    return path


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    global run_ts

    print("=" * 80)
    print("WP2 BATTERY OPTIMIZATION + DEGRADATION  v5.4 (wp2_econ integration)")
    print("Shi (2018) + Xu calendar | multi-year NPV | DoD (SoC-window) sweep")
    print("=" * 80)

    # ── 1. Load config + raw series ──
    print("\n[1/4] Loading WP2 configuration...")
    setup = quick_setup(HPP_YAML, config={"interp_n": 2000}, verbose=False)
    hpp   = setup["hpp"]

    ws_all, wd_all, ti_all = _load_inputs(hpp)
    price_all               = _load_prices()

    # ── 2. Choose horizon ──
    print("\n[2/4] Preparing data...")
    n = _choose_horizon(len(ws_all), len(price_all))
    ws        = ws_all[:n]
    wd        = wd_all[:n]
    ti        = ti_all[:n] if ti_all is not None else None
    price_eur = price_all[:n]
    print(f"  Horizon: {n:,} h ({n/24:.1f} days)")
    print(f"  Mean price: {float(np.mean(price_eur)):.2f} EUR/MWh")

    # ── 3. PyWake ──
    print("\n[3/4] Running PyWake simulation...")
    power_wind_MW = _run_pywake_power_MW(setup, wd, ws, ti)
    print(f"  Wind mean: {float(np.mean(power_wind_MW)):.1f} MW | "
          f"peak: {float(np.max(power_wind_MW)):.1f} MW")

    # ── 4. Battery params ──
    bat      = setup["battery"]
    p_max_MW = float(hpp["grid_connection_capacity"]) / 1e6

    e_cap_yaml = float(bat["energy_capacity_Wh"]) / 1e6
    p_cap_yaml = float(bat["power_capacity_W"])   / 1e6

    rte_dc  = float(bat["rte_nominal"])
    pcu_eff = float(bat["pcu_efficiency"])
    rte_ac  = rte_dc * (pcu_eff ** 2)

    e_cost_EUR_per_MWh = float(bat["capex_EUR_per_kWh"]) * 1000.0
    p_cost_EUR_per_MW  = float(bat["capex_EUR_per_kW"])  * 1000.0
    repl_e_EUR_per_MWh = float(bat["repl_energy_EUR_per_kWh"]) * 1000.0  # energy expansion: replacement + deg valuation
    repl_p_EUR_per_MW  = float(bat["repl_power_EUR_per_kW"])   * 1000.0  # power expansion: replacement only

    bat_params = setup["battery"]

    # ── (E, P) overwrite for the DoD study; YAML stays the single default ──
    e_cap_fixed = E_CAP_FIXED_MWh if E_CAP_FIXED_MWh is not None else e_cap_yaml
    p_cap_fixed = P_CAP_FIXED_MW  if P_CAP_FIXED_MW  is not None else p_cap_yaml

    print(f"  Battery (YAML default): {p_cap_yaml:.0f} MW / {e_cap_yaml:.0f} MWh")
    print(f"  Fixed for DoD study   : E={e_cap_fixed:.0f} MWh, P={p_cap_fixed:.0f} MW "
          f"(E/P={e_cap_fixed/p_cap_fixed:.1f}h)")
    print(f"  Grid: {p_max_MW:.0f} MW | RTE(ac): {rte_ac*100:.1f}%")
    print(f"  e_cost: {e_cost_EUR_per_MWh/1000:.0f} EUR/kWh | "
          f"N_YEARS: {N_YEARS} | EOL: {EOL_REPLACEMENT*100:.0f}%")

    # ── DoD sweep (Phi refit per window inside the driver) ──
    print(f"\n[4/4] Running DoD sweep at fixed E={e_cap_fixed:.0f} MWh, "
          f"P={p_cap_fixed:.0f} MW...")
    results = _run_dod_sweep(
        power_wind_MW=power_wind_MW, price_eur=price_eur, p_max_MW=p_max_MW,
        e_cap_fixed=e_cap_fixed, p_cap_fixed=p_cap_fixed, rte_ac=rte_ac,
        e_cost_EUR_per_MWh=e_cost_EUR_per_MWh, p_cost_EUR_per_MW=p_cost_EUR_per_MW,
        repl_e_EUR_per_MWh=repl_e_EUR_per_MWh, repl_p_EUR_per_MW=repl_p_EUR_per_MW,
        bat_params=bat_params,
    )
    _save_dod_sweep_csv(results, e_cap_fixed, p_cap_fixed)

    # ── Config sidecar ──
    config_path = RESULTS_DIR / f"{FILE_TAG}_dodsweep_{run_ts}_E{int(e_cap_fixed)}_P{int(p_cap_fixed)}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": "v5.4", "sweep_kind": "dod",
            "revenue_basis": f"headline={HEADLINE_BASIS}; arbitrage + marginal + total all reported",
            "E_CAP_FIXED_MWh": e_cap_fixed, "P_CAP_FIXED_MW": p_cap_fixed,
            "width_series": {"center": DOD_WIDTH_SERIES_CENTER, "widths": DOD_WIDTH_SERIES},
            "center_series": {"width": DOD_CENTER_SERIES_WIDTH, "centers": DOD_CENTER_SERIES},
            "N_YEARS": N_YEARS, "discount_rate": discount_rate,
            "eta_symmetric": eta_symmetric(rte_ac), "round_trip_ac": rte_ac,
            "e_cost_eur_kwh": e_cost_EUR_per_MWh / 1000.0,
            "RUN_HOURS": n, "EOL_REPLACEMENT": EOL_REPLACEMENT,
            "LMO_NONCONVEX_DELTA": LMO_NONCONVEX_DELTA,
        }, f, indent=2)
    print(f"  ✓ Config: {config_path.name}")
    print(f"\n{'═'*80}\n✓ v5.4 COMPLETE (DoD sweep)\n{'═'*80}")


if __name__ == "__main__":
    log_path = RESULTS_DIR / f"{FILE_TAG}_log_{run_ts}.txt"
    with TeeLogger(log_path):
        main()
    # Print outside tee so we know the file is written
    print(f"\n  ✓ Full log saved: {log_path.name}")