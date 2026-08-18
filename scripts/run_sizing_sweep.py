"""Sizing sweep: net present value over battery energy and power capacity.

Fork origin: planB_2d_parameter_sweep_a_npv.py

Sweeps a two-dimensional grid of battery energy capacity and power capacity, solving the dispatch LP at each point and evaluating net present value under
three degradation treatments: none, Xu et al. (2016), and the Shi et al. (2018) polynomial. Comparing the optimum across the three isolates the effect of
pricing degradation into the sizing decision.

Method
    At each grid point the LP is solved with fixed capacity, the resulting hourly state of charge is rainflow-counted, and capacity fade is valued at
    the replacement-energy cost. Points where power exceeds energy, or where duration exceeds eight hours, are filled from a neighbouring point rather
    than solved, since they are not candidate designs.

Modes
    SWEEP_MODE = "annual"   one year, extrapolated across the horizon
    SWEEP_MODE = "lifetime" 20 sequential years with capacity fade carried forward and replacement when SoH reaches the threshold

Outputs (timestamped, written to results/sizing_sweep/)
    *_sweep_*.csv           one row per grid point
    *_report_*.txt          summary and optimum coordinates
    *_npv_comparison_*.png  NPV heatmaps for the three degradation treatments
    *_margins_*.png         NPV difference between treatments
    *_slices_*.png          one-dimensional cuts through the optimum

Usage:
    python scripts/run_sizing_sweep.py

Author: Thodoris Tsonopoulos, MSc thesis, TU Delft Wind Energy
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

# ─── Path setup ─────────────────────────────────────────────────────────────
from degradation.paths import (
    HPP_YAML,
    PRICE_CSV_2022,
    results_dir as make_results_dir,
)

OUTPUT_DIR = make_results_dir("sizing_sweep")

import numpy as np
import numpy_financial as npf
import pyomo.environ as pyo

from shipp.kernel_pyomo import solve_lp_pyomo
from shipp.components import Storage, Production, TimeSeries

from degradation.xu import (
    rainflow_cycle_counting,
    analyze_degradation,
    ft_calendar,
    sei_capacity_loss,
    count_equivalent_full_cycles,
    fit_shi_polynomial,
)
from degradation.shi import analyze_degradation_shi
from degradation.economics import (
    eta_symmetric, capex, replacement_cost, annuity_factor, discount_weights,
    degradation_cost, annualize_to_year, revenue_annual, lifetime_npv, HEADLINE_BASIS,
)

PRICE_CSV = PRICE_CSV_2022          # switch to PRICE_CSV_2019 for the 2019 price year


# ════════════════════════════════════════════════════════════════════════════
# 0.  QUICK TOGGLES
# ════════════════════════════════════════════════════════════════════════════

# Expanded, high-resolution grid parameters

# E_CAP_GRID = [150, 300, 450, 600, 700, 800, 900, 1000, 1100, 1300, 1500] # 11 points
# P_CAP_GRID = [75, 125, 175, 200, 225, 250, 275, 325]                   # 8 points

# E_CAP_GRID = [300, 350, 400, 425, 450, 475, 500, 525, 550, 600]   # MWh #old RTE version
# P_CAP_GRID = [75, 100, 125, 150, 175, 200]                         # MW

E_CAP_GRID = [450, 500, 550, 600, 650, 700, 750]   # 50-MWh steps, peak (600) interior
P_CAP_GRID = [125, 150, 175, 200, 225, 250]        # 25-MW steps, brackets Xu 175 and Shi 200
# 7 x 6 = 42 points

# ── Re-plot from a previous CSV run (skip the LP sweep entirely) ─────────
# Set REPLOT_FROM_LAST = True to automatically find and load the most recent
# CSV in the output folder — no need to type a filename.
# Set REPLOT_CSV to an explicit path only if you want a specific older run.
# If both are set, REPLOT_CSV takes priority.
# Leave both at their defaults to run the full sweep normally.
REPLOT_FROM_LAST: bool = False
REPLOT_CSV: str | None = None

# Include calendar aging in the degradation evaluation?
INCLUDE_CALENDAR = True

# ── Sweep mode ────────────────────────────────────────────────────────────
# "annual"   : one LP per (E, P) point, single-year dispatch, degradation
#              post-processed, 20-year NPV via annuity factor (fast, ~5 min)
# "lifetime" : 20 LPs per (E, P) point, capacity degrades year-on-year,
#              replacement triggered when SoH < REPL_THRESHOLD, three parallel
#              loops (no-deg / Xu / Shi), ~20× slower than annual
SWEEP_MODE: str = "lifetime"   # change to "lifetime" to run the 20-year version

# Battery replacement SoH threshold for lifetime mode
REPL_THRESHOLD: float = 0.70   # replace when capacity retention drops below 70%

# ── Lifetime smart filtering ─────────────────────────────────────────────
# Runs the fast annual sweep first, then skips lifetime evaluation for
# (E, P) points that are clearly non-competitive based on their 1-year NPV.
# A point is kept if its annual NPV (Xu OR Shi) exceeds this fraction of
# the best annual NPV for that model.  0.65 is safe — the v5.2 lifetime
# optimum at 800 MWh had ~90% of the annual optimum's NPV at that point.
LIFETIME_NPV_FLOOR: float = 0.65

# Duration guard for lifetime mode (same logic as annual smart filter)
LIFETIME_MIN_DURATION_H: float = 1.0
LIFETIME_MAX_DURATION_H: float = 8.0

# Horizons
RUN_HOURS  = 8760         # full year

# Economics
DISCOUNT_RATE = 0.03
N_YEARS       = 20

# Diagnostic: print a_npv vs wp2_econ arbitrage reconstruction per point (annual
# mode). Default ON as a regression guard — if a cost-wiring edit ever breaks the
# a_npv == (-capex + arb_rev*annuity) identity, the printed diff goes non-zero.
RECON_CHECK = True
DT            = 1.0
P_MIN_MW      = 0.0


# ════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_data(n_hours: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load wind power and price data. Returns (wind_MW, price_eur, params)."""
    import pandas as pd
    import xarray as xr
    from py_wake.site import XRSite
    from degradation.site import quick_setup, get_wake_model

    print("  Loading WP2 configuration...")
    setup = quick_setup(HPP_YAML, config={"interp_n": 2000}, verbose=False)
    hpp = setup["hpp"]

    ts = hpp["site"]["energy_resource"]["time_series"]["wind_resource"]
    ws = np.asarray(ts["wind_speed"], dtype=float)
    wd = np.asarray(ts["wind_direction"], dtype=float)
    ti_dat = ts.get("turbulence_intensity")
    ti = None
    if isinstance(ti_dat, dict) and "data" in ti_dat:
        ti = np.asarray(ti_dat["data"], dtype=float)
        if ti.shape != ws.shape:
            ti = None

    n = min(n_hours, len(ws))
    ws, wd = ws[:n], wd[:n]
    if ti is not None:
        ti = ti[:n]

    print("  Running PyWake simulation...")
    site = XRSite(ds=xr.Dataset(data_vars=dict(P=1)))
    wf_model = get_wake_model("Bastankhah", site, setup["windturbine"])
    kwargs = {"x": setup["x"], "y": setup["y"],
              "wd": wd, "ws": ws, "time": np.arange(n) / 24.0}
    if ti is not None:
        kwargs["TI"] = ti
    sim_res = wf_model(**kwargs)
    power_wind = sim_res.Power.sum(["wt"]).values / 1e6

    df = pd.read_csv(PRICE_CSV)
    for col in ("price_eur_mwh", "price_eur_per_mwh", "Price", "price"):
        if col in df.columns:
            break
    price_eur = df[col].astype(float).to_numpy()[:n]

    bat = setup["battery"]
    rte_dc  = float(bat["rte_nominal"])
    pcu_eff = float(bat["pcu_efficiency"])
    rte_ac  = rte_dc * pcu_eff ** 2            # full AC round trip
    eta     = eta_symmetric(rte_ac)            # symmetric one-way split (= sqrt(rte_ac))

    params = {
        "e_cap":   float(bat["energy_capacity_Wh"]) / 1e6,
        "p_cap":   float(bat["power_capacity_W"]) / 1e6,
        "eta_in":  eta, # Symmetric split of the round-trip loss across charge and discharge.
        "eta_out": eta, # eta_in = eta_out = sqrt(round_trip), so eta_in*eta_out = round_trip is preserved (same economics) but charge is no longer modeled lossless.
        "rte_ac":  rte_ac, # full AC round trip (eta_out**2); for correct RTE reporting
        "soc_min": float(bat.get("soc_min", 0.10)),
        "soc_max": float(bat.get("soc_max", 0.90)),
        "e_cost_eur_per_kwh": float(bat["capex_EUR_per_kWh"]),
        "p_cost_eur_per_kw":  float(bat["capex_EUR_per_kW"]),
        "e_cost_eur_per_mwh": float(bat["capex_EUR_per_kWh"]) * 1000.0,
        "p_cost_eur_per_mw":  float(bat["capex_EUR_per_kW"]) * 1000.0,
        "repl_e_eur_per_mwh": float(bat["repl_energy_EUR_per_kWh"]) * 1000.0,  # energy expansion: replacement + deg
        "repl_p_eur_per_mw":  float(bat["repl_power_EUR_per_kW"])   * 1000.0,  # power expansion: replacement only
        "p_max": float(hpp["grid_connection_capacity"]) / 1e6,
        "bat_params": {"power_capacity_W": float(bat["power_capacity_W"])},
        "T_cell_C": 25.0,
    }

    shi_fit = fit_shi_polynomial(params["soc_min"], params["soc_max"], verbose=True)
    params["shi_fit"] = shi_fit
    params["k3"] = shi_fit.k3
    params["k4"] = shi_fit.k4

    # E/P ratio from YAML (for scaled mode)
    params["ep_ratio"] = params["e_cap"] / params["p_cap"]

    print(f"  Battery (YAML): {params['p_cap']:.0f} MW / {params['e_cap']:.0f} MWh")
    print(f"  Grid: {params['p_max']:.0f} MW")
    print(f"  RTE(ac): {rte_ac*100:.1f}%  (eta_symmetric = {params['eta_out']:.4f})")
    print(f"  SoC: {params['soc_min']*100:.0f}%–{params['soc_max']*100:.0f}%")

    return power_wind[:n], price_eur[:n], params


# ════════════════════════════════════════════════════════════════════════════
# 2.  SINGLE-POINT EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate_single_ecap(
    e_cap:      float,
    p_cap:      float,
    power_wind: np.ndarray,
    price_eur:  np.ndarray,
    params:     dict,
) -> dict:
    """Solve LP and evaluate degradation for one (E_cap, P_cap) point."""
    n       = len(price_eur)
    eta_out = params["eta_out"]
    eta_in  = params["eta_in"]
    soc_min = params["soc_min"]
    soc_max = params["soc_max"]
    p_max   = params["p_max"]
    e_cost  = params["e_cost_eur_per_mwh"]
    p_cost  = params["p_cost_eur_per_mw"]
    repl_e = params["repl_e_eur_per_mwh"]

    dod = 1.0 - soc_min

    stor = Storage(e_cap=e_cap, p_cap=p_cap, eff_in=eta_in, eff_out=eta_out,
                        e_cost=e_cost, p_cost=p_cost, dod=dod)
    stor_null = Storage(e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
                        e_cost=0, p_cost=0)

    price_dam = TimeSeries((price_eur).tolist(), DT)
    prod      = Production(TimeSeries(power_wind.tolist(), DT), p_cost=0.0)
    prod_null = Production(TimeSeries([0.0] * n, DT), p_cost=0.0)

    # ── Solve LP ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    os_res = solve_lp_pyomo(
        price_dam, prod, prod_null, stor, stor_null,
        DISCOUNT_RATE, N_YEARS, P_MIN_MW, p_max, n,
        "gurobi", fixed_cap=True, soc_max1=soc_max,
    )
    solve_s = time.perf_counter() - t0

    e_vec = np.array(os_res.storage_e[0].data, dtype=float)
    p_vec = np.array(os_res.storage_p[0].data, dtype=float)

    # ── NPV baseline from SHIPP's a_npv ──────────────────────────────────
    # a_npv (get_added_npv): battery arbitrage NPV, capex netted internally,
    # SHIPP 19-yr convention. MEUR → EUR. Still the annual-mode headline in P1.
    a_npv_eur = os_res.get_added_npv(DISCOUNT_RATE, N_YEARS) * 1e6

    # Shared economics via wp2_econ (single source of truth). These values are
    # IDENTICAL to the old hand-rolled npf.npv()-1 factor and
    # p_cost*p_cap + e_cost*e_cap capex — this rewire is value-neutral.
    factor    = annuity_factor(DISCOUNT_RATE, N_YEARS)    # 19-yr SHIPP annuity
    capex_eur = capex(e_cap, p_cap, e_cost, p_cost)

    # Single-year revenue, both definitions, from one dispatch (wp2_econ);
    # multiplied by the annuity for the flat lifetime extrapolation.
    p_prod = np.array(os_res.production_p[0].data, dtype=float)   # wind after curtailment
    rev_yr = revenue_annual(price_eur, p_vec, p_prod, power_wind, p_max, n, DT)
    revenue_arb_eur  = rev_yr["arbitrage"] * factor
    revenue_marg_eur = rev_yr["marginal"]  * factor

    # Marginal NPV baseline = marginal revenue − capex.
    npv_marg_eur = revenue_marg_eur - capex_eur

    # Curtailment-recovery value (UNCHANGED definition for P1: marginal − a_npv).
    # NOTE: this mixes capex bases (npv_marg uses capex_eur; a_npv uses SHIPP's
    # internal capex). It is pure curtailment ONLY if those agree — which the
    # reconciliation below tests. If recon_diff != 0, revisit this line in P1.5.
    curtailment_value_eur = npv_marg_eur - a_npv_eur

    # keep `revenue_eur` name alive for any downstream code that reads it
    revenue_eur = revenue_arb_eur

    # ── P1 RECONCILIATION (diagnostic; no effect on reported NPVs) ────────
    # Rebuild the arbitrage NPV from wp2_econ and compare to SHIPP a_npv.
    #   diff ~ 0  → routes agree; a_npv can be retired with zero change, and
    #               curtailment_value_eur above is clean.
    #   diff != 0 → a_npv nets a different capex / extra term; investigate
    #               before replacing, and curtailment_value_eur is off by diff.
    arb_npv_recon  = revenue_arb_eur - capex_eur
    recon_diff_eur = a_npv_eur - arb_npv_recon
    if RECON_CHECK:
        print(f"    [recon E={e_cap:>4.0f}] a_npv={a_npv_eur*1e-6:8.4f}  "
              f"arb_recon={arb_npv_recon*1e-6:8.4f}  "
              f"diff={recon_diff_eur*1e-6:+8.4f} MEUR")


    # ── Degradation evaluation ───────────────────────────────────────────
    # Xu model (full: cycle + calendar internally via compute_fd)
    xu_result = analyze_degradation(
        storage_p=p_vec.tolist(),
        storage_e=e_vec.tolist(),
        e_cap_nominal=e_cap,
        battery_params=params["bat_params"],
        dt_hours=DT,
        T_cell_C=params["T_cell_C"],
    )
    # BUG FIX vs 1D version: xu_result["fd"] is the TOTAL (cycle + calendar).
    # Using it as fd_xu_cycle and then adding fd_calendar again double-counted.
    # Correct: read the separate components directly from the result dict.
    fd_xu_cycle  = xu_result["fd_cycle"]     # cycle-only  (Xu rainflow)
    fd_calendar  = xu_result["fd_calendar"]  # calendar-only (Xu ft_calendar)

    # Shi model (cycle-only by design — no calendar term in Shi framework)
    shi_result = analyze_degradation_shi(
        storage_p=p_vec.tolist(),
        storage_e=e_vec.tolist(),
        e_cap_nominal=e_cap,
        battery_params=params["bat_params"],
        shi_fit=params["shi_fit"],
        T_cell_C=params["T_cell_C"],
        dt_hours=DT,
    )
    fd_shi_cycle = shi_result["fd_shi"]

    # Total fd: add calendar to both branches if enabled
    # For Xu:  calendar already computed inside analyze_degradation, read directly
    # For Shi: calendar is structurally absent; add Xu calendar term to reporting
    fd_xu_total  = xu_result["fd"] if INCLUDE_CALENDAR else fd_xu_cycle
    fd_shi_total = (fd_shi_cycle + fd_calendar) if INCLUDE_CALENDAR else fd_shi_cycle

    # EFC
    efc = count_equivalent_full_cycles(p_vec.tolist(), e_vec.tolist(), e_cap, DT)

    # Degradation cost via wp2_econ: annualize fd, value at repl_e on capacity
    # consumed, carry over lifetime by the annuity. Identical to the old
    # deg_scale = 365*24/n * factor formulation.
    fd_xu_annual  = annualize_to_year(fd_xu_total,  n)
    fd_shi_annual = annualize_to_year(fd_shi_total, n)
    deg_cost_xu  = degradation_cost(fd_xu_annual,  e_cap, repl_e, factor)  # deg valued at energy-expansion cost (capacity consumed)
    deg_cost_shi = degradation_cost(fd_shi_annual, e_cap, repl_e, factor)

    # NPV variants on BOTH baselines (same deg_cost, two revenue definitions).
    # _arb = arbitrage (SHIPP a_npv);  _marg = with curtailment recovery.
    npv_no_deg      = a_npv_eur                       # arbitrage baseline (= SHIPP a_npv)
    npv_with_xu     = a_npv_eur   - deg_cost_xu        # "physical accurate" (full Xu)
    npv_with_shi    = a_npv_eur   - deg_cost_shi       # "polynomial" (Shi-fit + Xu calendar)

    npv_marg_no_deg = npv_marg_eur                    # marginal baseline
    npv_marg_xu     = npv_marg_eur - deg_cost_xu
    npv_marg_shi    = npv_marg_eur - deg_cost_shi

    # SoH at end of year 1
    soh_xu  = (1.0 - sei_capacity_loss(fd_xu_total)) * 100.0
    soh_shi = (1.0 - sei_capacity_loss(fd_shi_total)) * 100.0

    # Rainflow cycle count
    cycles = rainflow_cycle_counting(e_vec, e_cap)
    n_cycles = len(cycles)

    return {
        "e_cap":          e_cap,
        "p_cap":          p_cap,
        "pruned":         False,
        "revenue_eur":    revenue_eur,
        "capex_eur":      capex_eur,
        "npv_no_deg":     npv_no_deg,
        "npv_with_xu":    npv_with_xu,
        "npv_with_shi":   npv_with_shi,
        "npv_marg_no_deg": npv_marg_no_deg,
        "npv_marg_xu":     npv_marg_xu,
        "npv_marg_shi":    npv_marg_shi,
        "revenue_marg_eur": revenue_marg_eur,
        "curtailment_value_eur": curtailment_value_eur,
        "fd_xu_cycle":    fd_xu_cycle,
        "fd_shi_cycle":   fd_shi_cycle,
        "fd_calendar":    fd_calendar,
        "fd_xu_total":    fd_xu_total,
        "fd_shi_total":   fd_shi_total,
        "deg_cost_xu":    deg_cost_xu,
        "deg_cost_shi":   deg_cost_shi,
        "efc":            efc,
        "n_cycles":       n_cycles,
        "soh_xu_pct":     soh_xu,
        "soh_shi_pct":    soh_shi,
        "solve_s":        solve_s,
        "throughput_mwh": float(np.sum(np.abs(p_vec))) * DT,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3.  SWEEP + OUTPUT
# ════════════════════════════════════════════════════════════════════════════

def find_latest_csv() -> Path:
    """Return the most recent sizing sweep CSV in OUTPUT_DIR.

    Searches for annual (sizing_sweep_*.csv) or lifetime
    (sizing_lifetime_sweep_*.csv) files depending on SWEEP_MODE.
    Sorting is done on the timestamp embedded in the filename
    (YYYYMMDD_HHMMSS), so it is independent of filesystem modification times.
    Raises FileNotFoundError if no matching CSV exists yet.
    """
    if SWEEP_MODE == "lifetime":
        pattern = "sizing_lifetime_sweep_*.csv"
    else:
        pattern = "sizing_sweep_*.csv"

    candidates = sorted(OUTPUT_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No {pattern} files found in {OUTPUT_DIR}.\n"
            "Run the full sweep first (set REPLOT_FROM_LAST = False)."
        )
    latest = candidates[-1]
    print(f"  Auto-selected most recent CSV: {latest.name}")
    return latest


def load_and_verify(csv_path: str) -> tuple[List[dict], dict, int]:
    """Load sweep results from a previous CSV run and verify config match.

    Parameters
    ----------
    csv_path : str or Path
        Path to a previously saved sizing_sweep_*.csv file.

    Returns
    -------
    results : List[dict]   — list of row dicts, same format as run_sweep_2d
    params  : dict         — minimal params dict needed for plotting
    n       : int          — number of time steps from the saved run

    Config verification
    -------------------
    Looks for a .json sidecar next to the CSV (same stem).  If found, compares
    E_CAP_GRID, P_CAP_GRID, INCLUDE_CALENDAR, and RUN_HOURS against the current
    module-level toggles.  Prints a warning for each mismatch so you know
    whether the saved data is still valid for the current configuration.
    If no sidecar exists, prints a reminder and continues.

    Raises
    ------
    FileNotFoundError if the CSV does not exist.
    """
    import json
    import pandas as pd

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"REPLOT_CSV not found: {csv_path}")

    print(f"  Loading results from: {csv_path.name}")
    df = pd.read_csv(csv_path)
    results = df.to_dict(orient="records")
    print(f"  Loaded {len(results)} rows.")

    # ── Config verification ────────────────────────────────────────────────
    json_path = csv_path.with_suffix(".json")
    if json_path.exists():
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        mismatches = []

        if saved.get("E_CAP_GRID") != E_CAP_GRID:
            mismatches.append(
                f"  E_CAP_GRID: saved={saved['E_CAP_GRID']}  current={E_CAP_GRID}"
            )
        if saved.get("P_CAP_GRID") != P_CAP_GRID:
            mismatches.append(
                f"  P_CAP_GRID: saved={saved['P_CAP_GRID']}  current={P_CAP_GRID}"
            )
        if saved.get("INCLUDE_CALENDAR") != INCLUDE_CALENDAR:
            mismatches.append(
                f"  INCLUDE_CALENDAR: saved={saved['INCLUDE_CALENDAR']}  "
                f"current={INCLUDE_CALENDAR}"
            )
        if saved.get("RUN_HOURS") != RUN_HOURS:
            mismatches.append(
                f"  RUN_HOURS: saved={saved['RUN_HOURS']}  current={RUN_HOURS}"
            )

        if mismatches:
            print("\n  WARNING: config mismatch between saved run and current toggles:")
            for m in mismatches:
                print(m)
            print("  Plots will use saved data but reflect the OLD configuration.")
            print("  Set REPLOT_CSV = None and re-run if you changed the grid or settings.\n")
        else:
            print("  Config verified, saved run matches current toggles. ✓")

        # Recover n from the sidecar
        n = int(saved.get("RUN_HOURS", RUN_HOURS))

        # Reconstruct minimal params dict for plotting (costs, reference design)
        params = {
            "e_cap":             300.0,    # WP2 reference — used only for plot markers
            "p_cap":             150.0,
            "soc_min":           saved.get("soc_min", 0.10),
            "soc_max":           saved.get("soc_max", 0.90),
            "eta_out":           saved.get("eta_out", 0.877),
            "e_cost_eur_per_kwh":saved.get("e_cost_eur_kwh", 150.0),
            "k3":                saved.get("k3", 3.24e-5),
            "k4":                saved.get("k4", 1.179),
        }
    else:
        print(
            "  NOTE: no JSON sidecar found next to CSV.  Cannot verify config.\n"
            "  Proceeding with current toggle values for plot parameters."
        )
        n = RUN_HOURS
        params = {
            "e_cap": 300.0, "p_cap": 150.0,
            "soc_min": 0.10, "soc_max": 0.90,
            "eta_out": 0.877, "e_cost_eur_per_kwh": 150.0,
            "k3": 3.24e-5, "k4": 1.179,
        }

    return results, params, n


# ════════════════════════════════════════════════════════════════════════════
# 3b.  LIFETIME SINGLE-POINT EVALUATION  (three parallel 20-year loops)
# ════════════════════════════════════════════════════════════════════════════

def evaluate_20year(
    e_cap:      float,
    p_cap:      float,
    power_wind: np.ndarray,
    price_eur:  np.ndarray,
    params:     dict,
) -> dict:
    """Run three parallel 20-year loops for one (E_cap, P_cap) design point.

    The three loops share identical price/wind data and LP setup.
    They differ only in how SoH evolves and whether replacement is triggered:

      Loop A — no degradation:
        SoH = 1.0 every year.  e_cap_eff = e_cap_nom always.
        No replacement cost.  Upper bound on NPV.

      Loop B — Xu degradation:
        fd accumulated via Xu rainflow + calendar each year.
        SoH = 1 - sei_capacity_loss(fd_cum_xu).
        Replace when SoH < REPL_THRESHOLD.  Reset fd_cum and SoH on replacement.

      Loop C — Shi degradation:
        Same structure as Loop B but fd accumulated via Shi polynomial.
        Calendar term added from Xu (Shi has no calendar model of its own).

    Price/wind data is repeated identically for all 20 years.
    This is standard practice for design sweeps — the year's dispatch pattern
    is assumed stationary.  Year-to-year price variation is a future extension.

    NPV assembly (all three loops):
        NPV = -capex_at_t0
              + sum_k [ annual_rev_k * (1+r)^(-k) ]
              - sum_repl [ repl_cost * (1+r)^(-k_repl) ]

    Capex includes both energy and power components.
    Replacement cost covers energy capex only (power electronics not replaced).
    """
    n         = len(price_eur)
    eta_out   = params["eta_out"]
    eta_in    = params["eta_in"]
    soc_min   = params["soc_min"]
    soc_max   = params["soc_max"]
    p_max_g   = params["p_max"]
    e_cost_eur = params["e_cost_eur_per_mwh"]   # EUR/MWh
    p_cost_eur = params["p_cost_eur_per_mw"]    # EUR/MW
    repl_e_eur = params["repl_e_eur_per_mwh"]   # energy expansion
    repl_p_eur = params["repl_p_eur_per_mw"]    # power expansion

    # Initial capex (paid at t=0) and replacement cost — via wp2_econ, shared
    # with v5.3 and the annual mode (energy + power; replacement reuses BOP).
    capex_eur     = capex(e_cap, p_cap, e_cost_eur, p_cost_eur)
    repl_cost_eur = replacement_cost(e_cap, p_cap, repl_e_eur, repl_p_eur)

    # Per-year discount weights, SHIPP 19-yr convention (single flip point lives
    # in wp2_econ.discount_weights). weights[k-1] = (1+r)^-k for k = 1..N_YEARS-1.
    weights = discount_weights(DISCOUNT_RATE, N_YEARS)

    # ── Shared state for all three loops ──────────────────────────────────
    # Each loop tracks its own fd_cum and SoH independently.
    # Revenue is identical every year for loop A (SoH=1 always).
    # Loops B and C re-solve the LP with reduced e_cap each year.

    # Initialise per-loop accumulators
    # Dual-basis lifetime NPV accumulators (arbitrage + marginal), -capex at t0.
    npv_nd_arb  = -capex_eur;  npv_nd_marg  = -capex_eur
    npv_xu_arb  = -capex_eur;  npv_xu_marg  = -capex_eur
    npv_shi_arb = -capex_eur;  npv_shi_marg = -capex_eur

    soh_xu  = 1.0;  fd_cum_xu  = 0.0
    soh_shi = 1.0;  fd_cum_shi = 0.0

    n_repl_xu  = 0;  eol_xu  = None
    n_repl_shi = 0;  eol_shi = None

    # Tracking for output
    soh_traj_xu  = []
    soh_traj_shi = []
    fd_annual_xu = []
    fd_annual_shi = []

    # SoC inter-year pinning: the start-level fraction captured from each loop's
    # first solve, held constant across years, reset to None on replacement.
    # Per-loop because Xu and Shi capacities diverge after year 1.
    soc_frac = {"xu": None, "shi": None}

    for k in range(1, N_YEARS):              # 19 revenue years (SHIPP convention)
        discount_f = weights[k - 1]          # (1+r)^-k, from wp2_econ.discount_weights

        # ── Loop A: no degradation — always full capacity ──────────────────
        # The LP is identical every year so we only solve it once at k=1
        # and reuse the revenue for all 20 years (SoH=1, capacity never drops).
        if k == 1:
            stor_nd = Storage(
                e_cap=e_cap, p_cap=p_cap,
                eff_in=eta_in, eff_out=eta_out,
                e_cost=e_cost_eur, p_cost=p_cost_eur,
                dod=1.0 - soc_min,
            )
            stor_null_nd = Storage(e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
                                   e_cost=0, p_cost=0)
            price_ts  = TimeSeries((price_eur).tolist(), DT)
            prod      = Production(TimeSeries(power_wind.tolist(), DT), p_cost=0.0)
            prod_null = Production(TimeSeries([0.0]*n, DT), p_cost=0.0)

            # Solve with n_year=N_YEARS to match evaluate_single_ecap exactly.
            # This means storage_p[0].data and os.revenue are computed on the
            # same basis as the annual sweep — consistent sign convention and
            # annuity scaling.
            os_nd = solve_lp_pyomo(
                price_ts, prod, prod_null, stor_nd, stor_null_nd,
                DISCOUNT_RATE, N_YEARS, P_MIN_MW, p_max_g, n,
                "gurobi", fixed_cap=True, soc_max1=soc_max,
            )

            # Single-year undiscounted revenue, both bases, from the full-cap
            # dispatch (via wp2_econ: arbitrage + marginal/curtailment).
            p_vec_nd    = np.array(os_nd.storage_p[0].data, dtype=float)
            p_prod_nd   = np.array(os_nd.production_p[0].data, dtype=float)
            rev_nd      = revenue_annual(price_eur, p_vec_nd, p_prod_nd,
                                         power_wind, p_max_g, n, DT)
            raw_arb_nd  = rev_nd["arbitrage"]
            raw_marg_nd = rev_nd["marginal"]
            # P2 guard reference: SHIPP a_npv (arbitrage) for this dispatch. The
            # assembled no-deg arbitrage NPV must equal this (checked after loop).
            if RECON_CHECK:
                a_npv_nd_eur = os_nd.get_added_npv(DISCOUNT_RATE, N_YEARS) * 1e6

        npv_nd_arb  += raw_arb_nd  * discount_f
        npv_nd_marg += raw_marg_nd * discount_f

        # ── Loops B & C: degradation — solve LP with degraded capacity ─────
        # B — Xu effective capacity
        e_cap_eff_xu  = e_cap * soh_xu
        # C — Shi effective capacity
        e_cap_eff_shi = e_cap * soh_shi

        for loop_id, e_eff in [("xu", e_cap_eff_xu), ("shi", e_cap_eff_shi)]:
            # Rebuild Storage with this year's effective capacity.
            # Use actual e_cost/p_cost so the LP objective is on the same
            # basis as evaluate_single_ecap (matching sign convention in p_vec).
            stor = Storage(
                e_cap=e_eff, p_cap=p_cap,
                eff_in=eta_in, eff_out=eta_out,
                e_cost=e_cost_eur, p_cost=p_cost_eur,
                dod=1.0 - soc_min,
            )
            stor_null = Storage(e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
                                e_cost=0, p_cost=0)

            # SoC inter-year pinning: hold the start level at the fraction
            # captured from this loop's first solve (None until then, and after
            # each replacement). Scales with the degraded capacity, so the SoC
            # band fraction stays constant year to year. Closure (e[0]==e[n]) is
            # already enforced by the kernel; this fixes the absolute level.
            e_start1 = (soc_frac[loop_id] * e_eff
                        if soc_frac[loop_id] is not None else None)

            try:
                os_deg = solve_lp_pyomo(
                    price_ts, prod, prod_null, stor, stor_null,
                    DISCOUNT_RATE, N_YEARS, P_MIN_MW, p_max_g, n,
                    "gurobi", fixed_cap=True, soc_max1=soc_max,
                    e_start1=e_start1,
                )
            except RuntimeError:
                # Battery too small to operate — stop this loop early
                break

            p_vec = np.array(os_deg.storage_p[0].data, dtype=float)
            e_vec = np.array(os_deg.storage_e[0].data, dtype=float)

            # Capture the SoC fraction from this loop's first (re)built solve.
            if soc_frac[loop_id] is None:
                soc_frac[loop_id] = os_deg.soc_final / e_eff

            # Single-year undiscounted revenue, both bases, at this year's
            # degraded capacity (via wp2_econ).
            p_prod   = np.array(os_deg.production_p[0].data, dtype=float)
            rev      = revenue_annual(price_eur, p_vec, p_prod,
                                      power_wind, p_max_g, n, DT)
            raw_arb  = rev["arbitrage"]
            raw_marg = rev["marginal"]

            if loop_id == "xu":
                npv_xu_arb  += raw_arb  * discount_f
                npv_xu_marg += raw_marg * discount_f
            else:
                npv_shi_arb  += raw_arb  * discount_f
                npv_shi_marg += raw_marg * discount_f

            # ── Degradation this year ──────────────────────────────────────
            xu_res = analyze_degradation(
                storage_p=p_vec.tolist(),
                storage_e=e_vec.tolist(),
                e_cap_nominal=e_eff,
                battery_params=params["bat_params"],
                dt_hours=DT,
                T_cell_C=params["T_cell_C"],
            )
            fd_xu_yr  = xu_res["fd"] if INCLUDE_CALENDAR else xu_res["fd_cycle"]
            fd_cal_yr = xu_res["fd_calendar"]   # used by Shi branch below

            if loop_id == "xu":
                fd_cum_xu += fd_xu_yr
                soh_xu = 1.0 - sei_capacity_loss(fd_cum_xu)
                soh_traj_xu.append(soh_xu)
                fd_annual_xu.append(fd_xu_yr)

                # Replacement check
                if soh_xu < REPL_THRESHOLD:
                    if eol_xu is None:
                        eol_xu = k
                    npv_xu_arb  -= repl_cost_eur * discount_f
                    npv_xu_marg -= repl_cost_eur * discount_f
                    n_repl_xu += 1
                    fd_cum_xu = 0.0
                    soh_xu    = 1.0
                    soc_frac["xu"] = None   # fresh battery re-establishes its SoC band

            else:  # Shi
                shi_res = analyze_degradation_shi(
                    storage_p=p_vec.tolist(),
                    storage_e=e_vec.tolist(),
                    e_cap_nominal=e_eff,
                    battery_params=params["bat_params"],
                    shi_fit=params["shi_fit"],
                    T_cell_C=params["T_cell_C"],
                    dt_hours=DT,
                )
                fd_shi_yr = shi_res["fd_shi"]
                # Add calendar term from Xu (Shi has no calendar model)
                fd_shi_total_yr = (fd_shi_yr + fd_cal_yr) if INCLUDE_CALENDAR \
                                  else fd_shi_yr

                fd_cum_shi += fd_shi_total_yr
                soh_shi = 1.0 - sei_capacity_loss(fd_cum_shi)
                soh_traj_shi.append(soh_shi)
                fd_annual_shi.append(fd_shi_total_yr)

                if soh_shi < REPL_THRESHOLD:
                    if eol_shi is None:
                        eol_shi = k
                    npv_shi_arb  -= repl_cost_eur * discount_f
                    npv_shi_marg -= repl_cost_eur * discount_f
                    n_repl_shi += 1
                    fd_cum_shi = 0.0
                    soh_shi    = 1.0
                    soc_frac["shi"] = None   # fresh battery re-establishes its SoC band

    # ── P2 GUARD: lifetime no-deg (arbitrage) must equal annual a_npv ────────
    # npv_nd = -capex + raw_rev_nd * sum(weights) = -capex + arb_rev * annuity_19,
    # which is exactly what get_added_npv returns for the same full-cap dispatch.
    # diff ~ 0 confirms the horizon / capex / discounting assembly is correct.
    if RECON_CHECK:
        diff_nd = npv_nd_arb - a_npv_nd_eur
        print(f"    [P2 guard E={e_cap:>4.0f}] npv_nd={npv_nd_arb*1e-6:8.4f}  "
              f"a_npv={a_npv_nd_eur*1e-6:8.4f}  diff={diff_nd*1e-6:+8.4f} MEUR")

    # ── Headline basis selection (HEADLINE_BASIS; shared with v5.3 + annual) ──
    # Both bases are always computed; this only chooses which one the existing
    # downstream keys (npv_no_deg/with_xu/with_shi) — and thus the optimum
    # selection, report, and plots — report as the headline.
    if HEADLINE_BASIS == "arbitrage":
        npv_no_deg, npv_with_xu, npv_with_shi = npv_nd_arb, npv_xu_arb, npv_shi_arb
    else:  # "marginal"
        npv_no_deg, npv_with_xu, npv_with_shi = npv_nd_marg, npv_xu_marg, npv_shi_marg

    return {
        # Design point
        "e_cap":          e_cap,
        "p_cap":          p_cap,
        # Lifetime NPVs (EUR) — headline basis = HEADLINE_BASIS
        "npv_no_deg":     npv_no_deg,
        "npv_with_xu":    npv_with_xu,
        "npv_with_shi":   npv_with_shi,
        # Explicit bases (both always computed, regardless of headline)
        "npv_no_deg_arb":   npv_nd_arb,  "npv_with_xu_arb":  npv_xu_arb,  "npv_with_shi_arb":  npv_shi_arb,
        "npv_no_deg_marg":  npv_nd_marg, "npv_with_xu_marg": npv_xu_marg, "npv_with_shi_marg": npv_shi_marg,
        # Capex (for reference)
        "capex_eur":      capex_eur,
        # Replacement info
        "n_repl_xu":      n_repl_xu,
        "n_repl_shi":     n_repl_shi,
        "eol_year_xu":    eol_xu  if eol_xu  is not None else float("nan"),
        "eol_year_shi":   eol_shi if eol_shi is not None else float("nan"),
        # Final SoH at year 20 (or end of last complete cycle)
        "soh_final_xu":   soh_traj_xu[-1]  if soh_traj_xu  else float("nan"),
        "soh_final_shi":  soh_traj_shi[-1] if soh_traj_shi else float("nan"),
        # Mean annual fd across all years (useful for heatmaps)
        "fd_mean_xu":     float(np.mean(fd_annual_xu))  if fd_annual_xu  else float("nan"),
        "fd_mean_shi":    float(np.mean(fd_annual_shi)) if fd_annual_shi else float("nan"),
    }


def _build_viable_set(
    annual_results: List[dict],
    p_max: float,
) -> Tuple[set, dict]:
    """Identify (E, P) pairs worth evaluating in lifetime mode.

    Applies three filters to the annual sweep results:
      1. Duration guard: skip E/P < LIFETIME_MIN_DURATION_H or > LIFETIME_MAX_DURATION_H
      2. NPV floor: keep if annual Xu NPV ≥ floor × max(Xu) OR annual Shi NPV ≥ floor × max(Shi)
         (We don't filter on no-deg because it's linear and not the optimisation target.)
      3. Always keep the annual optima themselves (safety net)

    Returns
    -------
    viable : set of (e_cap, p_cap) tuples
    stats  : dict with filtering diagnostics for logging
    """
    # Find annual maxima for the two degradation models
    max_xu  = max(r.get("npv_with_xu",  -np.inf) for r in annual_results)
    max_shi = max(r.get("npv_with_shi", -np.inf) for r in annual_results)
    floor_xu  = LIFETIME_NPV_FLOOR * max_xu
    floor_shi = LIFETIME_NPV_FLOOR * max_shi

    # Always-include: the annual optima for each model
    solved   = [r for r in annual_results if not r.get("pruned", False)]
    best_xu  = max(solved, key=lambda r: r.get("npv_with_xu",  -np.inf))
    best_shi = max(solved, key=lambda r: r.get("npv_with_shi", -np.inf))
    best_nd  = max(solved, key=lambda r: r.get("npv_no_deg",   -np.inf))
    always_keep = {
        (best_xu["e_cap"],  best_xu["p_cap"]),
        (best_shi["e_cap"], best_shi["p_cap"]),
        (best_nd["e_cap"],  best_nd["p_cap"]),
    }

    viable = set()
    n_duration_skip = 0
    n_npv_skip = 0

    for r in annual_results:
        e = r["e_cap"]
        p = min(r["p_cap"], p_max)
        key = (e, p)

        # Filter 1: duration guard
        dur = e / p if p > 0 else 0.0
        if dur < LIFETIME_MIN_DURATION_H or dur > LIFETIME_MAX_DURATION_H:
            if key not in always_keep:
                n_duration_skip += 1
                continue

        # Filter 2: NPV floor (must pass for at least one degradation model)
        npv_xu  = r.get("npv_with_xu",  -np.inf)
        npv_shi = r.get("npv_with_shi", -np.inf)
        if npv_xu < floor_xu and npv_shi < floor_shi:
            if key not in always_keep:
                n_npv_skip += 1
                continue

        viable.add(key)

    # Safety: ensure always_keep points are included
    viable |= always_keep

    stats = {
        "total_annual": len(annual_results),
        "n_duration_skip": n_duration_skip,
        "n_npv_skip": n_npv_skip,
        "n_viable": len(viable),
        "floor_xu_meur": floor_xu / 1e6,
        "floor_shi_meur": floor_shi / 1e6,
    }
    return viable, stats


def _nan_result_20year(e_cap: float, p_cap: float) -> dict:
    """Return a placeholder result dict with NaN values for skipped points."""
    return {
        "e_cap": e_cap, "p_cap": p_cap,
        "npv_no_deg": float("nan"), "npv_with_xu": float("nan"),
        "npv_with_shi": float("nan"), "capex_eur": float("nan"),
        "n_repl_xu": 0, "n_repl_shi": 0,
        "eol_year_xu": float("nan"), "eol_year_shi": float("nan"),
        "soh_final_xu": float("nan"), "soh_final_shi": float("nan"),
        "fd_mean_xu": float("nan"), "fd_mean_shi": float("nan"),
    }


def run_sweep_20year(
    power_wind: np.ndarray,
    price_eur:  np.ndarray,
    params:     dict,
    annual_results: List[dict] | None = None,
) -> List[dict]:
    """Nested loop calling evaluate_20year for every viable (E_cap, P_cap) point.

    If annual_results is provided, applies smart filtering (duration + NPV floor)
    to skip non-competitive points.  Otherwise runs the full grid.

    Each evaluated point runs 2×20 + 1 = 41 LP solves.  With filtering this
    typically reduces the grid from 88 to ~30–35 points → ~1.5–2 hours.
    """
    # ── Build viable set if annual results available ──────────────────────
    if annual_results is not None:
        viable, stats = _build_viable_set(annual_results, params["p_max"])
        print(f"\n  ── Smart filtering ──")
        print(f"  Annual points:     {stats['total_annual']}")
        print(f"  Duration skipped:  {stats['n_duration_skip']}")
        print(f"  NPV floor skipped: {stats['n_npv_skip']}  "
              f"(Xu floor={stats['floor_xu_meur']:.1f} MEUR, "
              f"Shi floor={stats['floor_shi_meur']:.1f} MEUR)")
        print(f"  Viable points:     {stats['n_viable']}")
        print(f"  LP solves:         ~{stats['n_viable'] * 41}  "
              f"(est. {stats['n_viable'] * 41 * 5 / 3600:.1f} hours)")
        print()
    else:
        viable = None  # run everything

    total = len(E_CAP_GRID) * len(P_CAP_GRID)
    results = []
    done = 0
    n_evaluated = 0
    t_start = time.perf_counter()

    for e_cap in E_CAP_GRID:
        for p_cap in P_CAP_GRID:
            done += 1
            p_cap_eff = min(p_cap, params["p_max"])

            # ── Check if this point should be evaluated ───────────────────
            if viable is not None and (e_cap, p_cap_eff) not in viable:
                print(f"  [{done:3d}/{total}]  E={e_cap:5.0f} MWh  P={p_cap_eff:5.0f} MW  "
                      f"-> SKIPPED (filtered)")
                results.append(_nan_result_20year(e_cap, p_cap_eff))
                continue

            n_evaluated += 1
            elapsed = time.perf_counter() - t_start
            eta = (elapsed / n_evaluated) * (len(viable or E_CAP_GRID) * (1 if viable else len(P_CAP_GRID)) - n_evaluated) if n_evaluated > 1 else 0.0
            print(f"  [{done:3d}/{total}]  E={e_cap:5.0f} MWh  P={p_cap_eff:5.0f} MW  "
                  f"[{n_evaluated}/{len(viable) if viable else total}]  "
                  f"ETA={eta/60:.1f}min ...", end="", flush=True)

            r = evaluate_20year(e_cap, p_cap_eff, power_wind, price_eur, params)
            results.append(r)

            print(f"  NPV_nd={r['npv_no_deg']/1e6:.2f}M  "
                  f"NPV_xu={r['npv_with_xu']/1e6:.2f}M  "
                  f"NPV_shi={r['npv_with_shi']/1e6:.2f}M  "
                  f"repl_xu={r['n_repl_xu']}  repl_shi={r['n_repl_shi']}")

    elapsed_total = time.perf_counter() - t_start
    print(f"\n  Evaluated {n_evaluated}/{total} points in {elapsed_total/60:.1f} min "
          f"(skipped {total - n_evaluated})")
    return results


def save_results_20year(results: List[dict], params: dict, n: int):
    """Save lifetime sweep results to CSV, JSON sidecar, and text report."""
    import json
    import pandas as pd
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── CSV ──────────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / f"sizing_lifetime_sweep_{n}h_{N_YEARS}yr_{ts}.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path.name}")

    # ── JSON sidecar ──────────────────────────────────────────────────────
    config_snap = {
        "E_CAP_GRID":        E_CAP_GRID,
        "P_CAP_GRID":        P_CAP_GRID,
        "INCLUDE_CALENDAR":  INCLUDE_CALENDAR,
        "RUN_HOURS":         n,
        "N_YEARS":           N_YEARS,
        "REPL_THRESHOLD":    REPL_THRESHOLD,
        "DISCOUNT_RATE":     DISCOUNT_RATE,
        "LIFETIME_NPV_FLOOR": LIFETIME_NPV_FLOOR,
        "LIFETIME_MIN_DURATION_H": LIFETIME_MIN_DURATION_H,
        "LIFETIME_MAX_DURATION_H": LIFETIME_MAX_DURATION_H,
        "soc_min":           params["soc_min"],
        "soc_max":           params["soc_max"],
        "eta_out":           params["eta_out"],
        "e_cost_eur_kwh":    params["e_cost_eur_per_kwh"],
        "k3":                params["k3"],
        "k4":                params["k4"],
    }
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(config_snap, indent=2), encoding="utf-8")
    print(f"  JSON: {json_path.name}")

    # ── Text report ──────────────────────────────────────────────────────
    report_path = OUTPUT_DIR / f"sizing_lifetime_report_{n}h_{N_YEARS}yr_{ts}.txt"
    lines = []
    w = lines.append

    w("=" * 110)
    w(f"PLAN B: Lifetime sweep, {N_YEARS}-year NPV vs (E_cap, P_cap), three degradation scenarios")
    w("=" * 110)
    w("")
    w("CONFIGURATION")
    w(f"  Horizon per year:  {n} hours ({n/24:.1f} days), repeated {N_YEARS} years")
    w(f"  E_cap grid:        {E_CAP_GRID} MWh")
    w(f"  P_cap grid:        {P_CAP_GRID} MW")
    w(f"  Grid points:       {len(results)}")
    w(f"  Calendar aging:    {INCLUDE_CALENDAR}")
    w(f"  Replacement SoH:   {REPL_THRESHOLD*100:.0f}%")
    w(f"  Discount rate:     {DISCOUNT_RATE*100:.1f}%")
    w(f"  SoC window:        {params['soc_min']*100:.0f}%–{params['soc_max']*100:.0f}%")
    w(f"  RTE(ac):           {params['rte_ac']*100:.1f}%  (eta_symmetric = {params['eta_out']:.4f})")
    w("")
    w("=" * 110)
    w("RESULTS")
    w("=" * 110)
    w("")
    header = (f"  {'E':>5}  {'P':>5}  {'E/P':>4}  "
              f"{'NPV_noDeg':>10}  {'NPV_Xu':>10}  {'NPV_Shi':>10}  "
              f"{'Repl_Xu':>7}  {'Repl_Shi':>8}  "
              f"{'EoL_Xu':>6}  {'EoL_Shi':>7}  "
              f"{'SoH_Xu':>6}  {'SoH_Shi':>7}")
    units  = (f"  {'MWh':>5}  {'MW':>5}  {'h':>4}  "
              f"{'MEUR':>10}  {'MEUR':>10}  {'MEUR':>10}  "
              f"{'n':>7}  {'n':>8}  "
              f"{'yr':>6}  {'yr':>7}  "
              f"{'%':>6}  {'%':>7}")
    w(header)
    w(units)
    w("  " + "-" * 107)

    for r in results:
        ep = r['e_cap'] / r['p_cap']
        eol_xu  = f"{r['eol_year_xu']:.0f}"  if not np.isnan(r['eol_year_xu'])  else "none"
        eol_shi = f"{r['eol_year_shi']:.0f}" if not np.isnan(r['eol_year_shi']) else "none"
        w(f"  {r['e_cap']:5.0f}  {r['p_cap']:5.0f}  {ep:4.1f}  "
          f"{r['npv_no_deg']/1e6:10.2f}  "
          f"{r['npv_with_xu']/1e6:10.2f}  "
          f"{r['npv_with_shi']/1e6:10.2f}  "
          f"{r['n_repl_xu']:7.0f}  {r['n_repl_shi']:8.0f}  "
          f"{eol_xu:>6}  {eol_shi:>7}  "
          f"{r['soh_final_xu']*100:6.1f}  {r['soh_final_shi']*100:7.1f}")

    w("")
    w("OPTIMAL DESIGN POINT")
    w("")
    for label, key in [("No degradation", "npv_no_deg"),
                       ("Xu degradation", "npv_with_xu"),
                       ("Shi degradation", "npv_with_shi")]:
        best = max(results, key=lambda r, k=key: r[k] if np.isfinite(r[k]) else -np.inf)
        ep = best['e_cap'] / best['p_cap']
        w(f"  {label:<18s}: E={best['e_cap']:.0f} MWh  P={best['p_cap']:.0f} MW  "
          f"E/P={ep:.1f}h  NPV={best[key]/1e6:.2f} MEUR")

    w("")
    w("=" * 110)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {report_path.name}")

    return csv_path, report_path, ts


def run_sweep_2d(
    power_wind: np.ndarray,
    price_eur:  np.ndarray,
    params:     dict,
) -> List[dict]:
    """Run the 2D parameter sweep over all (E_cap, P_cap) combinations.
    
    Uses intelligent fallback approximation for extreme durations to keep 
    contour grids unbroken without executing slow LP solves.
    """
    total = len(E_CAP_GRID) * len(P_CAP_GRID)
    results = []
    done = 0
    t_sweep_start = time.perf_counter()

    for e_cap in E_CAP_GRID:
        for p_cap in P_CAP_GRID:
            done += 1
            p_cap_eff = min(p_cap, params["p_max"])
            duration_h = e_cap / p_cap_eff if p_cap_eff > 0 else 0.0

            elapsed = time.perf_counter() - t_sweep_start
            eta = (elapsed / done) * (total - done) if done > 1 else 0.0

            # ── Pruned: outside the considered duration band [1h, 8h]. ──
            # We do NOT solve or fabricate these — emit a flagged NaN row so they're carried in the CSV but excluded from plots and the optimum.
            is_pruned = (duration_h < LIFETIME_MIN_DURATION_H) or (duration_h > LIFETIME_MAX_DURATION_H)
            if is_pruned:
                print(f"  [{done:3d}/{total}]  E={e_cap:5.0f} MWh  P={p_cap_eff:5.0f} MW  "
                      f"Duration={duration_h:.1f}h -> PRUNED (not solved)")
                r_pruned = {
                    "e_cap": e_cap, "p_cap": p_cap_eff,
                    "pruned": True,
                    "revenue_eur": float("nan"), "revenue_marg_eur": float("nan"),
                    "capex_eur": (params["p_cost_eur_per_mw"] * p_cap_eff
                                  + params["e_cost_eur_per_mwh"] * e_cap),
                    "npv_no_deg": float("nan"), "npv_with_xu": float("nan"),
                    "npv_with_shi": float("nan"),
                    "npv_marg_no_deg": float("nan"), "npv_marg_xu": float("nan"),
                    "npv_marg_shi": float("nan"), "curtailment_value_eur": float("nan"),
                    "fd_xu_total": float("nan"), "fd_shi_total": float("nan"),
                    "efc": float("nan"), "solve_s": 0.0,
                }
                results.append(r_pruned)
                continue

            # ── Normal Execution Block ──
            print(f"  [{done:3d}/{total}]  E={e_cap:5.0f} MWh  P={p_cap_eff:5.0f} MW  "
                  f"ETA={eta:.0f}s ...", end="", flush=True)

            r = evaluate_single_ecap(e_cap, p_cap_eff, power_wind, price_eur, params)
            results.append(r)
            
            print(f"  rev={r['revenue_eur']/1e3:7.0f}k  "
                  f"fd_xu={r['fd_xu_total']:.5f}  "
                  f"EFC={r['efc']:.0f}  t={r['solve_s']:.1f}s")

    return results


def save_results_2d(results: List[dict], params: dict, n: int) -> None:
    """Save 2D sweep results to CSV, JSON sidecar, and text report."""
    import json
    import pandas as pd
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── CSV ──────────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / f"sizing_sweep_{n}h_{ts}.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path.name}")

    # ── JSON sidecar — config snapshot for replot verification ───────────
    # Saved next to the CSV with the same stem.  load_and_verify() reads
    # this to check whether the current toggles match the saved run.
    config_snap = {
        "E_CAP_GRID":       E_CAP_GRID,
        "P_CAP_GRID":       P_CAP_GRID,
        "INCLUDE_CALENDAR": INCLUDE_CALENDAR,
        "RUN_HOURS":        n,
        "soc_min":          params["soc_min"],
        "soc_max":          params["soc_max"],
        "eta_out":          params["eta_out"],
        "e_cost_eur_kwh":   params["e_cost_eur_per_kwh"],
        "k3":               params["k3"],
        "k4":               params["k4"],
    }
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(config_snap, indent=2), encoding="utf-8")
    print(f"  JSON: {json_path.name}")

    # ── Text report ──────────────────────────────────────────────────────
    report_path = OUTPUT_DIR / f"sizing_report_{n}h_{ts}.txt"

    lines = []
    w = lines.append

    w("=" * 100)
    w("PLAN B 2D: Parameter sweep, NPV vs (E_cap, P_cap) with degradation")
    w("=" * 100)
    w("")
    w("CONFIGURATION")
    w(f"  Horizon:         {n} hours ({n/24:.1f} days)")
    w(f"  E_cap grid:      {E_CAP_GRID} MWh")
    w(f"  P_cap grid:      {P_CAP_GRID} MW")
    w(f"  Grid points:     {len(E_CAP_GRID) * len(P_CAP_GRID)}")
    w(f"  Calendar aging:  {INCLUDE_CALENDAR}")
    w(f"  SoC window:      {params['soc_min']*100:.0f}%–{params['soc_max']*100:.0f}%")
    w(f"  RTE(ac):         {params['rte_ac']*100:.1f}%  (eta_symmetric = {params['eta_out']:.4f})")
    w(f"  Shi k3:          {params['k3']:.4e}")
    w(f"  Shi k4:          {params['k4']:.4f}")
    w(f"  Cost:            {params['e_cost_eur_per_kwh']:.0f} EUR/kWh")

    w("")
    w("=" * 100)
    w("RESULTS")
    w("=" * 100)
    w("")
    w(f"  {'E_cap':>6s}  {'P_cap':>6s}  {'E/P_h':>5s}  {'Revenue':>10s}  "
      f"{'NPV_noDeg':>10s}  {'NPV_Xu':>10s}  {'NPV_Shi':>10s}  "
      f"{'fd_Xu':>8s}  {'fd_Shi':>8s}  {'EFC':>5s}")
    w(f"  {'MWh':>6s}  {'MW':>6s}  {'h':>5s}  {'kEUR':>10s}  "
      f"{'kEUR':>10s}  {'kEUR':>10s}  {'kEUR':>10s}  "
      f"{'':>8s}  {'':>8s}  {'':>5s}")
    w(f"  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*10}  "
      f"{'─'*10}  {'─'*8}  {'─'*8}  {'─'*5}")

    for r in results:
        ep_h = r['e_cap'] / r['p_cap'] if r['p_cap'] > 0 else float('inf')
        w(f"  {r['e_cap']:6.0f}  {r['p_cap']:6.0f}  {ep_h:5.1f}  "
          f"{r['revenue_eur']/1e3:10.0f}  "
          f"{r['npv_no_deg']/1e3:10.0f}  "
          f"{r['npv_with_xu']/1e3:10.0f}  "
          f"{r['npv_with_shi']/1e3:10.0f}  "
          f"{r['fd_xu_total']:8.5f}  "
          f"{r['fd_shi_total']:8.5f}  "
          f"{r['efc']:5.0f}")

    # Best design point for each NPV metric
    w("")
    w("OPTIMAL DESIGN POINT")
    w("")
    for label, key in [("Without degradation", "npv_no_deg"),
                       ("With Xu degradation", "npv_with_xu"),
                       ("With Shi degradation", "npv_with_shi")]:
        best = max(results, key=lambda r: r[key])
        ep_h = best['e_cap'] / best['p_cap']
        w(f"  {label:<25s}:  E={best['e_cap']:.0f} MWh  P={best['p_cap']:.0f} MW  "
          f"E/P={ep_h:.1f}h  NPV={best[key]/1e6:.2f} MEUR")

    w("")
    w("=" * 100)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {report_path.name}")

    return csv_path, report_path, ts


def _to_grid(results: List[dict], key: str) -> np.ndarray:
    """Reshape flat results list into a 2D array (n_e × n_p) for heatmaps."""
    n_e = len(E_CAP_GRID)
    n_p = len(P_CAP_GRID)
    grid = np.full((n_e, n_p), np.nan)
    for r in results:
        i = E_CAP_GRID.index(r["e_cap"]) if r["e_cap"] in E_CAP_GRID else None
        # p_cap may have been clamped to p_max — find nearest grid value
        p_diffs = [abs(r["p_cap"] - p) for p in P_CAP_GRID]
        j = int(np.argmin(p_diffs))
        if i is not None:
            grid[i, j] = r[key]
    return grid


# ════════════════════════════════════════════════════════════════════════════
# 3c.  QUADRATIC SURFACE FITTING
# ════════════════════════════════════════════════════════════════════════════

def fit_quadratic_surface(
    results: List[dict],
    npv_key: str,
    label: str = "",
) -> dict | None:
    """Fit a 2D quadratic to NPV(E, P) and find the continuous optimum.

    Model: NPV = a0 + a1*E + a2*P + a3*E² + a4*P² + a5*E*P

    Optimum: solve  dNPV/dE = 0,  dNPV/dP = 0  →  2×2 linear system.
    Valid only if the Hessian is negative definite (both eigenvalues < 0),
    meaning the surface is concave (has a maximum, not a saddle or minimum).

    Returns dict with fitted optimum, R², and diagnostics, or None if the
    fit fails (too few points, non-concave surface, etc).
    """
    # Filter out NaN results (from pruned/skipped points)
    valid = [(r["e_cap"], r["p_cap"], r[npv_key])
             for r in results if np.isfinite(r.get(npv_key, float("nan")))]
    if len(valid) < 6:
        print(f"  [{label}] Too few valid points ({len(valid)}) for quadratic fit.")
        return None

    E = np.array([v[0] for v in valid])
    P = np.array([v[1] for v in valid])
    Z = np.array([v[2] for v in valid])

    # Normalise for numerical stability
    E_mean, E_std = E.mean(), max(E.std(), 1.0)
    P_mean, P_std = P.mean(), max(P.std(), 1.0)
    En = (E - E_mean) / E_std
    Pn = (P - P_mean) / P_std

    # Design matrix: [1, En, Pn, En², Pn², En·Pn]
    A = np.column_stack([np.ones(len(En)), En, Pn, En**2, Pn**2, En * Pn])

    coeffs, residuals, rank, sv = np.linalg.lstsq(A, Z, rcond=None)
    a0, a1, a2, a3, a4, a5 = coeffs

    # R²
    Z_pred = A @ coeffs
    ss_res = np.sum((Z - Z_pred) ** 2)
    ss_tot = np.sum((Z - Z.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Hessian in normalised coordinates
    H = np.array([[2 * a3, a5],
                   [a5, 2 * a4]])
    eigvals = np.linalg.eigvalsh(H)
    is_concave = bool(np.all(eigvals < 0))

    # Grid optimum (always available as fallback)
    best_grid = max(results, key=lambda r: r.get(npv_key, -np.inf) if np.isfinite(r.get(npv_key, -np.inf)) else -np.inf)

    if not is_concave:
        print(f"  [{label}] Hessian NOT negative definite "
              f"(eigenvalues: {eigvals[0]:.2e}, {eigvals[1]:.2e}).")
        print(f"    Surface is not concave — grid optimum is more reliable.")
        return {
            "method": "grid_only",
            "e_opt": best_grid["e_cap"],
            "p_opt": best_grid["p_cap"],
            "npv_opt": best_grid[npv_key],
            "r_squared": r_squared,
            "eigvals": eigvals.tolist(),
            "is_concave": False,
            "in_range": True,
            "label": label,
        }

    # Solve for optimum: H @ [En*, Pn*] = -[a1, a2]
    opt_norm = np.linalg.solve(H, -np.array([a1, a2]))
    E_opt = opt_norm[0] * E_std + E_mean
    P_opt = opt_norm[1] * P_std + P_mean
    NPV_opt = float(np.array([1, opt_norm[0], opt_norm[1],
                                opt_norm[0]**2, opt_norm[1]**2,
                                opt_norm[0] * opt_norm[1]]) @ coeffs)

    # Sanity: is the optimum within a reasonable range of the grid?
    e_range = (min(E), max(E))
    p_range = (min(P), max(P))
    in_range = (e_range[0] * 0.8 <= E_opt <= e_range[1] * 1.2 and
                p_range[0] * 0.8 <= P_opt <= p_range[1] * 1.2)

    print(f"  [{label}] Quadratic fit: R²={r_squared:.4f}")
    print(f"    Grid optimum:       E={best_grid['e_cap']:.0f} MWh, "
          f"P={best_grid['p_cap']:.0f} MW, "
          f"NPV={best_grid[npv_key]/1e6:.2f} MEUR")
    print(f"    Continuous optimum: E*={E_opt:.1f} MWh, P*={P_opt:.1f} MW, "
          f"NPV*={NPV_opt/1e6:.2f} MEUR")
    if not in_range:
        print(f"    WARNING: optimum outside grid range — extrapolation, "
              f"treat with caution")

    return {
        "method": "quadratic",
        "e_opt": E_opt,
        "p_opt": P_opt,
        "npv_opt": NPV_opt,
        "e_opt_grid": best_grid["e_cap"],
        "p_opt_grid": best_grid["p_cap"],
        "npv_opt_grid": best_grid[npv_key],
        "r_squared": r_squared,
        "eigvals": eigvals.tolist(),
        "is_concave": True,
        "in_range": in_range,
        "label": label,
    }


def run_quadratic_fits(results: List[dict]) -> Dict[str, dict]:
    """Run quadratic fits for all three NPV models, return dict keyed by npv_key."""
    print("\n" + "=" * 78)
    print("QUADRATIC SURFACE FITTING")
    print("=" * 78)
    quad_fits: Dict[str, dict] = {}
    for label, key in [("Xu degradation", "npv_with_xu"),
                       ("Shi degradation", "npv_with_shi")]:
        qf = fit_quadratic_surface(results, key, label=label)
        if qf is not None:
            quad_fits[key] = qf
    return quad_fits


# ════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print(f"PLAN B: Parameter sweep, NPV vs (E_cap, P_cap)  [mode: {SWEEP_MODE}]")
    print("=" * 78)
    print(f"  E_cap grid:     {E_CAP_GRID} MWh")
    print(f"  P_cap grid:     {P_CAP_GRID} MW")
    print(f"  Grid points:    {len(E_CAP_GRID) * len(P_CAP_GRID)}")
    print(f"  Calendar aging: {INCLUDE_CALENDAR}")
    print(f"  Horizon:        {RUN_HOURS} hours")
    if SWEEP_MODE == "lifetime":
        print(f"  N years:        {N_YEARS}")
        print(f"  Repl threshold: SoH < {REPL_THRESHOLD*100:.0f}%")
        print(f"  NPV floor:      {LIFETIME_NPV_FLOOR*100:.0f}% of annual max")
        print(f"  Duration guard:  {LIFETIME_MIN_DURATION_H}–{LIFETIME_MAX_DURATION_H} h")
        print(f"  (Annual sweep runs first for smart filtering)")

    # ── Resolve which CSV to load, if any ────────────────────────────────
    csv_to_load: Path | None = None
    if REPLOT_CSV is not None:
        csv_to_load = Path(REPLOT_CSV)
    elif REPLOT_FROM_LAST:
        csv_to_load = find_latest_csv()

    # ── Branch: replot from CSV or run the sweep ──────────────────────────
    if csv_to_load is not None:
        print(f"\n[REPLOT MODE] Loading from CSV, LP sweep will be skipped.")
        results, params, n = load_and_verify(csv_to_load)
        stem_parts = csv_to_load.stem.split("_")
        ts = f"{stem_parts[-2]}_{stem_parts[-1]}_replot"

    elif SWEEP_MODE == "annual":
        print(f"\n[1/3] Loading data...")
        power_wind, price_eur, params = load_data(RUN_HOURS)
        n = len(price_eur)

        print(f"\n[2/3] Running annual 2D sweep "
              f"({len(E_CAP_GRID)*len(P_CAP_GRID)} points)...")
        t0 = time.perf_counter()
        results = run_sweep_2d(power_wind, price_eur, params)
        print(f"  Total sweep time: {time.perf_counter()-t0:.1f}s")

        print(f"\n[3/3] Saving results...")
        _, _, ts = save_results_2d(results, params, n)

    elif SWEEP_MODE == "lifetime":
        print(f"\n[1/4] Loading data...")
        power_wind, price_eur, params = load_data(RUN_HOURS)
        n = len(price_eur)

        # ── Stage 1: fast annual sweep for smart filtering ────────────────
        print(f"\n[2/4] Running annual sweep for smart filtering "
              f"({len(E_CAP_GRID)*len(P_CAP_GRID)} points)...")
        t0 = time.perf_counter()
        annual_results = run_sweep_2d(power_wind, price_eur, params)
        print(f"  Annual sweep: {time.perf_counter()-t0:.1f}s")

        # ── Stage 2: lifetime sweep on filtered points ────────────────────
        print(f"\n[3/4] Running lifetime 2D sweep (smart-filtered)...")
        t0 = time.perf_counter()
        results = run_sweep_20year(power_wind, price_eur, params,
                                   annual_results=annual_results)
        elapsed = time.perf_counter() - t0
        print(f"  Lifetime sweep: {elapsed/60:.1f} min")

        print(f"\n[4/4] Saving results...")
        _, _, ts = save_results_20year(results, params, n)

    else:
        raise ValueError(f"Unknown SWEEP_MODE: '{SWEEP_MODE}'. "
                         "Set to 'annual' or 'lifetime'.")

# ── Quadratic Surface Fitting ──────────────────────────────────────────
    quad_fits = run_quadratic_fits(results)

    # ── Print optimal design point ─────────────────────────────────────────
    print("\n" + "=" * 78)
    print("OPTIMAL DESIGN POINT")
    print("=" * 78)
    for label, key in [("Without degradation", "npv_no_deg"),
                       ("With Xu degradation", "npv_with_xu"),
                       ("With Shi degradation", "npv_with_shi")]:
        best = max(results, key=lambda r, k=key: r[k] if np.isfinite(r[k]) else -np.inf)
        ep_h = best["e_cap"] / best["p_cap"]
        line = (f"  {label:<25s}:  E={best['e_cap']:.0f} MWh  P={best['p_cap']:.0f} MW  "
                f"E/P={ep_h:.1f}h  NPV={best[key]/1e6:.2f} MEUR")
        if key in quad_fits and quad_fits[key].get("is_concave"):
            qf = quad_fits[key]
            line += (f"  |  quad: E*={qf['e_opt']:.0f}  P*={qf['p_opt']:.0f}  "
                     f"NPV*={qf['npv_opt']/1e6:.2f}  R²={qf['r_squared']:.3f}")
        print(line)

    print("\nDone.")
    return results


if __name__ == "__main__":
    main()