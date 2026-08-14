"""Monolithic NLP: degradation embedded directly in the dispatch objective.

Fork origin: path3.py

Solves dispatch and rainflow degradation as a single non-linear program, rather than nesting a degradation calculation around an LP. The SHIPP kernel
supplies the LP baseline and the constraint set; degradation is added to the objective and the whole problem is passed to scipy.optimize.minimize with
trust-constr or SLSQP.

This is reported as a negative result. The rainflow degradation cost has a discontinuous gradient at cycle-topology boundaries, so the solver converges
on only a minority of days, and apparent gains on the remainder are artefacts of premature termination. Objective scaling and a finite-difference Hessian
both reduce but do not remove the problem.

Variable layout (SHIPP lp_alt formulation, 5*T + 6 total):
    x[0:T]          stor1_p     battery power (positive = discharge)
    x[T:2*T]        stor2_p     null storage (always 0)
    x[2*T:3*T]      p_curtailed curtailed power (always 0, no production)
    x[3*T:4*T+1]    stor1_e     battery energy state (T+1 values)
    x[4*T+1:5*T+2]  stor2_e     null storage energy (always 0)
    x[5*T+2:5*T+6]  capacities  [p_cap1, e_cap1, p_cap2, e_cap2] (fixed)

Usage:
    python scripts/run_nlp_monolithic.py --year 2022 --slot D43       # one day
    python scripts/run_nlp_monolithic.py --year 2019 --slot W1        # one week
    python scripts/run_nlp_monolithic.py --year 2022 --month 7        # one month
    python scripts/run_nlp_monolithic.py --year 2019 --month full     # full year
    python scripts/run_nlp_monolithic.py --year 2022 --all-days       # every day
    python scripts/run_nlp_monolithic.py --year 2022 --slot D43 --no-scale
    python scripts/run_nlp_monolithic.py --year 2022 --slot D43 --hess 2-point

Objective scaling (f and grad f divided by the LP revenue) is on by default and improves conditioning by five to six orders of magnitude; --no-scale disables
it for comparison runs.

Author: Thodoris Tsonopoulos, MSc thesis, TU Delft Wind Energy
Based on: Jenna Iori, examples/example_degradation.py, SHIPP feature_degradation
"""

from __future__ import annotations

import argparse
import json as _json
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linprog, minimize, LinearConstraint

# -- SHIPP imports ---------------------------------------------------------
from shipp.kernel import build_lp_obj_revenues, build_lp_cst_sparse
from shipp.components import Storage

# -- Repository paths ------------------------------------------------------
from degradation.paths import (
    HPP_YAML,
    PRICE_CSV_2019,
    PRICE_CSV_2022,
    results_dir as make_results_dir,
)

from degradation.site import quick_setup, get_wake_model
from degradation.economics import eta_symmetric

from degradation.xu import (
    XuModelParams,
    XU_LMO,
    ShiPolynomialFit,
    fit_shi_polynomial,
    phi_shi,
    phi_shi_prime,
    rainflow_cycle_counting,
    s_soc,
    s_temp,
    ft_calendar,
)

_RUNNER_NAME = "path3"


# ══════════════════════════════════════════════════════════════════════════
# 1.  DEGRADATION FUNCTIONS  (validated, unchanged)
# ══════════════════════════════════════════════════════════════════════════

def compute_f_deg(
    storage_e: np.ndarray,
    e_cap: float,
    shi_fit: ShiPolynomialFit,
    T_C: float = 25.0,
    p: XuModelParams = XU_LMO,
) -> Tuple[float, List[Dict]]:
    """Total Shi cycling degradation from a stored-energy trace."""
    cycles = rainflow_cycle_counting(storage_e, e_cap)
    k3, k4 = shi_fit.k3, shi_fit.k4
    S_T = float(s_temp(T_C, p))

    f_total = 0.0
    for c in cycles:
        delta = c["dod"]
        if delta < 1e-12:
            continue
        phi_val = float(phi_shi(delta, k3, k4))
        S_sigma = float(s_soc(c["soc_mean"], p))
        f_total += phi_val * S_sigma * S_T * c["count"]

    return f_total, cycles


def compute_df_de(
    storage_e: np.ndarray,
    cycles: List[Dict],
    e_cap: float,
    shi_fit: ShiPolynomialFit,
    T_C: float = 25.0,
    p: XuModelParams = XU_LMO,
) -> np.ndarray:
    """Sparse subgradient df_deg/de_t — nonzero at SoC turning points only."""
    n = len(storage_e)
    df_de = np.zeros(n, dtype=np.float64)

    k3, k4 = shi_fit.k3, shi_fit.k4
    S_T = float(s_temp(T_C, p))

    for cyc in cycles:
        delta = cyc["dod"]
        if delta < 1e-12:
            continue

        phi_prime = float(phi_shi_prime(delta, k3, k4))
        S_sigma   = float(s_soc(cyc["soc_mean"], p))
        kernel    = phi_prime * S_sigma * S_T * cyc["count"]

        i_s, i_e = cyc["i_start"], cyc["i_end"]
        i0 = max(min(i_s, i_e), 0)
        i1 = min(max(i_s, i_e), n - 1)
        segment = storage_e[i0 : i1 + 1]

        peak_idx   = i0 + int(np.argmax(segment))
        trough_idx = i0 + int(np.argmin(segment))

        df_de[peak_idx]   += kernel
        df_de[trough_idx] -= kernel

    df_de /= e_cap
    return df_de


# ══════════════════════════════════════════════════════════════════════════
# 2.  NLP OBJECTIVE + GRADIENT  (with scaling)
# ══════════════════════════════════════════════════════════════════════════

def make_nlp_functions(
    vec_obj: np.ndarray,
    n: int,
    e_cap_nominal: float,
    shi_fit: ShiPolynomialFit,
    w_deg: float,
    x0: np.ndarray,
    T_C: float = 25.0,
    obj_scale: float = 1.0,
    alpha: float | None = None,
):
    """Return (objective, gradient, state_dict) closures for minimize().

    Two normalisation modes (selected by "alpha"):

    alpha=None  — legacy w_deg mode (default):
        f(x) = (lp_cost + w_deg * f_deg) / obj_scale
        Same as before; obj_scale = |LP_revenue|.

    alpha=float — Jenna's dual-normalisation mode:
        f(x) = (1-alpha) * lp_cost / norm_rev
             + alpha     * f_deg   / norm_deg
        where norm_rev = |dot(vec_obj, x0)| and norm_deg = f_deg(x0) are computed ONCE from the LP solution x0 and held constant.
        Both terms are O(1) at x0, giving the degradation signal "alpha" weight regardless of the deg/rev ratio of that day.

    Parameters
    ----------
    x0 : np.ndarray
        LP solution used to compute normalisation constants.
    alpha : float or None
        If None, use w_deg/obj_scale mode.
        Suggested value: 0.2 (80% revenue, 20% degradation weight).
    """
    e_slice = slice(3*n, 4*n + 1)

    # ── Normalisation constants — computed ONCE from x0 ──────────────────
    if alpha is not None:
        norm_rev = max(abs(float(np.dot(vec_obj, x0))), 1e-12)
        f_deg_x0, _ = compute_f_deg(x0[e_slice], e_cap_nominal, shi_fit, T_C)
        norm_deg = max(float(f_deg_x0), 1e-12)
    else:
        inv_scale = 1.0 / obj_scale   # only needed for w_deg mode

    _state = {"n_obj": 0, "last_f_deg": 0.0}

    def objective(x):
        e = x[e_slice]
        lp_cost = float(np.dot(vec_obj, x))
        f_deg, _ = compute_f_deg(e, e_cap_nominal, shi_fit, T_C)
        _state["n_obj"] += 1
        _state["last_f_deg"] = f_deg
        if alpha is not None:
            return (1.0 - alpha) * lp_cost / norm_rev + alpha * f_deg / norm_deg
        return (lp_cost + w_deg * f_deg) * inv_scale

    def gradient(x):
        e = x[e_slice]
        _, cycles = compute_f_deg(e, e_cap_nominal, shi_fit, T_C)
        df_de = compute_df_de(e, cycles, e_cap_nominal, shi_fit, T_C)
        if alpha is not None:
            grad = (1.0 - alpha) * vec_obj / norm_rev
            grad = grad.copy()
            grad[e_slice] += alpha * df_de / norm_deg
        else:
            grad = vec_obj.copy()
            grad[e_slice] += w_deg * df_de
            grad = grad * inv_scale
        return grad

    return objective, gradient, _state


# ══════════════════════════════════════════════════════════════════════════
# 3.  LP BUILDER
# ══════════════════════════════════════════════════════════════════════════

def daily_revenue(price, storage_p, curtailed, wind, p_max, dt=1.0) -> Dict:
    """Single-window UNDISCOUNTED revenue [EUR], two bases (mirrors
    wp2_econ.revenue_annual but raw, with no annualisation).

        arbitrage : price . storage_p
        marginal  : price . (wind - curtailed + storage_p) - price . min(wind, p_max)

    With no wind (wind == 0, curtailed == 0) marginal collapses to arbitrage,
    so the battery-only path keeps its original headline number.
    """
    price     = np.asarray(price, float)
    storage_p = np.asarray(storage_p, float)
    curtailed = np.asarray(curtailed, float)
    wind      = np.asarray(wind, float)
    prod_after_curt = wind - curtailed
    wind_no_bat     = np.minimum(wind, p_max)
    arb  = dt * float(np.dot(price, storage_p))
    marg = dt * (float(np.dot(price, prod_after_curt + storage_p))
                 - float(np.dot(price, wind_no_bat)))
    return {"arbitrage": arb, "marginal": marg}


def build_lp_problem(prices: np.ndarray, config: Dict,
                     power_wind: np.ndarray | None = None) -> Tuple[Dict, Dict]:
    """Build SHIPP LP matrices → solve LP → return (lp_mats, lp_data).

    power_wind : wind-farm production [MW] for this slice, or None for the legacy battery-only arbitrage problem.

    Grid bounds are mode-dependent:
      battery-only (power_wind is None) : grid in [-P_CAP, P_CAP] (legacy; charges from the grid - the only source when there is no wind).
      wind (power_wind given)           : grid in [p_min_grid, p_max_grid] with p_min_grid = 0, so the battery charges ONLY from wind surplus and
                                          export is capped at the grid connection (matches Plan B / v5.4).
    """
    E_CAP   = config["e_cap"]
    P_CAP   = config["p_cap"]
    SOC_MIN = config["soc_min"]
    SOC_MAX = config["soc_max"]
    dt      = config["dt"]
    T       = len(prices)

    stor = Storage(
        e_cap=E_CAP, p_cap=P_CAP,
        eff_in=config["eff_in"], eff_out=config["eff_out"],
        e_cost=0, p_cost=0, dod=1.0,
    )
    stor_null = Storage(e_cap=0, p_cap=0, eff_in=1, eff_out=1,
                        e_cost=0, p_cost=0)

    if power_wind is None:
        power      = np.zeros(T)
        grid_p_min = -P_CAP
        grid_p_max =  P_CAP
    else:
        power      = np.asarray(power_wind, dtype=float)[:T]
        grid_p_min = float(config.get("p_min_grid", 0.0))
        grid_p_max = float(config["p_max_grid"])

    options = dict(formulation='lp_alt', fixed_cap=True)

    vec_obj = build_lp_obj_revenues(prices, T, options)
    mat_eq, vec_eq, mat_ineq, vec_ineq, bounds_lower, bounds_upper = \
        build_lp_cst_sparse(
            power, dt, grid_p_min, grid_p_max, T,
            stor, stor_null,
            stor1_p_cap_max=P_CAP, stor2_p_cap_max=0,
            stor1_e_cap_max=E_CAP, stor2_e_cap_max=0,
            options=options,
        )

    # Tighten energy bounds to [SoC_min, SoC_max] × E_cap
    e1_slice = slice(3*T, 4*T + 1)
    bounds_lower[e1_slice] = SOC_MIN * E_CAP
    bounds_upper[e1_slice] = SOC_MAX * E_CAP
    bounds_list = list(zip(bounds_lower, bounds_upper))

    print(f"  Solving LP (HiGHS)...", end=" ", flush=True)
    t0 = time.perf_counter()
    res_lp = linprog(
        vec_obj,
        A_ub=mat_ineq.toarray(), b_ub=vec_ineq,
        A_eq=mat_eq.toarray(),   b_eq=vec_eq,
        bounds=bounds_list, method='highs',
    )
    t_lp = time.perf_counter() - t0
    if not res_lp.success:
        raise RuntimeError(f"LP failed: {res_lp.message}")
    print(f"{t_lp:.1f} s")

    x_lp = res_lp.x
    c_slice = slice(2*T, 3*T)                       # p_curtailed (lp_alt layout)
    rev_lp  = daily_revenue(prices[:T], x_lp[0:T], x_lp[c_slice],
                            power, float(config["p_max_grid"]), dt)

    lp_mats = dict(
        vec_obj=vec_obj, mat_eq=mat_eq, vec_eq=vec_eq,
        mat_ineq=mat_ineq, vec_ineq=vec_ineq,
        bounds_list=bounds_list, e1_slice=e1_slice, c_slice=c_slice,
        T=T, x_lp=x_lp, t_lp=t_lp,
    )
    lp_data = dict(
        p=x_lp[0:T], e=x_lp[e1_slice], curtailed=x_lp[c_slice], wind=power,
        revenue=rev_lp["marginal"], revenue_arbitrage=rev_lp["arbitrage"],
    )
    return lp_mats, lp_data


# ══════════════════════════════════════════════════════════════════════════
# 4.  NLP RUNNER  (core engine — horizon-agnostic)
# ══════════════════════════════════════════════════════════════════════════

def run_single_period(
    prices: np.ndarray,
    config: Dict,
    shi_fit: ShiPolynomialFit,
    max_iter: int,
    results_dir: Path,
    prefix: str,
    year: int,
    period_label: str,
    solver: str = "trust-constr",
    tr_radius: float | None = None,
    use_scaling: bool = True,
    hess_method: str | None = None,
    alpha: float | None = None,
    power_wind: np.ndarray | None = None,
    verbose: bool = True,
    make_plots: bool = True,
) -> Dict:
    
    #Full LP → NLP pipeline for one price slice. This function is horizon-agnostic: it works identically whether "prices" contains 24, 168, 720, or 8760 hours.

    E_CAP = config["e_cap"]
    B     = config["replacement_cost"]
    dt    = config["dt"]
    T     = len(prices)
    w_deg = B * E_CAP

    print(f"\n  Period: {period_label}  ({T} hours)")
    print(f"  Price: mean={prices.mean():.1f}  std={prices.std():.1f}  "
          f"range=[{prices.min():.1f}, {prices.max():.1f}]")

    # ── LP ────────────────────────────────────────────────────────────────
    lp_mats, lp_data = build_lp_problem(prices, config, power_wind=power_wind)
    print(f"  SHIPP layout: {len(lp_mats['vec_obj'])} vars  (5×{T}+6 = {5*T+6})")

    f_lp, cyc_lp = compute_f_deg(lp_data["e"], E_CAP, shi_fit)
    deg_cost_lp = w_deg * f_lp
    lp_data["deg_cost"]  = deg_cost_lp
    lp_data["n_cycles"]  = len(cyc_lp)

    print(f"  LP Revenue:  {lp_data['revenue']:>12,.2f}   "
          f"Deg cost: {deg_cost_lp:>10,.2f}   "
          f"Net: {lp_data['revenue'] - deg_cost_lp:>12,.2f}   "
          f"Cycles: {len(cyc_lp)}")

    # ── Scaling factor ────────────────────────────────────────────────────
    obj_scale = max(abs(lp_data["revenue"]), 1.0) if (use_scaling and alpha is None) else 1.0
    if alpha is not None:
        _norm_rev = max(abs(float(np.dot(lp_mats["vec_obj"], lp_mats["x_lp"]))), 1e-12)
        _norm_deg = max(f_lp, 1e-12)
        print(f"  Alpha normalisation: alpha={alpha}  "
              f"norm_rev={_norm_rev:,.0f}  norm_deg={_norm_deg:.4e}")
        print(f"  At x0: revenue_term={(1-alpha)*1.0:.2f}  "
              f"degradation_term={alpha*1.0:.2f}  (both O(1), balanced)")
    elif use_scaling:
        print(f"  Objective scaling: 1 / {obj_scale:,.0f}  "
              f"(scaled obj ≈ {(lp_data['revenue'] - deg_cost_lp)/obj_scale:.4f})")

    # ── NLP ───────────────────────────────────────────────────────────────
    x0 = lp_mats["x_lp"].copy()  # must be defined before make_nlp_functions

    obj_fn, grad_fn, state = make_nlp_functions(
        lp_mats["vec_obj"], T, E_CAP, shi_fit, w_deg,
        x0=x0,
        obj_scale=obj_scale,
        alpha=alpha,
    )

    # Build human-readable solver tag
    solver_label = solver
    if alpha is not None:
        solver_label += f" +alpha{alpha}"
    elif use_scaling:
        solver_label += " +scaled"
    if hess_method:
        solver_label += f" hess={hess_method}"
    if tr_radius:
        solver_label += f" tr={tr_radius}"

    print(f"  Solving NLP ({solver_label}, max_iter={max_iter})...", flush=True)
    t0 = time.perf_counter()

    _iter    = {"n": 0}
    _history = []

    if solver == "trust-constr":
        constraints = [
            LinearConstraint(lp_mats["mat_ineq"], ub=lp_mats["vec_ineq"]),
            LinearConstraint(lp_mats["mat_eq"],
                             lb=lp_mats["vec_eq"], ub=lp_mats["vec_eq"]),
        ]

        def callback_tc(xk, opt_state):
            _iter["n"] += 1
            obj_sc  = float(getattr(opt_state, 'fun', float('nan')))
            cv      = float(getattr(opt_state, 'constr_violation', float('nan')))
            opt_val = float(getattr(opt_state, 'optimality', float('nan')))
            # obj: always store as alpha-weighted value or w_deg-scaled value for the convergence plot; cross-run EUR comparison uses summary fields
            _history.append(dict(
                iter=_iter["n"],
                obj=obj_sc * obj_scale,
                obj_scaled=obj_sc,
                cv=cv,
                optimality=opt_val,
                f_deg=float(state['last_f_deg']),
            ))
            if verbose and _iter["n"] % 10 == 0:
                print(f"    iter {_iter['n']:4d}  obj_sc={obj_sc:+.6f}  "
                      f"cv={cv:.2e}  opt={opt_val:.2e}  "
                      f"f_deg={state['last_f_deg']:.4e}")
            return False

        tc_opts = dict(
            sparse_jacobian=True,
            verbose=2 if verbose else 0,
            maxiter=max_iter,
            xtol=1e-12,      # prevent early xtol termination; solver stops at gtol or maxiter
            initial_tr_radius=tr_radius if tr_radius is not None else 0.1,
            # 0.1 = Jenna's conservative default; avoids the large infeasible excursion in early iterations that wastes budget on recovery
        )

        kw = dict(
            fun=obj_fn, x0=x0, jac=grad_fn,
            constraints=constraints,
            bounds=lp_mats["bounds_list"],
            method='trust-constr',
            tol=1e-6, options=tc_opts,
            callback=callback_tc,
        )
        if hess_method:
            kw["hess"] = hess_method

        res_nlp = minimize(**kw)

    elif solver == "slsqp":
        A_eq_d = lp_mats["mat_eq"].toarray()
        A_ub_d = lp_mats["mat_ineq"].toarray()
        constraints_sq = [
            {"type": "eq",
             "fun": lambda x: A_eq_d @ x - lp_mats["vec_eq"],
             "jac": lambda x: A_eq_d},
            {"type": "ineq",
             "fun": lambda x: lp_mats["vec_ineq"] - A_ub_d @ x,
             "jac": lambda x: -A_ub_d},
        ]

        def callback_sq(xk):
            _iter["n"] += 1
            obj_sc = float(obj_fn(xk))
            _history.append(dict(
                iter=_iter["n"],
                obj=obj_sc * obj_scale,
                obj_scaled=obj_sc,
                cv=0.0, optimality=float('nan'),
                f_deg=float(state['last_f_deg']),
            ))
            if verbose and _iter["n"] % 25 == 0:
                print(f"    iter {_iter['n']:4d}  obj_sc={obj_sc:+.6f}  "
                      f"f_deg={state['last_f_deg']:.4e}")

        res_nlp = minimize(
            obj_fn, x0, jac=grad_fn,
            constraints=constraints_sq,
            bounds=lp_mats["bounds_list"], method='SLSQP',
            options=dict(maxiter=max_iter, disp=verbose, ftol=1e-9),
            callback=callback_sq,
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    t_nlp = time.perf_counter() - t0

    # ── Results ───────────────────────────────────────────────────────────
    x_nlp = res_nlp.x
    nlp_p = x_nlp[0:T]
    nlp_e = x_nlp[lp_mats["e1_slice"]]
    nlp_c = x_nlp[lp_mats["c_slice"]]
    rev_nlp_d   = daily_revenue(prices[:T], nlp_p, nlp_c, lp_data["wind"],
                                float(config["p_max_grid"]), dt)
    nlp_revenue = rev_nlp_d["marginal"]

    f_nlp, cyc_nlp = compute_f_deg(nlp_e, E_CAP, shi_fit)
    deg_cost_nlp = w_deg * f_nlp

    net_lp  = lp_data["revenue"] - deg_cost_lp
    net_nlp = nlp_revenue - deg_cost_nlp

    final_opt  = _history[-1]["optimality"] if _history else float('nan')
    converged  = final_opt < 1e-6

    sac = lp_data["revenue"] - nlp_revenue
    sav = deg_cost_lp - deg_cost_nlp

    print(f"\n  NLP status: {res_nlp.message}")
    print(f"  NLP time:   {t_nlp:.1f} s ({t_nlp/60:.1f} min)  iters: {res_nlp.nit}")
    print(f"  Final optimality: {final_opt:.2e}  "
          f"{'** CONVERGED **' if converged else '!! NOT CONVERGED !!'}")

    print(f"\n  {'Metric':<22s} {'LP':>12s} {'NLP':>12s} {'Delta':>10s}")
    print(f"  {'---'*18}")
    print(f"  {'Revenue [EUR]':<22s} {lp_data['revenue']:>12,.0f} "
          f"{nlp_revenue:>12,.0f} {nlp_revenue-lp_data['revenue']:>+10,.0f}")
    print(f"  {'Deg cost [EUR]':<22s} {deg_cost_lp:>12,.0f} "
          f"{deg_cost_nlp:>12,.0f} {deg_cost_nlp-deg_cost_lp:>+10,.0f}")
    print(f"  {'Net utility [EUR]':<22s} {net_lp:>12,.0f} "
          f"{net_nlp:>12,.0f} {net_nlp-net_lp:>+10,.0f}")
    print(f"  {'Cycles':<22s} {len(cyc_lp):>12d} {len(cyc_nlp):>12d}")
    if sac > 0:
        print(f"  Sacrifice: {sac:>+,.0f}  Saving: {sav:>+,.0f}  "
              f"Ratio: {sav/sac:.2f}x")

    nlp_result = dict(
        p=nlp_p, e=nlp_e, curtailed=nlp_c, wind=lp_data["wind"],
        revenue=nlp_revenue, revenue_arbitrage=rev_nlp_d["arbitrage"],
        f_deg=f_nlp,
        deg_cost_EUR=deg_cost_nlp, n_cycles=len(cyc_nlp),
        status_msg=res_nlp.message, n_iter=res_nlp.nit,
        history=_history,
    )

    _save_results(
        results_dir, prefix, year, period_label,
        lp_data, nlp_result, prices, config,
        {"LP": lp_mats["t_lp"], "NLP": t_nlp},
        solver_info=solver_label, obj_scale=obj_scale, alpha=alpha,
        make_plots=make_plots,
    )

    return nlp_result


# ══════════════════════════════════════════════════════════════════════════
# 5.  SAVE + PLOT
# ══════════════════════════════════════════════════════════════════════════

def _save_results(
    results_dir, prefix, year, period_label,
    lp_data, nlp_result, prices, config, timings,
    solver_info="", obj_scale=1.0, alpha=None, make_plots=True,
):
    results_dir.mkdir(exist_ok=True)
    T = len(prices)

    rev_lp,  rev_nlp  = lp_data["revenue"], nlp_result["revenue"]
    deg_lp,  deg_nlp  = lp_data["deg_cost"], nlp_result["deg_cost_EUR"]
    net_lp  = rev_lp  - deg_lp
    net_nlp = rev_nlp - deg_nlp
    sac = rev_lp - rev_nlp
    sav = deg_lp - deg_nlp

    history = nlp_result.get("history", [])
    final_opt = history[-1]["optimality"] if history else float('nan')

    # ── 1. Arrays (.npz) ─────────────────────────────────────────────────
    npz_path = results_dir / f"{prefix}_results.npz"
    np.savez_compressed(
        npz_path,
        lp_e=lp_data["e"], lp_p=lp_data["p"],
        lp_curtailed=lp_data.get("curtailed", np.zeros(T)),
        nlp_e=nlp_result["e"], nlp_p=nlp_result["p"],
        nlp_curtailed=nlp_result.get("curtailed", np.zeros(T)),
        wind=lp_data.get("wind", np.zeros(T)),
        prices=prices,
    )
    print(f"  Saved: {npz_path.name}")

    # ── 2. Convergence JSON ───────────────────────────────────────────────
    summary = dict(
        runner=_RUNNER_NAME, solver=solver_info,
        year=year, period=period_label, hours=T,
        e_cap=config["e_cap"], p_cap=config["p_cap"],
        soc_min=config["soc_min"], soc_max=config["soc_max"],
        replacement_cost=config["replacement_cost"],
        obj_scale=obj_scale,
        alpha=alpha,
        lp_revenue=round(rev_lp, 2),
        nlp_revenue=round(rev_nlp, 2),
        lp_revenue_arbitrage=round(lp_data.get("revenue_arbitrage", rev_lp), 2),
        nlp_revenue_arbitrage=round(nlp_result.get("revenue_arbitrage", rev_nlp), 2),
        wind_mode=bool(config.get("wind_mode", False)),
        lp_deg_cost=round(deg_lp, 2),
        nlp_deg_cost=round(deg_nlp, 2),
        lp_net=round(net_lp, 2),
        nlp_net=round(net_nlp, 2),
        revenue_sacrifice=round(sac, 2),
        degradation_saving=round(sav, 2),
        ratio=round(sav / sac, 2) if sac > 0 else 0.0,
        nlp_status=nlp_result.get("status_msg", "N/A"),
        nlp_iters=nlp_result.get("n_iter", 0),
        nlp_n_cycles=nlp_result.get("n_cycles", 0),
        lp_n_cycles=lp_data.get("n_cycles", 0),
        final_optimality=float(final_opt),
        converged=bool(final_opt < 1e-6),
        timings={k: round(v, 2) for k, v in timings.items()},
    )

    json_data = dict(summary=summary, convergence=history)
    json_path = results_dir / f"{prefix}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(json_data, f, indent=1)
    print(f"  Saved: {json_path.name} ({len(history)} iterations)")

    # ── 3. Console summary ────────────────────────────────────────────────
    print(f"\n  {'Metric':<28s} {'LP':>12s} {'NLP':>12s} {'Delta':>10s}")
    print(f"  {'---'*18}")
    print(f"  {'Revenue [EUR]':<28s} {rev_lp:>12,.2f} {rev_nlp:>12,.2f} "
          f"{rev_nlp-rev_lp:>+10,.2f}")
    print(f"  {'Deg cost [EUR]':<28s} {deg_lp:>12,.2f} {deg_nlp:>12,.2f} "
          f"{deg_nlp-deg_lp:>+10,.2f}")
    print(f"  {'Net utility [EUR]':<28s} {net_lp:>12,.2f} {net_nlp:>12,.2f} "
          f"{net_nlp-net_lp:>+10,.2f}")
    if sac > 0:
        print(f"  Sacrifice: {sac:>+,.0f}  Saving: {sav:>+,.0f}  "
              f"Ratio: {sav/sac:.2f}x")

    # ── 4. Comparison plot (3 panels) ─────────────────────────────────────
    if not make_plots:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        fig.patch.set_facecolor("#f7f9fc")
        for ax in axes:
            ax.set_facecolor("#f7f9fc")
        fig.suptitle(
            f"Path 3 ({solver_info})  |  DK1 {year} {period_label}",
            fontsize=13, fontweight="bold",
        )
        hours = np.arange(T)

        axes[0].plot(hours, prices, color="#2166ac", lw=0.5, alpha=0.8)
        axes[0].set_ylabel("Price [EUR/MWh]")
        axes[0].set_title("Day-Ahead Price", fontweight="bold")

        axes[1].plot(hours, lp_data["e"][:T], color="#2166ac", lw=0.6,
                     label="LP", alpha=0.8)
        axes[1].plot(hours, nlp_result["e"][:T], color="#b5351b", lw=0.6,
                     label="NLP", alpha=0.8)
        axes[1].set_ylabel("Stored Energy [MWh]")
        axes[1].set_title("Battery SoC", fontweight="bold")
        axes[1].legend(fontsize=9)

        diff = nlp_result["p"] - lp_data["p"]
        axes[2].fill_between(hours, diff, 0, where=diff > 0,
                             color="#2166ac", alpha=0.3, label="NLP > LP")
        axes[2].fill_between(hours, diff, 0, where=diff < 0,
                             color="#b5351b", alpha=0.3, label="NLP < LP")
        axes[2].set_ylabel("Delta Power [MW]")
        axes[2].set_xlabel("Hour")
        axes[2].set_title("Dispatch Difference", fontweight="bold")
        axes[2].legend(fontsize=9)

        plt.tight_layout()
        png = results_dir / f"{prefix}_comparison.png"
        fig.savefig(png, dpi=150, bbox_inches="tight",
                    facecolor="#f7f9fc", edgecolor="none")
        plt.close(fig)
        print(f"  Saved: {png.name}")

        # ── 5. Convergence plot (4 panels) ────────────────────────────────
        if len(history) > 2:
            fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
            fig2.patch.set_facecolor("#f7f9fc")
            for ax in axes2.flat:
                ax.set_facecolor("#f7f9fc")
            scale_tag = f"scale={obj_scale:.0f}" if obj_scale > 1 else "unscaled"
            fig2.suptitle(
                f"Convergence  |  DK1 {year} {period_label}  |  "
                f"{solver_info}  |  {scale_tag}",
                fontsize=13, fontweight="bold",
            )

            iters = [h["iter"] for h in history]
            objs  = [h["obj"] for h in history]
            cvs   = [h["cv"]  for h in history]
            opts  = [h["optimality"] for h in history]
            fdegs = [h["f_deg"] for h in history]

            axes2[0,0].plot(iters, objs, color="#2166ac", lw=0.8)
            axes2[0,0].set_ylabel("Objective (unscaled)")
            axes2[0,0].set_title("Objective Value", fontweight="bold")
            axes2[0,0].ticklabel_format(axis='y', style='scientific',
                                        scilimits=(-3,3))

            if len(objs) > 1:
                dobj = np.abs(np.diff(objs))
                dobj_safe = np.where(dobj > 0, dobj, 1e-20)
                axes2[0,1].semilogy(iters[1:], dobj_safe,
                                    color="#2166ac", lw=0.6, alpha=0.7)
                axes2[0,1].set_ylabel("|ΔObj|")
                axes2[0,1].set_title("Objective Change per Iter",
                                     fontweight="bold")

            opts_clean = [o for o in opts if o == o]  # drop NaN
            if opts_clean:
                axes2[1,0].semilogy(iters[:len(opts_clean)], opts_clean,
                                    color="#b5351b", lw=0.8)
                axes2[1,0].axhline(1e-6, ls='--', color='gray', lw=0.7,
                                   label='tol=1e-6')
                axes2[1,0].set_ylabel("Optimality")
                axes2[1,0].set_title("KKT Optimality", fontweight="bold")
                axes2[1,0].legend(fontsize=8)

            axes2[1,1].plot(iters, fdegs, color="#2166ac", lw=0.8)
            axes2[1,1].set_ylabel("f_deg")
            axes2[1,1].set_title("Degradation Fraction", fontweight="bold")
            axes2[1,1].ticklabel_format(axis='y', style='scientific',
                                        scilimits=(-3,3))

            for ax in axes2[1,:]:
                ax.set_xlabel("Iteration")

            plt.tight_layout()
            conv_png = results_dir / f"{prefix}_convergence.png"
            fig2.savefig(conv_png, dpi=150, bbox_inches="tight",
                         facecolor="#f7f9fc", edgecolor="none")
            plt.close(fig2)
            print(f"  Saved: {conv_png.name}")

    except Exception as exc:
        print(f"  Plot failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# 6.  TIME-RANGE DEFINITIONS  (days, weeks, multi-week, months)
# ══════════════════════════════════════════════════════════════════════════

_DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_MONTH_NAMES = [
    "Jan","Feb","Mar","Apr","May","Jun",
    "Jul","Aug","Sep","Oct","Nov","Dec",
]


def month_hour_range(month: int) -> Tuple[int, int]:
    """Return (h_start, h_end) for a 1-indexed month."""
    start = sum(_DAYS_PER_MONTH[:month-1]) * 24
    end   = start + _DAYS_PER_MONTH[month-1] * 24
    return start, end


# Named slots: (start_day_0indexed, n_days, label)
# Add new entries freely — the CLI picks them up automatically.
SLOTS = {
    # ── Single-day slots (24 h) ──────────────────────────────────────────
    # Original reference days (week-slot anchors)
    "D1":   (0,   1, "Jan 1 (2019-High, day 1 of W1)"),
    "D2":   (1,   1, "Jan 2 (2019-High, day 2 of W1)"),
    "D43":  (42,  1, "Feb 12 (2022-Low, day 1 of W7)"),
    "D49":  (48,  1, "Feb 18 (2022-Low, day 7 of W7)"),
    "D85":  (84,  1, "Mar 26 (2022-Mod, day 1 of W13)"),
    "D190": (189, 1, "Jul 9 (2019-Low, day 1 of W28)"),
    "D232": (231, 1, "Aug 20 (2022-High, day 1 of W34)"),
    "D239": (238, 1, "Aug 27 (2019-Mod, day 1 of W35)"),
    # 2022 convergence study days — HIGH / MID / LOW
    "D8":   (  7, 1, "Jan 8 (2022-High, winter spike)"),       # h168-192
    "D238": (237, 1, "Aug 26 (2022-High, crisis peak)"),       # h5688-5712
    "D248": (247, 1, "Sep 5 (2022-High, crisis tail)"),        # h5928-5952
    "D86":  ( 85, 1, "Mar 27 (2022-Mid, spring)"),             # h2040-2064
    "D166": (165, 1, "Jun 15 (2022-Mid, early summer)"),       # h3960-3984
    "D309": (308, 1, "Nov 5 (2022-Mid, autumn)"),              # h7392-7416
    "D128": (127, 1, "May 8 (2022-Low, spring renewables)"),   # h3048-3072
    "D197": (196, 1, "Jul 16 (2022-Low, mid summer)"),         # h4704-4728
    # ── Week slots (168 h) ───────────────────────────────────────────────
    "W1":   (0,   7, "Jan 1-7 (2019-High)"),
    "W7":   (42,  7, "Feb 12-18 (2022-Low)"),
    "W13":  (84,  7, "Mar 26-Apr 1 (2022-Mod)"),
    "W28":  (189, 7, "Jul 9-15 (2019-Low)"),
    "W34":  (231, 7, "Aug 20-26 (2022-High)"),
    "W35":  (238, 7, "Aug 27-Sep 2 (2019-Mod)"),
    # ── Multi-week slots (336 h) ─────────────────────────────────────────
    "W1W2":   (0,   14, "Jan 1-14 (2019-High)"),
    "W7W8":   (42,  14, "Feb 12-25 (2022-Low)"),
    "W28W29": (189, 14, "Jul 9-22 (2019-Low)"),
    "W34W35": (231, 14, "Aug 20-Sep 2 (2022-High)"),
}

# Batch groups
_BATCH_GROUPS = {
    "all-days":        [k for k in SLOTS if k.startswith("D")],
    "all-weeks":       [k for k in SLOTS if k.startswith("W") and len(k) <= 3],
    "all-2weeks":      [k for k in SLOTS if k.startswith("W") and len(k) > 3],
    # 2022 convergence study — 9 days spanning HIGH / MID / LOW regimes
    "2022-study":      ["D8","D238","D248",          # HIGH
                        "D86","D166","D309",          # MID
                        "D43","D128","D197"],         # LOW (D43 = reference, already run)
    "2022-study-new":  ["D8","D238","D248",           # HIGH
                        "D86","D166","D309",          # MID
                        "D128","D197"],               # LOW (skip D43 if already run)
}


def slot_hour_range(slot_key: str) -> Tuple[int, int, str]:
    """Return (h_start, h_end, label) for any named slot."""
    day0, n_days, label = SLOTS[slot_key]
    return day0 * 24, (day0 + n_days) * 24, label


# ══════════════════════════════════════════════════════════════════════════
# 7.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_dk1_prices(year: int) -> np.ndarray:
    """Load 8760 hourly DK1 prices for "year" from CSV."""
    csv_by_year = {2019: PRICE_CSV_2019, 2022: PRICE_CSV_2022}
    if year not in csv_by_year:
        raise ValueError(
            f"No price data for {year}. Available: {sorted(csv_by_year)}"
        )
    csv_path = csv_by_year[year]
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    import pandas as pd
    df = pd.read_csv(csv_path)
    for col in ["price", "price [EUR/MWh]", "Price", "EUR/MWh", "DK1"]:
        if col in df.columns:
            prices = df[col].values.astype(float)
            break
    else:
        prices = df.select_dtypes(include=[np.number]).iloc[:, -1].values

    prices = prices[:8760]
    if len(prices) < 8760:
        raise ValueError(f"Need 8760 hours, got {len(prices)}")

    nan_count = np.isnan(prices).sum()
    if nan_count > 0:
        print(f"  Warning: {nan_count} NaN prices replaced with mean")
        prices[np.isnan(prices)] = np.nanmean(prices)

    return prices


def compute_wind_power(setup: dict, n_hours: int = 8760) -> np.ndarray:
    """Full-year wind-farm power [MW] via the same PyWake pipeline as Plan B /
    v5.4 (Bastankhah + Crespo-Hernandez, ERA5 resource from the HPP YAML)."""
    import xarray as xr
    from py_wake.site import XRSite

    ts = setup["hpp"]["site"]["energy_resource"]["time_series"]["wind_resource"]
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

    site = XRSite(ds=xr.Dataset(data_vars=dict(P=1)))
    wf_model = get_wake_model("Bastankhah", site, setup["windturbine"])
    kwargs = {"x": setup["x"], "y": setup["y"],
              "wd": wd, "ws": ws, "time": np.arange(n) / 24.0}
    if ti is not None:
        kwargs["TI"] = ti
    sim_res = wf_model(**kwargs)
    return sim_res.Power.sum(["wt"]).values / 1e6


# ══════════════════════════════════════════════════════════════════════════
# 8.  CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    slot_choices = list(SLOTS.keys()) + list(_BATCH_GROUPS.keys())

    parser = argparse.ArgumentParser(
        description="Path 3 — Degradation-Aware NLP Dispatch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --year 2022 --slot D43                # single day, scaled
  %(prog)s --year 2019 --slot W1 --no-scale      # week, unscaled comparison
  %(prog)s --year 2022 --slot W7 --hess 2-point  # week, FD Hessian
  %(prog)s --year 2022 --month 7                 # July
  %(prog)s --year 2019 --month full              # full year
  %(prog)s --year 2022 --start-hour 1008 --n-hours 72  # arbitrary 3-day
  %(prog)s --year 2022 --slot all-days           # batch all day slots
""",
    )

    # ── Time range (mutually exclusive groups are too rigid; use priority) ─
    parser.add_argument("--year", type=int, default=2022,
                        choices=[2019, 2022])
    parser.add_argument("--slot", type=str, default=None,
                        choices=slot_choices,
                        help="Named slot or batch group")
    parser.add_argument("--month", type=str, default=None,
                        help="Month (1-12), 'all', or 'full'")
    parser.add_argument("--start-hour", type=int, default=None,
                        help="Start hour (0-indexed) for arbitrary slice")
    parser.add_argument("--n-hours", type=int, default=24,
                        help="Number of hours for --start-hour (default: 24)")

    # ── Solver options ────────────────────────────────────────────────────
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--solver", type=str, default="trust-constr",
                        choices=["trust-constr", "slsqp"])
    parser.add_argument("--tr-radius", type=float, default=None,
                        help="Initial trust-region radius")
    parser.add_argument("--hess", type=str, default=None,
                        choices=["2-point", "3-point"],
                        help="FD Hessian (trust-constr only)")

    # ── Scaling (ON by default) ───────────────────────────────────────────
    parser.add_argument("--no-scale", action="store_true",
                        help="Disable objective scaling (for comparison)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Enable Jenna-style dual normalisation with this "
                             "alpha weight on degradation (e.g. 0.2). "
                             "When set, --no-scale is ignored: both terms are "
                             "normalised separately at the LP solution. "
                             "Suggested starting value: 0.2")

    # ── Misc ──────────────────────────────────────────────────────────────
    parser.add_argument("--e-cap", type=float, default=None,
                        help="Energy capacity [MWh] override (default: YAML). "
                             "Use 550 for the degraded-optimum comparison.")
    parser.add_argument("--p-cap", type=float, default=None,
                        help="Power capacity [MW] override (default: YAML). "
                             "Use 175 for the degraded-optimum comparison.")
    parser.add_argument("--wind", action="store_true",
                        help="HPP mode: wind production + curtailment, P_MIN=0 "
                             "(battery charges only from wind surplus, matches "
                             "Plan B / v5.4). Default OFF = battery-only arbitrage, "
                             "which reproduces the existing converged runs.")

    parser.add_argument("--test", action="store_true",
                        help="Synthetic 24h test")
    parser.add_argument("--all-days", action="store_true",
                        help="Run every day of the year (plots OFF, quiet) into "
                             "results/nlp_monolithic_all_days/run_<ts>/. "
                             "For the upside distribution.")
    args = parser.parse_args()

    use_scaling = not args.no_scale

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = make_results_dir("nlp_monolithic")

    # Build file-name-safe solver tag
    solver_tag = args.solver
    if args.alpha is not None:
        solver_tag += f"_alpha{args.alpha}"
    elif use_scaling:
        solver_tag += "_scaled"
    if args.hess:
        solver_tag += f"_hess{args.hess.replace('-','')}"
    if args.tr_radius:
        solver_tag += f"_tr{args.tr_radius}"

    print("=" * 72)
    print(f"Path 3 NLP  |  DK1 {args.year}")
    print(f"Solver:  {solver_tag}   max_iter: {args.max_iter}")
    if args.alpha is not None:
        print(f"Normalisation: ALPHA mode (alpha={args.alpha}, dual-norm at LP)   "
              f"Hessian: {args.hess or 'BFGS (default)'}")
    else:
        print(f"Scaling: {'ON (obj / |LP_rev|)' if use_scaling else 'OFF'}   "
              f"Hessian: {args.hess or 'BFGS (default)'}")
    print(f"Timestamp: {timestamp}")
    print("=" * 72)

    # ── Battery configuration ─────────────────────────────────────────────
    setup = quick_setup(HPP_YAML, config={"interp_n": 2000}, verbose=False)
    bat   = setup["battery"]

    rte_ac = float(bat["rte_nominal"]) * float(bat["pcu_efficiency"]) ** 2   # 0.9025 * 0.986^2 = 0.877
    eta    = eta_symmetric(rte_ac)                                           # sqrt(0.877) = 0.9367  (was 0.95/0.95, RTE 0.9025, no PCU)
    print(f"  RTE_ac={rte_ac:.4f}  eta={eta:.4f}   (expect 0.9100 / 0.9539)")

    # (E,P) default to YAML; override at the degraded optimum (e.g. --e-cap 550 --p-cap 175)
    e_cap = args.e_cap if args.e_cap is not None else bat["energy_capacity_Wh"] / 1e6
    p_cap = args.p_cap if args.p_cap is not None else bat["power_capacity_W"]   / 1e6

    config = dict(
        e_cap=e_cap, p_cap=p_cap,
        soc_min=float(bat["soc_min"]), soc_max=float(bat["soc_max"]),
        eff_in=eta, eff_out=eta,
        dt=1.0,
        replacement_cost=float(bat["repl_energy_EUR_per_kWh"]) * 1000.0,     # 72_000 EUR/MWh  (was hardcoded 150_000)
        p_max_grid=float(setup["hpp"]["grid_connection_capacity"]) / 1e6,    # grid connection cap [MW]; export limit in wind mode
        p_min_grid=0.0,                                                      # no grid import in wind mode (matches Plan B / v5.4)
        wind_mode=bool(args.wind),
    )
    print(f"\n  Battery: {config['e_cap']} MWh / {config['p_cap']} MW")
    print(f"  SoC: [{config['soc_min']}, {config['soc_max']}]")
    print(f"  w_deg = {config['replacement_cost']*config['e_cap']:,.0f} EUR")

    shi_fit = fit_shi_polynomial(
        soc_min=config["soc_min"], soc_max=config["soc_max"],
        source=f"Path3 {args.year}", verbose=True,
    )

    # ── Dispatch helper ───────────────────────────────────────────────────
    wind_full = compute_wind_power(setup) if args.wind else None
    if args.wind:
        print(f"  Wind: HPP mode ON  (full-year mean {wind_full.mean():.1f} MW, "
              f"max {wind_full.max():.1f} MW)  grid cap {config['p_max_grid']:.0f} MW")
    else:
        print("  Wind: OFF (battery-only price arbitrage)")

    def _wind_slice(h0, h1):
        return None if wind_full is None else wind_full[h0:h1]

    def _run(prices_slice, wind_slice, label, pfx,
             make_plots=True, out_dir=None, verbose=True):
        run_single_period(
            prices_slice, config, shi_fit, args.max_iter,
            out_dir or results_dir, pfx, args.year, label,
            solver=args.solver, tr_radius=args.tr_radius,
            use_scaling=use_scaling, hess_method=args.hess,
            alpha=args.alpha, power_wind=wind_slice,
            make_plots=make_plots, verbose=verbose,
        )

    # ── Synthetic test ────────────────────────────────────────────────────
    if args.test:
        T = 24
        rng = np.random.RandomState(42)
        hours = np.arange(T) % 24
        prices = 30 + 20*np.sin(2*np.pi*(hours-6)/24) + rng.normal(0, 5, T)
        prices = np.maximum(prices, 0)
        _run(prices, None, f"Test ({T}h)",
             f"{timestamp}_{_RUNNER_NAME}_test_{solver_tag}")
        return

    # ── Load real prices ──────────────────────────────────────────────────
    prices_full = load_dk1_prices(args.year)
    print(f"  Loaded {len(prices_full)} hours of DK1 {args.year}")

    t_total_start = time.perf_counter()

    # ── Route: --all-days (full year, plots OFF, quiet; upside distribution) ──
    if args.all_days:
        days_dir = make_results_dir(f"nlp_monolithic_all_days/run_{timestamp}")
        n_days = len(prices_full) // 24
        print(f"\n  ALL-DAYS mode: {n_days} days -> {days_dir}  (plots OFF, quiet)")
        ok, failed = 0, []
        for day in range(n_days):
            h0, h1 = day * 24, (day + 1) * 24
            label = f"D{day+1} (h{h0}-{h1})"
            pfx = f"{timestamp}_{_RUNNER_NAME}_dk{args.year}_D{day+1:03d}_{solver_tag}"
            try:
                _run(prices_full[h0:h1], _wind_slice(h0, h1), label, pfx,
                     make_plots=False, out_dir=days_dir, verbose=False)
                ok += 1
            except Exception as e:
                failed.append(day + 1)
                print(f"  [day {day+1:3d}] FAILED: {type(e).__name__}: {e}")
            if (day + 1) % 25 == 0:
                el = time.perf_counter() - t_total_start
                print(f"  ... {day+1}/{n_days} done ({ok} ok, {len(failed)} failed)  "
                      f"[{el/60:.1f} min]")
        print(f"\n  ALL-DAYS complete: {ok}/{n_days} ok, {len(failed)} failed")
        if failed:
            print(f"  Failed days: {failed}")
        print(f'  Aggregate with: python aggregate_path3.py --dir "{days_dir}"')
        return

    # ── Route: --slot (named slots + batch groups) ────────────────────────
    if args.slot is not None:
        if args.slot in _BATCH_GROUPS:
            slot_keys = _BATCH_GROUPS[args.slot]
        else:
            slot_keys = [args.slot]

        for sk in slot_keys:
            h0, h1, slabel = slot_hour_range(sk)
            label = f"{sk} {slabel} (h{h0}-{h1})"
            pfx = f"{timestamp}_{_RUNNER_NAME}_dk{args.year}_{sk}_{solver_tag}"
            print(f"\n{'---'*24}")
            print(f"  Slot {sk}: {slabel}  ({h1-h0} hours)")
            print(f"{'---'*24}")
            _run(prices_full[h0:h1], _wind_slice(h0, h1), label, pfx)

    # ── Route: --start-hour (arbitrary range) ─────────────────────────────
    elif args.start_hour is not None:
        h0 = args.start_hour
        h1 = min(h0 + args.n_hours, 8760)
        label = f"Custom h{h0}-{h1} ({h1-h0}h)"
        pfx = (f"{timestamp}_{_RUNNER_NAME}_dk{args.year}"
               f"_h{h0}_{h1-h0}h_{solver_tag}")
        _run(prices_full[h0:h1], _wind_slice(h0, h1), label, pfx)

    # ── Route: --month ────────────────────────────────────────────────────
    elif args.month is not None:
        if args.month == "full":
            _run(prices_full, _wind_slice(0, len(prices_full)), "Full Year",
                 f"{timestamp}_{_RUNNER_NAME}_dk{args.year}_full_{solver_tag}")

        elif args.month == "all":
            for m in range(1, 13):
                h0, h1 = month_hour_range(m)
                label = f"{_MONTH_NAMES[m-1]} (h{h0}-{h1})"
                pfx = (f"{timestamp}_{_RUNNER_NAME}_dk{args.year}"
                       f"_m{m:02d}_{solver_tag}")
                print(f"\n{'---'*24}")
                print(f"  Month {m}/12: {_MONTH_NAMES[m-1]}")
                print(f"{'---'*24}")
                _run(prices_full[h0:h1], _wind_slice(h0, h1), label, pfx)
        else:
            m = int(args.month)
            if not 1 <= m <= 12:
                raise ValueError(f"Month must be 1-12, got {m}")
            h0, h1 = month_hour_range(m)
            label = f"{_MONTH_NAMES[m-1]} (h{h0}-{h1})"
            pfx = (f"{timestamp}_{_RUNNER_NAME}_dk{args.year}"
                   f"_m{m:02d}_{solver_tag}")
            _run(prices_full[h0:h1], _wind_slice(h0, h1), label, pfx)

    else:
        parser.error("Specify --slot, --month, or --start-hour")

    t_total = time.perf_counter() - t_total_start
    print(f"\n{'==='*24}")
    print(f"Done.  Wall time: {t_total:.1f} s ({t_total/60:.1f} min)")
    print(f"{'==='*24}")


if __name__ == "__main__":
    main()