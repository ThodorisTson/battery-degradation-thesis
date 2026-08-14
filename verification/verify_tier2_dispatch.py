r"""
verify_tier2_dispatch.py  --  Tier 2 verification of the dispatch model

WHAT THIS IS
------------
Verification, not validation. Tier 1 (verify_tier1_degradation.py) tested the rainflow counter and the fade models on synthetic state-of-charge traces with
the optimization removed. This file tests the dispatch itself, and its coupling to degradation, on synthetic price signals whose optimal answer is known before
the solver runs.

No ERA5 wind, no DK1 prices. The wind series is a constant and the price series is analytic, so every quantity checked below has a closed form.

TESTS
-----
  T2.1  Dispatch pattern on a sinusoidal price, one week.
        Charges into the daily price minimum and discharges into the maximum, stays inside the SoC window, returns to its starting level, and the
        resulting SoC trace produces exactly seven deep cycles at delta = 0.80.
  T2.2  State-of-charge dynamics on the same solution.
        The kernel imposes the SoC balance as TWO INEQUALITIES,
            e[i+1] - e[i] <= -dt * eta_in  * p[i]
            e[i+1] - e[i] <= -dt / eta_out * p[i]
        so the model may discard energy where the objective is indifferent.
        The test asserts the inequalities hold everywhere and reports how many steps carry slack, which is a direct measurement of the phantom
        micro-cycle mechanism described in the methodology.
  T2.3  Round-trip efficiency threshold.
        With a two-level price of ratio r, diverting 1 MWh from the low block returns r * RTE MWh-equivalent in the high block, so arbitrage is
        profitable if and only if r > 1 / RTE. For this battery RTE = 0.910000056 and the threshold is r = 1.0989010. The test sweeps r
        across it and checks that the battery is idle below, saturates above, and that the observed transition brackets the predicted threshold.

SCOPE
-----
Wind is constant at 160 MW and never exceeds the 325 MW grid limit, so no case here curtails. The lifetime NPV reported in Chapter 4 uses the marginal revenue
basis, which includes curtailment recovery. At this site that term is about 52 kEUR, roughly 0.02% of NPV, because the grid connection equals the installed
wind rating. The curtailment path is therefore not exercised by these tests and carries about 0.02% of the reported value.

WHY IT CALLS THE RUN SCRIPT
---------------------------
The LP is invoked through _solve_shipp() imported from the production run script, so the code path exercised here is the one that produced Chapter 4.
Reimplementing the solver call would verify a copy instead. The run script is import-safe: everything heavy sits inside main().

Battery parameters are read from config/battery.yaml independently of the run script's own builder, so that the efficiency the LP honours is compared against
the efficiency the configuration declares.

OUTPUT
------
  console table of expected vs obtained
  Verification/tier2_verification.csv
  Verification/tier2_verification_table.tex
  fig_verify_dispatch_week.pdf / .png    (not a thesis figure)
  fig_verify_rte_threshold.pdf / .png    (Figure D.2)
  both written beside this script

Reproducible on Windows / VS Code. Needs pyomo + gurobi, as the SoC window is only enforced in the Pyomo branch of the kernel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from degradation.paths import BATTERY_YAML, require
from degradation.style import apply_thesis_style, figsize, TUDELFT, FS_ANNOT, FS_LEGEND
from degradation.xu import rainflow_cycle_counting

HERE = Path(__file__).resolve().parent

# -- Output ------------------------------------------------------------------ #
OUTPUT = "both"     # "png", "pdf" or "both"
DPI = 300

OUT_DIR = HERE
BATTERY_YAML = require(BATTERY_YAML)

# =============================================================================
# 1.  Case configuration
# =============================================================================
DT_H      = 1.0
P_MAX_MW  = 325.0     # grid export limit; leaves headroom for full discharge
WIND_MW   = 160.0     # constant wind, above P_CAP so charging is not wind-limited

# T2.1 sinusoidal price
N_DAYS_SIN   = 7
N_DAYS_PLOT  = 3      # days shown in the dispatch figure; the solve uses N_DAYS_SIN
PRICE_MEAN   = 100.0  # EUR/MWh
PRICE_AMP    = 60.0   # EUR/MWh, so the daily ratio is 160/40 = 4.0

# T2.3 two-level price
N_DAYS_SQ    = 3
PRICE_LOW    = 50.0   # EUR/MWh in the first 12 hours of each day
RATIO_SWEEP     = (1.02, 1.05, 1.08, 1.09, 1.095, 1.0975, 1.0995, 1.10, 1.11, 1.13, 1.16, 1.20, 1.30)
RATIO_IDLE_MAX  = 1.08   # assert exactly zero cycling at or below this ratio
RATIO_FULL_MIN  = 1.13   # assert full saturation at or above this ratio
DEGENERACY_BAND = 0.0004  # relative width around 1/RTE excluded from the bracket
# The band between them straddles the threshold, where the LP is nearly
# degenerate; it is plotted but not asserted.

SOLVER_TOL = 1e-6     # MWh / MW, comparisons against LP output


def load_battery(path: Path) -> dict:
    """Read the battery configuration, independently of the run script."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    bs  = cfg["battery_systems"]
    lim = cfg["operating_limits"]
    rte_dc  = float(bs["round_trip_efficiency_nominal"])
    pcu     = float(cfg["power_conditioning_unit"]["efficiency"])
    rte_ac  = rte_dc * pcu ** 2
    return {
        "e_cap":   float(bs["energy_capacity"]) / 1e6,   # Wh -> MWh
        "p_cap":   float(bs["power_capacity"]) / 1e6,    # W  -> MW
        "rte_ac":  rte_ac,
        "eta":     rte_ac ** 0.5,                        # symmetric split
        "soc_min": float(lim["soc_min"]),
        "soc_max": float(lim["soc_max"]),
        "e_cost":  float(cfg["economics"]["capex_EUR_per_kWh"]) * 1000.0,
        "p_cost":  float(cfg["economics"]["capex_EUR_per_kW"]) * 1000.0,
    }


# =============================================================================
# 2.  Synthetic signals
# =============================================================================
def sinusoidal_price(n_days: int) -> np.ndarray:
    """Daily sinusoid, minimum at hour 3 and maximum at hour 15 of each day."""
    t = np.arange(n_days * 24, dtype=float)
    return PRICE_MEAN - PRICE_AMP * np.cos(2.0 * np.pi * (t - 3.0) / 24.0)


def two_level_price(n_days: int, ratio: float) -> np.ndarray:
    """Twelve cheap hours followed by twelve expensive hours, each day."""
    day = np.concatenate([np.full(12, PRICE_LOW),
                          np.full(12, PRICE_LOW * ratio)])
    return np.tile(day, n_days)


def constant_wind(n: int) -> np.ndarray:
    return np.full(n, WIND_MW, dtype=float)


# =============================================================================
# 3.  Solving through the production code path
# =============================================================================
def _import_run_module():
    """Import the run script for _solve_shipp.

    The check exists to verify the production solver call rather than a copy of
    it, so the run script is imported rather than reimplemented. It is
    import-safe: everything heavy sits inside main().
    """
    import importlib
    try:
        return importlib.import_module("scripts.run_baseline")
    except ImportError:
        pass
    # Fall back to a direct file load, for the case where scripts/ is not a
    # package on sys.path.
    import importlib.util
    import sys
    from degradation.paths import REPO_ROOT
    path = REPO_ROOT / "scripts" / "run_baseline.py"
    if not path.exists():
        raise ImportError(f"could not find {path}")
    spec = importlib.util.spec_from_file_location("run_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before executing. run_baseline.py uses postponed annotations, and dataclasses resolves them through sys.modules[cls.__module__]; without
    # this the first @dataclass in the module raises AttributeError on None.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def solve_case(price: np.ndarray, bat: dict, label: str) -> dict:
    """Solve one dispatch case and return the arrays needed by the checks."""
    from shipp.components import Storage
    runmod = _import_run_module()

    n    = len(price)
    wind = constant_wind(n)
    stor = Storage(e_cap=bat["e_cap"], p_cap=bat["p_cap"],
                   eff_in=bat["eta"], eff_out=bat["eta"],
                   e_cost=bat["e_cost"], p_cost=bat["p_cost"],
                   dod=1.0 - bat["soc_min"])
    stor_null = Storage(e_cap=0.0, p_cap=0.0, eff_in=1.0, eff_out=1.0,
                        e_cost=0.0, p_cost=0.0)

    _os, os_fixed = runmod._solve_shipp(price, wind, stor, stor_null,
                                        P_MAX_MW, n, soc_max=bat["soc_max"])
    p = np.asarray(os_fixed.storage_p[0].data, dtype=float)
    # The kernel returns storage_e truncated to n (e_vec1[:n]), so the final
    # level e[n] is not in the array. The periodic constraint e[0] == e[n] is
    # always active, so e[n] is recovered as e[0]; the last balance step then
    # tests the periodic closure as well.
    e = np.asarray(os_fixed.storage_e[0].data, dtype=float)
    assert len(e) == n, f"expected storage_e of length {n}, got {len(e)}"
    return {"label": label, "price": price, "wind": wind,
            "p": p, "e": e, "n": n, "bat": bat}


# =============================================================================
# 4.  Result collection
# =============================================================================
CHECKS: list[dict] = []


def record(tid, quantity, value, unit="", note="") -> None:
    """A measured diagnostic with no pass/fail criterion. Reported, not asserted."""
    CHECKS.append({
        "test": tid, "quantity": quantity, "unit": unit,
        "expected": np.nan, "obtained": float(value),
        "rel_error": np.nan, "rtol": np.nan, "status": "INFO", "note": note,
    })


def check(tid, quantity, expected, obtained, unit="", rtol=1e-9, atol=0.0,
          note="") -> None:
    exp, obt = float(expected), float(obtained)
    denom = abs(exp) if abs(exp) > 0 else 1.0
    CHECKS.append({
        "test": tid, "quantity": quantity, "unit": unit,
        "expected": exp, "obtained": obt,
        "rel_error": abs(obt - exp) / denom, "rtol": rtol,
        "status": "PASS" if np.isclose(obt, exp, rtol=rtol, atol=atol) else "FAIL",
        "note": note,
    })


# =============================================================================
# T2.1  dispatch pattern on a sinusoidal price
# =============================================================================
def test_dispatch_pattern(sol: dict) -> None:
    bat, p, e, price = sol["bat"], sol["p"], sol["e"], sol["price"]
    e_lo = bat["e_cap"] * bat["soc_min"]
    e_hi = bat["e_cap"] * bat["soc_max"]

    # bounds and periodicity
    check("T2.1", "min SoC level", e_lo, e.min(), "MWh", atol=SOLVER_TOL,
          note="lower window bound")
    check("T2.1", "max SoC level", e_hi, e.max(), "MWh", atol=SOLVER_TOL,
          note="upper window bound")
    # e[n] is not returned; the periodic closure is tested in T2.2 through the
    # final balance step. What is checked here is that the returned trace ends
    # at a level the closing step can reach.
    record("T2.1", "SoC at the last returned step", e[-1], "MWh",
           note="e[n] equals e[0] by the kernel's periodic constraint")
    check("T2.1", "peak power magnitude", bat["p_cap"], np.abs(p).max(), "MW",
          atol=SOLVER_TOL)
    check("T2.1", "steps outside the power limit", 0.0,
          float(np.sum(np.abs(p) > bat["p_cap"] + SOLVER_TOL)), "steps")

    # charge low, discharge high. The weighted marginal ratio must clear 1/RTE
    # or the trade would not pay for its own losses.
    chg, dis = p < -SOLVER_TOL, p > SOLVER_TOL
    price_chg = float(np.average(price[chg], weights=-p[chg]))
    price_dis = float(np.average(price[dis], weights=p[dis]))
    # Closed form: with the window saturating daily, the battery must move a  known quantity of energy, so at a fixed power cap it fills whole hours in
    # merit order plus one partial hour. The weighted mean price is therefore exact, and unlike the per-hour powers it is unique even where two hours
    # carry identical prices.
    def _merit_order_mean(prices_day, energy, cap):
        num, rem = 0.0, energy
        for h in np.argsort(prices_day):
            q = min(cap, rem)
            num += q * prices_day[h]
            rem -= q
            if rem <= 1e-12:
                break
        return num / energy

    day = price[:24]
    win = bat["e_cap"] * (bat["soc_max"] - bat["soc_min"])
    check("T2.1", "weighted mean charge price",
          _merit_order_mean(day, win / bat["eta"], bat["p_cap"]),
          price_chg, "EUR/MWh", rtol=1e-8,
          note="cheapest hours in merit order")
    check("T2.1", "weighted mean discharge price",
          _merit_order_mean(-day, win * bat["eta"], bat["p_cap"]) * -1.0,
          price_dis, "EUR/MWh", rtol=1e-8,
          note="most expensive hours in merit order")
    record("T2.1", "discharge / charge price ratio", price_dis / price_chg, "-",
           note=f"must exceed 1/RTE = {1.0 / bat['rte_ac']:.4f}")
    # Both orderings follow arithmetically from the two merit-order checks above, so they are reported rather than asserted. A check that cannot
    # fail independently inflates the count without adding evidence.

    # energy moved, closed form when the window saturates once per day
    exp_dis = N_DAYS_SIN * bat["e_cap"] * (bat["soc_max"] - bat["soc_min"]) * bat["eta"]
    exp_chg = N_DAYS_SIN * bat["e_cap"] * (bat["soc_max"] - bat["soc_min"]) / bat["eta"]
    check("T2.1", "energy discharged over the week", exp_dis,
          float(np.sum(p[dis]) * DT_H), "MWh", rtol=1e-6)
    check("T2.1", "energy drawn to charge over the week", exp_chg,
          float(-np.sum(p[chg]) * DT_H), "MWh", rtol=1e-6)

    # degradation read-back: the dispatch must produce one deep cycle per day
    cycles = rainflow_cycle_counting(e, bat["e_cap"])
    deep = [c for c in cycles if c["dod"] > 0.5]
    check("T2.1", "deep cycle count", float(N_DAYS_SIN),
          sum(c["count"] for c in deep), "cycles", rtol=1e-9,
          note="one full cycle per price period")
    check("T2.1", "deep cycle depth", bat["soc_max"] - bat["soc_min"],
          max(c["dod"] for c in deep), "-", atol=1e-9)
    check("T2.1", "deep cycle mean SoC", 0.5 * (bat["soc_max"] + bat["soc_min"]),
          float(np.mean([c["soc_mean"] for c in deep])), "-", atol=1e-9)

    sol["cycles"] = cycles
    sol["n_shallow"] = sum(1 for c in cycles if c["dod"] <= 0.5)
    record("T2.1", "shallow rainflow records (delta <= 0.5)",
           float(sol["n_shallow"]), "records",
           note="zero on an exact trace; non-zero indicates LP degeneracy")


# =============================================================================
# T2.2  state-of-charge dynamics
# =============================================================================
def test_soc_dynamics(sol: dict) -> None:
    bat, p, e = sol["bat"], sol["p"], sol["e"]
    n = len(p)
    e_ext = np.append(e, e[0])          # e[n] == e[0], periodic horizon
    de = e_ext[1:n + 1] - e_ext[0:n]
    bound = np.minimum(-DT_H * bat["eta"] * p, -DT_H / bat["eta"] * p)
    slack = bound - de                         # must be non-negative

    check("T2.2", "steps violating the SoC inequality", 0.0,
          float(np.sum(slack < -SOLVER_TOL)), "steps",
          note="kernel imposes the balance as two upper bounds")
    check("T2.2", "largest violation", 0.0, float(max(0.0, -slack.min())),
          "MWh", atol=SOLVER_TOL)

    # Slack is permitted by the formulation, so these are measurements rather
    # than assertions. A correct model may discard energy where the objective
    # is indifferent; only the inequality above is a correctness condition.
    discarded = float(np.sum(slack[slack > SOLVER_TOL]))
    drawn_tot = float(-np.sum(np.minimum(p, 0.0)) * DT_H)
    record("T2.2", "steps with slack in the SoC balance",
           float(np.sum(slack > SOLVER_TOL)), "steps",
           note=f"out of {n}; these seed phantom micro-cycles")
    record("T2.2", "energy discarded over the horizon", discarded, "MWh")
    # The known slack mechanism needs an idle step at which the SoC is free to drift, i.e. strictly inside the window. When the window saturates once per
    # day the battery rests ON the lower bound, where drift is infeasible. This counts the population in which slack could have appeared at all.
    e_lo = bat["e_cap"] * bat["soc_min"]
    e_hi = bat["e_cap"] * bat["soc_max"]
    idle = np.abs(p) <= SOLVER_TOL
    interior = (e > e_lo + 1e-6) & (e < e_hi - 1e-6)
    record("T2.2", "idle steps strictly inside the window",
           float(np.sum(idle & interior)), "steps",
           note="population in which free SoC drift is possible")    

    check("T2.2", "discarded energy as a share of energy drawn", 0.0,
          discarded / drawn_tot if drawn_tot > 0 else 0.0, "-", atol=1e-3,
          note="must be negligible, need not be exactly zero")

    # the loss on a full round trip must equal 1 - RTE of the energy drawn
    chg, dis = p < -SOLVER_TOL, p > SOLVER_TOL
    drawn = float(-np.sum(p[chg]) * DT_H)
    delivered = float(np.sum(p[dis]) * DT_H)
    check("T2.2", "delivered / drawn energy", bat["rte_ac"],
          delivered / drawn, "-", rtol=1e-6,
          note="closes the round trip through the LP")


# =============================================================================
# T2.3  round-trip efficiency threshold
# =============================================================================
def test_rte_threshold(bat: dict) -> dict:
    threshold = 1.0 / bat["rte_ac"]
    window = bat["e_cap"] * (bat["soc_max"] - bat["soc_min"])
    exp_full = N_DAYS_SQ * window * bat["eta"]

    ratios, discharged = [], []
    for r in RATIO_SWEEP:
        sol = solve_case(two_level_price(N_DAYS_SQ, r), bat, f"square r={r:.3f}")
        p = sol["p"]
        discharged.append(float(np.sum(p[p > SOLVER_TOL]) * DT_H))
        ratios.append(r)
        print(f"    r = {r:6.4f}   discharged {discharged[-1]:9.3f} MWh")

    ratios = np.asarray(ratios)
    discharged = np.asarray(discharged)

    for r, d in zip(ratios, discharged):
        if r <= RATIO_IDLE_MAX:
            check("T2.3", f"discharge at price ratio {r:.3f}", 0.0, d, "MWh",
                  atol=1e-3, note="below the threshold, arbitrage loses money")
        elif r >= RATIO_FULL_MIN:
            check("T2.3", f"discharge at price ratio {r:.3f}", exp_full, d,
                  "MWh", rtol=1e-6, note="above the threshold, window saturates")

    # Within a narrow band around the threshold the LP is nearly degenerate:
    # at r = 1.100 the margin is 0.05 EUR per MWh diverted. Those points are
    # plotted but excluded from the bracket, which is otherwise at the mercy of
    # the solver's optimality tolerance rather than of the model.
    far = np.abs(ratios / threshold - 1.0) > DEGENERACY_BAND
    active = discharged > 1e-3
    r_last_idle = float(ratios[far & ~active].max()) if (far & ~active).any() else np.nan
    r_first_act = float(ratios[far & active].min()) if (far & active).any() else np.nan
    record("T2.3", "ratios excluded as near-degenerate",
           float(np.sum(~far)), "-",
           note=f"within {DEGENERACY_BAND:.1%} of the threshold")
    check("T2.3", "threshold lies above the last idle ratio", 1.0,
          float(threshold > r_last_idle), "-",
          note=f"last idle ratio {r_last_idle:.4f}")
    check("T2.3", "threshold lies at or below the first active ratio", 1.0,
          float(threshold <= r_first_act), "-",
          note=f"first active ratio {r_first_act:.4f}")
    check("T2.3", "predicted threshold 1/RTE", 1.0989010312, threshold, "-",
          rtol=1e-9, note="from the YAML efficiencies")

    return {"ratios": ratios, "discharged": discharged,
            "threshold": threshold, "exp_full": exp_full}


# =============================================================================
# 5.  Figures
# =============================================================================
def figure_week(sol: dict, out_dir: Path) -> None:
    pal = apply_thesis_style(palette="brand", usetex=False)
    bat = sol["bat"]
    # The solve covers N_DAYS_SIN days. Only the first N_DAYS_PLOT are drawn:
    # at full text width, 168 hourly bars are under 1 mm wide on the page and
    # the two-level structure of each charge and discharge event is lost.
    n = min(len(sol["p"]), 24 * N_DAYS_PLOT)
    p, price = sol["p"][:n], sol["price"][:n]
    e_ext = sol["e"][:n + 1] if len(sol["e"]) > n else np.append(sol["e"], sol["e"][0])
    t = np.arange(n)

    fig, ax = plt.subplots(3, 1, figsize=figsize(1.0, aspect=0.85), sharex=True)

    ax[0].plot(t, price, color=TUDELFT["navy"], lw=1.1)
    ax[0].set_ylabel("Price\n(EUR/MWh)")

    ax[1].plot(np.arange(len(e_ext)), e_ext / bat["e_cap"],
               color=TUDELFT["blue"], lw=1.1)
    for lvl in (bat["soc_min"], bat["soc_max"]):
        ax[1].axhline(lvl, color=pal["neutral"], lw=0.7, ls=(0, (5, 4)), alpha=0.8)
    ax[1].set_ylabel("State of charge\n(-)")
    ax[1].set_ylim(0.0, 1.0)

    ax[2].axhline(0.0, color=pal["neutral"], lw=0.7, alpha=0.6)
    ax[2].fill_between(t, 0.0, p, where=p >= 0, step="mid",
                       color=TUDELFT["darkred"], alpha=0.85, linewidth=0)
    ax[2].fill_between(t, 0.0, p, where=p < 0, step="mid",
                       color=TUDELFT["blue"], alpha=0.85, linewidth=0)
    ax[2].set_ylim(-1.18 * bat["p_cap"], 1.18 * bat["p_cap"])
    ax[2].set_ylabel("Battery power\n(MW)")
    ax[2].set_xlabel("Time step  (h)")
    ax[2].set_xlim(0, n)
    ax[2].set_xticks(np.arange(0, n + 1, 24))
    ax[2].text(0.99, 0.92, "discharge", transform=ax[2].transAxes, ha="right",
               va="top", fontsize=FS_ANNOT, color=TUDELFT["darkred"])
    ax[2].text(0.99, 0.08, "charge", transform=ax[2].transAxes, ha="right",
               va="bottom", fontsize=FS_ANNOT, color=TUDELFT["blue"])

    out_dir.mkdir(parents=True, exist_ok=True)
    if OUTPUT in ("pdf", "both"):
        fig.savefig(out_dir / "fig_verify_dispatch_week.pdf")
    if OUTPUT in ("png", "both"):
        fig.savefig(out_dir / "fig_verify_dispatch_week.png", dpi=DPI)
    plt.close(fig)


def figure_threshold(res: dict, out_dir: Path) -> None:
    pal = apply_thesis_style(palette="brand", usetex=False)
    fig, axes = plt.subplots(1, 2, figsize=figsize(1.0, aspect=0.44))

    thr, full = res["threshold"], res["exp_full"]
    for k, ax in enumerate(axes):
        # Left: line plus markers, where the connecting segments run along the two plateaus and are real. Right: markers only. Nothing was sampled
        # between the bracketing ratios, so a connecting line there would draw a ramp through values that were never computed.
        ax.plot(res["ratios"], res["discharged"], color=TUDELFT["navy"],
                lw=1.3 if k == 0 else 0.0,
                ls="-" if k == 0 else "none",
                marker="o", markersize=4)
        ax.axhline(full, color=pal["neutral"], lw=0.7, ls=(0, (5, 4)),
                   alpha=0.8)
        ax.axvline(thr, color=TUDELFT["darkred"], lw=1.0, ls=":")
        ax.set_ylim(-0.05 * full, 1.15 * full)
        ax.set_xlabel("Price ratio  (-)")

    # left: full sweep, showing that nothing happens away from the threshold
    axes[0].set_ylabel("Energy discharged\nover three days  (MWh)")
    axes[0].text(res["ratios"].max(), full * 1.03, "saturated window",
                 fontsize=FS_ANNOT, color=pal["neutral"], ha="right",
                 va="bottom")

    # right: detail around the transition
    lo = res["ratios"][res["discharged"] <= 1e-3].max()
    hi = res["ratios"][res["discharged"] > 1e-3].min()
    axes[1].set_xlim(lo - 0.0015, hi + 0.0015)
    # Ticks are the two ratios that were solved; the predicted threshold gets the dotted line and its own label, so measurement and prediction are not
    # mixed on the same axis. Labelling all three collides at this zoom.
    axes[1].set_xticks([lo, hi])
    axes[1].set_xticklabels([f"{lo:.4f}", f"{hi:.4f}"])
    axes[1].tick_params(axis="x", labelsize=FS_ANNOT)
    axes[1].text(thr, 0.30 * full, f"  1/RTE = {thr:.4f}", fontsize=FS_ANNOT,
                 color=TUDELFT["darkred"], ha="left", va="center")

    out_dir.mkdir(parents=True, exist_ok=True)
    if OUTPUT in ("pdf", "both"):
        fig.savefig(out_dir / "fig_verify_rte_threshold.pdf")
    if OUTPUT in ("png", "both"):
        fig.savefig(out_dir / "fig_verify_rte_threshold.png", dpi=DPI)
    plt.close(fig)


# =============================================================================
# 6.  Reporting
# =============================================================================
def report(out_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(CHECKS)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "tier2_verification.csv", index=False)

    width = max(len(r["quantity"]) for r in CHECKS) + 2
    print()
    print("=" * (width + 58))
    print(f"{'test':<6}{'quantity':<{width}}{'expected':>16}{'obtained':>16}"
          f"{'rel err':>12}  status")
    print("=" * (width + 58))
    last = None
    for r in CHECKS:
        if last is not None and r["test"] != last:
            print("-" * (width + 58))
        last = r["test"]
        exp_s = "-" if np.isnan(r["expected"]) else f"{r['expected']:.8g}"
        err_s = "-" if np.isnan(r["rel_error"]) else f"{r['rel_error']:.2e}"
        print(f"{r['test']:<6}{r['quantity']:<{width}}"
              f"{exp_s:>16}{r['obtained']:>16.8g}{err_s:>12}  {r['status']}")
    print("=" * (width + 58))
    n_fail = int((df["status"] == "FAIL").sum())
    print(f"{len(df)} checks, {len(df) - n_fail} passed, {n_fail} failed")

    rows = []
    for r in CHECKS:
        q = r["quantity"].replace("_", r"\_").replace("%", r"\%")
        e_s = "--" if np.isnan(r["expected"]) else f"{r['expected']:.6g}"
        r_s = "--" if np.isnan(r["rel_error"]) else f"{r['rel_error']:.1e}"
        rows.append(f"{r['test']} & {q} & {e_s} & "
                    f"{r['obtained']:.6g} & {r_s} & {r['status']} \\\\")
    tex = ("\\begin{tabular}{llrrrl}\n\\hline\\noalign{\\smallskip}\n"
           "Test & Quantity & Expected & Obtained & Rel. error & Status \\\\\n"
           "\\noalign{\\smallskip}\\hline\\noalign{\\smallskip}\n"
           + "\n".join(rows)
           + "\n\\noalign{\\smallskip}\\hline\n\\end{tabular}\n")
    (out_dir / "tier2_verification_table.tex").write_text(tex, encoding="utf-8")

    if n_fail:
        raise SystemExit(f"{n_fail} verification check(s) failed.")
    return df


def main() -> None:
    bat = load_battery(BATTERY_YAML)
    print("Tier 2 verification: dispatch on synthetic price signals")
    print(f"  battery   : {bat['e_cap']:.0f} MWh / {bat['p_cap']:.0f} MW, "
          f"window {bat['soc_min']:.0%}-{bat['soc_max']:.0%}")
    print(f"  RTE_ac    : {bat['rte_ac']:.9f}   eta = {bat['eta']:.10f}")
    print(f"  threshold : price ratio 1/RTE = {1.0 / bat['rte_ac']:.7f}")
    print(f"  wind      : constant {WIND_MW:.0f} MW, grid limit {P_MAX_MW:.0f} MW")

    print(f"\n  [T2.1/T2.2] sinusoidal price, {N_DAYS_SIN} days ...")
    sol = solve_case(sinusoidal_price(N_DAYS_SIN), bat, "sinusoid")
    test_dispatch_pattern(sol)
    test_soc_dynamics(sol)
    print(f"    shallow rainflow records (delta <= 0.5): {sol['n_shallow']}")

    print(f"\n  [T2.3] two-level price sweep, {N_DAYS_SQ} days per solve ...")
    res = test_rte_threshold(bat)

    figure_week(sol, OUT_DIR)
    figure_threshold(res, OUT_DIR)
    report(OUT_DIR)
    print(f"\nwrote CSV, LaTeX table and two figures to {OUT_DIR}")


if __name__ == "__main__":
    main()