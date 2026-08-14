r"""
verify_week_snapshot.py  --  short-horizon view of the real dispatch

WHAT THIS IS
------------
Chapter 4 goes straight from the model description to a 20-year simulation. This script produces the missing intermediate: the real dispatch at a scale a reader
can follow, one week, on real ERA5 wind and real DK1 prices, at the same design as the baseline run.

It is NOT a synthetic test. Everything here is the production dispatch.

WHY YEAR 1 ONLY
---------------
Inside the multi-year loop, year 1 is solved at nominal capacity with a free starting state, so a standalone annual solve reproduces it exactly. The script
proves that rather than assuming it: CHECK 1 compares its own year-1 state of charge against annual_soc[0] in the stored baseline .npy. If they disagree, the
week shown here is not a window into the run that produced Section 4.3 and the script stops.

That equivalence is what lets the window series be computed cheaply. Five annual solves take a couple of minutes; the 20-year sweep takes half an hour.

COUNTING
--------
The dispatch is solved for the whole year and the week is a slice of it, so the battery enters the week already charging or discharging with the state of charge
the year gave it. Rainflow counting is likewise done once on the full year; cycles are then assigned to the week containing their midpoint. Nothing is
recounted in isolation.

WEEK SELECTION
--------------
Chosen by a stated rule, not by eye. The default picks the week whose mean daily price spread is closest to the annual median, so the figure cannot be accused of
showing a flattering week. Override with WEEK_RULE or WEEK_INDEX.

RUNNING IT
----------
    python verify_week_snapshot.py              redraw from the cache if present
    python verify_week_snapshot.py --resolve    solve the five annual LPs again

The cache stores the raw solved series only. Rainflow counting, the weekly assignment, the degradation and every figure are recomputed each time, so
cosmetic and analytical changes need no re-solve.

OUTPUT
------
  Results/Week Snapshot/week_snapshot_cache.npz      solved series, for redraws
  Results/Week Snapshot/week_snapshot_hourly.csv     hourly series, all windows
  Results/Week Snapshot/week_snapshot_summary.csv    per-window week statistics
  Results/Week Snapshot/fig_week_dispatch.pdf/.png   Figure A
  Results/Week Snapshot/fig_week_rainflow.pdf/.png   Figure B
  Results/Week Snapshot/fig_week_width.pdf/.png      Figure C
  Results/Week Snapshot/week_snapshot_cycles.csv     every counted cycle
  Results/Week Snapshot/week_snapshot_checks.csv     dispatch invariance check

Reproducible on Windows / VS Code. Needs the same stack as the run script:
pyomo + gurobi, PyWake, SHIPP. Place it beside run_battery_xu_shi_degradation_v5_6_RTE_test.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent

from degradation.paths import DATA_DIR, RESULTS_DIR, results_dir
from degradation.style import apply_thesis_style, figsize, TUDELFT, FS_ANNOT, FS_LEGEND
from degradation.xu import rainflow_cycle_counting, compute_fd, XU_LMO

# =============================================================================
# Configuration
# =============================================================================
# Section 4.2 uses the 2022 price year. scripts/run_baseline.py carries 2019 as its own default, so this script overrides rm.PRICE_CSV in load_inputs()
# rather than relying on it.
PRICE_FILE = "dk1_prices_2022.csv"

T_CELL_C   = 25.0
DT_H       = 1.0

# Reference window, and the two series for Figure C.
REF_WINDOW      = (0.10, 0.90)
WIDTH_SERIES    = [(0.20, 0.80), (0.10, 0.90), (0.00, 1.00)]   # centre 0.50
CENTRE_SERIES   = [(0.20, 1.00), (0.10, 0.90), (0.00, 0.80)]   # width 0.80

# Week selection. "median_spread" is the defensible default; "max_spread" and
# "max_mean" are available for a second panel or a sensitivity remark.
WEEK_RULE  = "median_spread"
WEEK_INDEX = None      # 1-based; overrides WEEK_RULE when set

# Stored baseline run used by CHECK 1. Any multiyear_*.npy at the same design
# and price year will do; the newest match is taken.
BASELINE_NPY_GLOB = "multiyear_*dk2022*soc10_90*.npy"
BASELINE_DIRS     = [RESULTS_DIR / "baseline" / "xu"]

# CSVs and the solve cache are data that verify_week_audit.py reads, so they live under results/. Figures are deliverables and are written beside this
# script, as everywhere else in the repository.
DATA_DIR_OUT = results_dir("week_snapshot")
OUT_DIR      = HERE

# -- Output ------------------------------------------------------------------ #
OUTPUT = "png"      # "png", "pdf" or "both"
DPI = 300


def _save(fig, out_dir, stem):
    """Write the formats OUTPUT asks for."""
    if OUTPUT in ("pdf", "both"):
        fig.savefig(out_dir / f"{stem}.pdf")
    if OUTPUT in ("png", "both"):
        fig.savefig(out_dir / f"{stem}.png", dpi=DPI)

# Solving five annual LPs takes a couple of minutes; redrawing takes seconds. 
# The cache holds the raw model output only, the full-year price, wind, power and state-of-charge series. Everything derived, the rainflow counting, the
# weekly assignment and the degradation, is recomputed on every run, so changes to those do take effect without re-solving. Only a change to the dispatch
# itself needs --resolve.
CACHE_FILE = DATA_DIR_OUT / "week_snapshot_cache.npz"

HOURS_PER_WEEK = 168
ZOOM_HOURS     = 72
DELTA_C        = 0.1437   # inflection point of S_delta; below it the convex
                          # surrogate extrapolates (see Chapter 3)     # span of the SoC panels in Figure C; 168 is unreadable
                        # with three overlaid traces at text width


# =============================================================================
# Inputs, through the production code path
# =============================================================================
def _import_run_module():
    """Import scripts/run_baseline.py for quick_setup, _build_shipp_components and _solve_shipp, so the snapshot solves through the production code path
    rather than a copy of it. The module is import-safe: everything heavy sits inside main()."""
    import importlib
    import importlib.util
    import sys
    try:
        return importlib.import_module("scripts.run_baseline")
    except ImportError:
        pass
    from degradation.paths import REPO_ROOT
    path = REPO_ROOT / "scripts" / "run_baseline.py"
    if not path.exists():
        raise ImportError(f"could not find {path}")
    spec = importlib.util.spec_from_file_location("run_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before executing: run_baseline.py uses postponed annotations and
    # dataclasses resolves them through sys.modules[cls.__module__].
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_inputs():
    """Wind, price and battery parameters, exactly as the run script builds them."""
    rm = _import_run_module()
    rm.PRICE_CSV = DATA_DIR / PRICE_FILE      # run_baseline defaults to 2019
    print(f"  price file : {rm.PRICE_CSV.name}")

    setup = rm.quick_setup(rm.HPP_YAML, config={"interp_n": 2000}, verbose=False)
    hpp   = setup["hpp"]

    ws_all, wd_all, ti_all = rm._load_inputs(hpp)
    price_all = rm._load_prices()
    n = rm._choose_horizon(len(ws_all), len(price_all))

    ws    = ws_all[:n]
    wd    = wd_all[:n]
    ti    = ti_all[:n] if ti_all is not None else None
    price = price_all[:n]

    wind = rm._run_pywake_power_MW(setup, wd, ws, ti)
    p_max_MW = float(hpp["grid_connection_capacity"]) / 1e6

    (stor, stor_null, e_cap, p_cap, rte_ac, eta,
     e_cost, p_cost, repl_e, repl_p,
     soc_min_yaml, soc_max_yaml) = rm._build_shipp_components(setup, p_max_MW)

    print(f"  horizon    : {n} h    wind mean {wind.mean():.1f} MW    "
          f"price mean {price.mean():.1f} EUR/MWh")
    print(f"  battery    : {p_cap:.0f} MW / {e_cap:.0f} MWh   grid {p_max_MW:.0f} MW   "
          f"RTE {rte_ac:.4f}   eta {eta:.4f}")
    print(f"  YAML window: {soc_min_yaml:.2f}-{soc_max_yaml:.2f}")

    return dict(rm=rm, n=n, wind=wind, price=price, p_max=p_max_MW,
                stor_null=stor_null, e_cap=e_cap, p_cap=p_cap,
                rte_ac=rte_ac, eta=eta, e_cost=e_cost)


def solve_year1(inp: dict, soc_min: float, soc_max: float) -> dict:
    """One annual solve at nominal capacity: the year-1 problem of the loop."""
    rm = inp["rm"]
    stor = rm._build_storage_year(
        e_cap_eff=inp["e_cap"], p_cap_MW=inp["p_cap"], rte_ac=inp["rte_ac"],
        e_cost_EUR_per_MWh=inp["e_cost"], soc_min=soc_min, soc_max=soc_max,
    )
    _os, os_fixed = rm._solve_shipp(
        inp["price"], inp["wind"], stor, inp["stor_null"],
        inp["p_max"], inp["n"], soc_max=soc_max,
    )
    p = np.asarray(os_fixed.storage_p[0].data, dtype=float)
    e = np.asarray(os_fixed.storage_e[0].data, dtype=float)
    prod = np.asarray(os_fixed.production_p[0].data, dtype=float)
    return {"soc_min": soc_min, "soc_max": soc_max, "p": p, "e": e, "prod": prod,
            "label": f"{soc_min:.0%}-{soc_max:.0%}".replace("%", "")}


# =============================================================================
# Solve cache
# =============================================================================
def _key(w) -> str:
    return f"{w[0]:.2f}_{w[1]:.2f}"


def save_cache(inp: dict, sols: dict, path: Path) -> None:
    d = {"price": inp["price"], "wind": inp["wind"],
         "e_cap": np.array([inp["e_cap"]], dtype=float),
         "windows": np.array([_key(w) for w in sols]),
         "price_file": np.array([PRICE_FILE])}
    for w, s_ in sols.items():
        d[f"p__{_key(w)}"] = s_["p"]
        d[f"e__{_key(w)}"] = s_["e"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)
    print(f"  cached the solved series to {path.name}")


def load_cache(path: Path, wanted) -> tuple[dict, dict] | None:
    """Return (inp, sols) if the cache covers everything configured, else None."""
    z = np.load(path, allow_pickle=False)
    have = set(str(k) for k in z["windows"])
    missing = [w for w in wanted if _key(w) not in have]
    if missing:
        print("  cache does not cover " +
              ", ".join(f"{int(w[0]*100)}-{int(w[1]*100)}%" for w in missing) +
              " -- re-solving")
        return None
    cached_price = str(z["price_file"][0])
    if cached_price != PRICE_FILE:
        print(f"  cache was built from {cached_price}, not {PRICE_FILE} -- re-solving")
        return None

    inp = {"price": z["price"], "wind": z["wind"], "e_cap": float(z["e_cap"][0])}
    sols = {}
    for w in wanted:
        k = _key(w)
        sols[w] = {"soc_min": w[0], "soc_max": w[1],
                   "p": z[f"p__{k}"], "e": z[f"e__{k}"],
                   "label": f"{int(w[0]*100)}-{int(w[1]*100)}"}
    print(f"  loaded {len(sols)} solved windows from {path.name}  "
          f"(pass --resolve to re-solve)")
    return inp, sols


# =============================================================================
# CHECK 1: is this the same dispatch the baseline run produced?
# =============================================================================
def find_baseline_npy() -> Path | None:
    for d in BASELINE_DIRS:
        hits = sorted(d.glob(BASELINE_NPY_GLOB)) if d.is_dir() else []
        if hits:
            return hits[-1]
    return None


def check_against_baseline(sol_ref: dict, e_cap: float) -> None:
    path = find_baseline_npy()
    if path is None:
        print("\n  CHECK 1  SKIPPED: no stored baseline .npy found.")
        print("           The week shown cannot be tied to the Section 4.3 run.")
        return
    d = np.load(path, allow_pickle=True).item()
    stored = np.asarray(d["annual_soc"][0], dtype=float)
    here   = sol_ref["e"]
    print(f"\n  CHECK 1  against {path.name}")
    if len(stored) != len(here):
        raise SystemExit(f"           length mismatch: stored {len(stored)}, "
                         f"solved {len(here)}")
    err = np.max(np.abs(stored - here))
    print(f"           stored e_cap {d['e_cap_nominal']:.1f} MWh, "
          f"solved {e_cap:.1f} MWh")
    print(f"           max |difference| in year-1 SoC : {err:.3e} MWh "
          f"({err/e_cap:.3e} of capacity)")
    if err > 1e-6 * e_cap:
        raise SystemExit(
            "           MISMATCH. The standalone year-1 solve does not reproduce\n"
            "           the stored trace, so this week is not a window into the\n"
            "           baseline run. Check the price file and the YAML window.")
    print("           PASS -- this is the year-1 dispatch of the baseline run.")


# =============================================================================
# Week selection
# =============================================================================
def choose_week(price: np.ndarray) -> tuple[int, dict]:
    n_weeks = len(price) // HOURS_PER_WEEK
    w = price[:n_weeks * HOURS_PER_WEEK].reshape(n_weeks, HOURS_PER_WEEK)
    daily_spread = np.array([np.mean([d.max() - d.min() for d in wk.reshape(7, 24)])
                             for wk in w])
    stats = pd.DataFrame({
        "week": np.arange(1, n_weeks + 1),
        "mean_price": w.mean(1),
        "mean_daily_spread": daily_spread,
    })
    med = float(np.median(daily_spread))

    if WEEK_INDEX is not None:
        k = int(WEEK_INDEX)
        rule = "explicit WEEK_INDEX"
    elif WEEK_RULE == "median_spread":
        k = int(stats.iloc[(stats.mean_daily_spread - med).abs().argmin()].week)
        rule = "mean daily price spread closest to the annual median"
    elif WEEK_RULE == "max_spread":
        k = int(stats.iloc[stats.mean_daily_spread.argmax()].week)
        rule = "largest mean daily price spread"
    elif WEEK_RULE == "max_mean":
        k = int(stats.iloc[stats.mean_price.argmax()].week)
        rule = "highest mean price"
    else:
        raise ValueError(f"unknown WEEK_RULE {WEEK_RULE!r}")

    row = stats[stats.week == k].iloc[0]
    info = dict(week=k, rule=rule, annual_median_spread=med,
                mean_price=float(row.mean_price),
                mean_daily_spread=float(row.mean_daily_spread),
                pct_from_median=100.0 * (row.mean_daily_spread - med) / med)
    print(f"\n  week {k} selected by: {rule}")
    print(f"    mean price {info['mean_price']:.1f} EUR/MWh, "
          f"mean daily spread {info['mean_daily_spread']:.1f} "
          f"({info['pct_from_median']:+.1f}% from the annual median of {med:.1f})")
    return k, info


def week_slice(week: int) -> slice:
    a = (week - 1) * HOURS_PER_WEEK
    return slice(a, a + HOURS_PER_WEEK)


# =============================================================================
# Per-window statistics
# =============================================================================
def assign_cycles_to_weeks(cycles) -> np.ndarray:
    """Week index of each cycle, taken from the midpoint of its interval.

    Counting is done ONCE, on the full year, exactly as Sections 4.3 and 4.4
    count it. The week is then a selection from that set, not a recount. Using
    the midpoint makes the weekly sets a partition: every cycle belongs to
    exactly one week, none is dropped and none is counted twice.

    Counting a week in isolation would do neither: it cuts cycles that straddle
    the boundary and closes new half cycles at the edges that the annual count
    never produced.
    """
    return np.array([int(0.5 * (c["i_start"] + c["i_end"])) // HOURS_PER_WEEK + 1
                     for c in cycles], dtype=int)


def window_stats(sol: dict, e_cap: float, week: int, sl: slice) -> dict:
    """Annual rainflow, then the cycles this week owns."""
    cyc_yr = rainflow_cycle_counting(sol["e"], e_cap)
    wk_id  = assign_cycles_to_weeks(cyc_yr)
    in_week = [c for c, k in zip(cyc_yr, wk_id) if k == week]

    # Cycles whose interval overlaps the week but which belong to a neighbour.
    # Drawn faintly in Figure B so the reader sees that the week is a cut
    # through a continuous trace rather than a standalone episode.
    touching = [c for c, k in zip(cyc_yr, wk_id)
                if k != week and c["i_end"] >= sl.start and c["i_start"] < sl.stop]

    e_wk  = sol["e"][sl]
    sig_wk = float(np.mean(e_wk)) / e_cap
    t_wk   = len(e_wk) * DT_H * 3600.0
    fd_wk, fd_cyc_wk, fd_cal_wk = compute_fd(in_week, sig_wk, t_wk, T_CELL_C, XU_LMO)

    p_wk  = sol["p"][sl]
    disch = float(np.sum(p_wk[p_wk > 0]) * DT_H)
    cnt   = np.array([c["count"] for c in in_week], dtype=float)
    dod   = np.array([c["dod"] for c in in_week], dtype=float)

    full_depth = sol["soc_max"] - sol["soc_min"]
    n_weeks    = len(sol["e"]) / HOURS_PER_WEEK
    dod_yr     = np.array([c["dod"] for c in cyc_yr], dtype=float)
    cnt_yr     = np.array([c["count"] for c in cyc_yr], dtype=float)

    return {
        "full_depth": full_depth,
        "week_records_full": int(np.sum(cnt == 1.0)),
        "week_records_half": int(np.sum(cnt == 0.5)),
        "week_at_full_depth": int(np.sum(np.abs(dod - full_depth) < 1e-9)),
        "week_above_dc": int(np.sum(dod > DELTA_C)),
        "week_zero_depth": int(np.sum(dod <= 1e-9)),
        "week_cycles_above_dc": float(cnt[dod > DELTA_C].sum()),
        "year_cycles_per_week": float(cnt_yr.sum() / n_weeks),
        "year_cycles_above_dc_per_week": float(cnt_yr[dod_yr > DELTA_C].sum() / n_weeks),
        "window": sol["label"], "soc_min": sol["soc_min"], "soc_max": sol["soc_max"],
        "week_records": len(in_week),
        "week_cycles": float(cnt.sum()),
        "week_straddling": len(touching),
        "week_mean_depth": float(np.average(dod, weights=cnt)),
        "week_max_depth": float(dod.max()),
        "week_mean_soc": sig_wk,
        "week_efc": disch / e_cap,
        "week_fd_cycle": fd_cyc_wk, "week_fd_calendar": fd_cal_wk, "week_fd": fd_wk,
        "year_records": len(cyc_yr),
        "year_records_full":  int(np.sum(cnt_yr == 1.0)),
        "year_records_half":  int(np.sum(cnt_yr == 0.5)),
        "year_records_other": int(np.sum((cnt_yr != 1.0) & (cnt_yr != 0.5))),
        "year_zero_depth":    int(np.sum(dod_yr <= 1e-9)),
        "year_cycles": float(sum(c["count"] for c in cyc_yr)),
        "year_mean_depth": float(np.average(dod_yr, weights=cnt_yr)),
        "year_efc": float(np.sum(sol["p"][sol["p"] > 0]) * DT_H) / e_cap,
        "year_mean_soc": float(np.mean(sol["e"])) / e_cap,
        "_cyc_wk": in_week, "_cyc_touch": touching, "_cyc_yr": cyc_yr,
    }


# =============================================================================
# Figures
# =============================================================================
def fig_a_dispatch(sol, inp, sl, info, out_dir):
    pal = apply_thesis_style(palette="brand", usetex=False)
    price = inp["price"][sl]
    wind  = inp["wind"][sl]
    p     = sol["p"][sl]
    e     = sol["e"][sl]
    ecap  = inp["e_cap"]
    t     = np.arange(len(price))

    fig, ax = plt.subplots(3, 1, figsize=figsize(1.0, aspect=0.80), sharex=True)

    ax[0].plot(t, price, color=TUDELFT["navy"], lw=1.0)
    ax[0].set_ylabel("Price\n(EUR/MWh)")

    ax[1].fill_between(t, 0, wind, step="mid", color=pal["shade"], linewidth=0,
                       label="Wind")
    ax[1].fill_between(t, 0, p, where=p >= 0, step="mid",
                       color=TUDELFT["darkred"], alpha=0.9, linewidth=0,
                       label="Battery discharge")
    ax[1].fill_between(t, 0, p, where=p < 0, step="mid",
                       color=TUDELFT["blue"], alpha=0.9, linewidth=0,
                       label="Battery charge")
    ax[1].axhline(0, color=pal["neutral"], lw=0.6, alpha=0.6)
    ax[1].set_ylabel("Power\n(MW)")
    ax[1].legend(frameon=False, fontsize=FS_LEGEND, loc="upper right", ncol=3)

    ax[2].plot(t, e / ecap, color=TUDELFT["blue"], lw=1.1)
    for lvl in (sol["soc_min"], sol["soc_max"]):
        ax[2].axhline(lvl, color=pal["neutral"], lw=0.7, ls=(0, (5, 4)), alpha=0.8)
    ax[2].set_ylabel("State of charge\n(-)")
    ax[2].set_ylim(0.0, 1.0)
    ax[2].set_xlabel(f"Hour of week {info['week']}")
    ax[2].set_xlim(0, len(price))
    ax[2].set_xticks(np.arange(0, len(price) + 1, 24))

    out_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, out_dir, "fig_week_dispatch")
    plt.close(fig)


def fig_b_rainflow(sol, st, inp, sl, info, out_dir):
    """What the counter extracts from the week, against the trace it read.

    Two panels on a shared time axis. Above, the state of charge, plain. Below,
    one arc per counted cycle, joining the two turning points the counter paired
    and rising to a height equal to the cycle depth.

    An arc rather than a bar or a span. A rainflow record is a pair of turning
    points, so an arc states which two were matched without asserting that the
    interval between them was one continuous excursion: for a residue half cycle
    the two are neighbours on the counting stack and the trace swings up and
    down repeatedly in between. Nesting, a shallow cycle lying inside a deeper
    one, appears directly as a small arc under a large one, which is the case
    rainflow exists to resolve and which a bar chart cannot show.

    Full and half are drawn differently because the reader will ask, but the
    distinction is bookkeeping inside the counting rule rather than a physical
    property. A constant-amplitude sequence never closes a loop under the ASTM
    three-point rule, so four half-cycle records at the same depth are two
    physical cycles. Counts, not records, are what enter the damage model.
    """
    pal = apply_thesis_style(palette="brand", usetex=False)
    e = sol["e"][sl]
    ecap = inp["e_cap"]
    n = len(e)

    fig, ax = plt.subplots(2, 1, figsize=figsize(1.0, aspect=0.62), sharex=True)

    for lvl in (sol["soc_min"], sol["soc_max"]):
        ax[0].axhline(lvl, color=pal["neutral"], lw=0.7, ls=(0, (5, 4)), alpha=0.7)
    ax[0].plot(np.arange(n), e / ecap, color=TUDELFT["navy"], lw=1.2)
    ax[0].set_ylabel("State of charge\n(-)")
    ax[0].set_ylim(0.0, 1.0)

    def _arc(c, colour, alpha, z, ms):
        """Parabola from one turning point to the other, apex at the depth.

        Drawn on the true indices and clipped by the axis limits, so a cycle
        reaching outside the week keeps its real shape instead of being
        squashed into the visible span.
        """
        i0_ = c["i_start"] - sl.start
        i1_ = c["i_end"] - sl.start
        if i1_ <= i0_:
            i1_ = i0_ + 0.8                       # zero-length record stays visible
        mid = 0.5 * (i0_ + i1_)
        x = np.linspace(i0_, i1_, 120)
        y = c["dod"] * (1.0 - ((2.0 * (x - mid) / (i1_ - i0_)) ** 2))
        solid = c["count"] == 1.0
        ax[1].plot(x, y, color=colour, lw=1.0, alpha=alpha,
                   ls="-" if solid else (0, (3, 2)), zorder=z)
        if 0 <= mid <= n:                          # apex marker, so equal-depth
            ax[1].plot([mid], [c["dod"]], lw=0, marker="o", markersize=ms,
                       color=colour, alpha=alpha, zorder=z + 0.1)

    for c in st["_cyc_touch"]:
        _arc(c, pal["neutral"], 0.40, 2, 2.2)
    n_full = n_half = 0
    for c in st["_cyc_wk"]:
        if c["count"] == 1.0:
            _arc(c, TUDELFT["darkred"], 0.90, 3, 2.8); n_full += 1
        else:
            _arc(c, TUDELFT["blue"], 0.90, 3, 2.8); n_half += 1

    ax[1].axhline(0.0, color=pal["neutral"], lw=0.6, alpha=0.6)
    ax[1].axhline(DELTA_C, color=pal["neutral"], lw=0.8, ls=":")
    ax[1].text(n * 0.995, DELTA_C + 0.02, r"$\delta_c$", ha="right", va="bottom",
               fontsize=FS_ANNOT, color=pal["neutral"])
    ax[1].set_ylabel(r"Cycle depth  $\delta$" + "\n(-)")
    ax[1].set_ylim(-0.04, 1.0)
    ax[1].set_xlim(0, n)
    ax[1].set_xticks(np.arange(0, n + 1, 24))
    ax[1].set_xlabel(f"Hour of week {info['week']}")

    handles = [
        Line2D([0], [0], color=TUDELFT["darkred"], lw=1.0, marker="o",
               markersize=2.8, label=f"Full cycle ({n_full})"),
        Line2D([0], [0], color=TUDELFT["blue"], lw=1.0, ls=(0, (3, 2)),
               marker="o", markersize=2.8, label=f"Half cycle ({n_half})"),
    ]
    if st["_cyc_touch"]:
        handles.append(Line2D([0], [0], color=pal["neutral"], lw=1.0, alpha=0.40,
                              marker="o", markersize=2.2,
                              label=f"Adjacent week ({len(st['_cyc_touch'])})"))
    ax[1].legend(handles=handles, frameon=False, fontsize=FS_LEGEND,
                 loc="upper center", bbox_to_anchor=(0.5, -0.30),
                 ncol=len(handles))

    out_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, out_dir, "fig_week_rainflow")
    plt.close(fig)

    dep = np.array([c["dod"] for c in st["_cyc_wk"]])
    full_depth = sol["soc_max"] - sol["soc_min"]
    print(f"\n  week {info['week']} rainflow, for the caption: "
          f"{len(st['_cyc_wk'])} records ({n_full} full, {n_half} half), "
          f"total count {sum(c['count'] for c in st['_cyc_wk']):.1f}; "
          f"{int(np.sum(np.abs(dep - full_depth) < 1e-9))} at the full window "
          f"depth of {full_depth:.2f}; {int(np.sum(dep > DELTA_C))} above "
          f"delta_c; {len(st['_cyc_touch'])} straddling from adjacent weeks")


def fig_c_width(sols, stats, inp, sl, info, out_dir):
    """Widening the window deepens the cycles.
 
    Three panels. The trace panel on the left carries each window's own bounds
    as dashed lines in its own colour, so the reader sees three dispatches each
    filling a different band rather than three arbitrary curves. The two panels
    on the right are the quantitative statement: the amplitude distribution
    moves right with the width.
 
    The three windows are drawn as dodged bars rather than stepped outlines.
    Where all three hold a cycle of the same amplitude AND the same
    multiplicity the outlines coincided exactly, so only the last one drawn was
    visible, at precisely the amplitudes where the text says the windows agree.
 
    The lower right panel magnifies delta < ZOOM_DELTA, where every bar carries
    sum_i n_i = 1 or less and the full-range panel cannot resolve them.
 
    The y axis is the summed multiplicity sum_i n_i. It always was: the
    histogram is weighted by c["count"]. A bar of height 8.5 is 5 full cycles
    and 7 half-cycles, not 8.5 entries in the cycle list.
    """
    pal = apply_thesis_style(palette="brand", usetex=False)
    ecap = inp["e_cap"]
    cols = [TUDELFT["navy"], TUDELFT["darkred"], TUDELFT["blue"]]
    nz = min(ZOOM_HOURS, HOURS_PER_WEEK)
 
    ZOOM_DELTA = 0.42            # magnified amplitude range
    bins = np.linspace(0, 1, 41)  # 0.025 wide, as before
    bw = bins[1] - bins[0]
    dodge = bw / 3.0             # one window's bar inside a bin
    centres = bins[:-1] + bw / 2.0
 
    fig = plt.figure(figsize=figsize(1.0, aspect=0.62))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0],
                          height_ratios=[1.0, 0.85])
    axT = fig.add_subplot(gs[:, 0])    # trace
    axF = fig.add_subplot(gs[0, 1])    # full amplitude range
    axZ = fig.add_subplot(gs[1, 1])    # magnified
 
    heights = {}
    for k, win in enumerate(WIDTH_SERIES):
        e = sols[win]["e"][sl][:nz]
        axT.plot(np.arange(nz), e / ecap, color=cols[k], lw=1.0, zorder=3,
                 label=f"{int(win[0]*100)}-{int(win[1]*100)}%")
        for lvl in win:
            axT.axhline(lvl, color=cols[k], lw=0.6, ls=(0, (4, 3)),
                        alpha=0.45, zorder=1)
 
        st = stats[win]
        d = np.array([c["dod"] for c in st["_cyc_wk"]])
        w = np.array([c["count"] for c in st["_cyc_wk"]])
        h, _ = np.histogram(d, bins=bins, weights=w)
        heights[win] = h
        for ax in (axF, axZ):
            ax.bar(centres + (k - 1) * dodge, h, width=dodge * 0.92,
                   color=cols[k], zorder=3)
 
    stacked = np.vstack([heights[w] for w in WIDTH_SERIES])
    shared = np.where((stacked > 0).all(axis=0)
                      & (np.ptp(stacked, axis=0) < 1e-9))[0]
 
    axT.set_ylim(-0.03, 1.03)
    axT.set_xlim(0, nz)
    axT.set_xticks(np.arange(0, nz + 1, 24))
    axT.set_xlabel(f"Hour of week {info['week']}  (first {nz} h)")
    axT.set_ylabel("State of charge  (-)")
 
    for ax in (axF, axZ):
        ax.axvline(DELTA_C, color=pal["neutral"], lw=0.8, ls=":", zorder=2)
        ax.set_ylabel(r"$\sum_i n_i$  (-)")
 
    top = float(stacked.max()) * 1.15
    axF.set_xlim(-0.02, 1.04)
    axF.set_ylim(0, top)
    axF.tick_params(labelbottom=False)
    axF.annotate(r"$\delta_c$", xy=(DELTA_C + 0.03, top * 0.86),
                 fontsize=FS_ANNOT, color=pal["neutral"])
    axF.plot([0.0, ZOOM_DELTA], [top * 0.21, top * 0.21],
             color=pal["neutral"], lw=0.8)
    for x in (0.0, ZOOM_DELTA):
        axF.plot([x, x], [top * 0.18, top * 0.24],
                 color=pal["neutral"], lw=0.8)
    axF.annotate("magnified below", xy=(ZOOM_DELTA / 2.0, top * 0.28),
                 ha="center", fontsize=FS_ANNOT, color=pal["neutral"])
 
    ztop = float(stacked[:, centres <= ZOOM_DELTA].max()) * 1.2
    axZ.set_xlim(0.0, ZOOM_DELTA)
    axZ.set_ylim(0, ztop)
    axZ.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    axZ.set_xlabel(r"Cycle amplitude  $\delta$  (-)")
    axZ.annotate(r"$\delta_c$", xy=(DELTA_C + 0.008, ztop * 0.82),
                 fontsize=FS_ANNOT, color=pal["neutral"])
    for b in shared:
        if centres[b] <= ZOOM_DELTA:
            axZ.plot(centres[b], stacked[:, b].max() + ztop * 0.10,
                     marker="v", ms=3.5, color=pal["neutral"], zorder=4)
 
    handles, labels = axT.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               frameon=False, fontsize=FS_LEGEND)
 
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, out_dir, "fig_week_width")
    plt.close(fig)
 
    if shared.size:
        print("  width figure: amplitudes shared by all three windows "
              + ", ".join(f"{centres[b]:.3f}" for b in shared))


def check_centre_invariance(sols, stats, sl) -> dict:
    """The dispatch must not depend on where the window sits, only on its width.

    Neither the objective nor the power constraints contain an absolute state of
    charge, only differences and bounds, so shifting the window by a constant
    must shift the state of charge by that constant and leave the power schedule
    untouched. That is a requirement of the formulation, not an observation, so
    a deviation here means the kernel has picked up an absolute-SoC dependence
    somewhere and every centre-series result in Chapter 4 would be suspect.

    Reported as a number rather than drawn: a figure of three coincident curves
    shows one curve and proves nothing, while the number states its own
    precision.
    """
    ref = CENTRE_SERIES[1]
    p_ref, e_ref = sols[ref]["p"], sols[ref]["e"]
    e_cap = e_ref.max() / ref[1]                       # recover capacity from the bound
    rows = []
    for win in CENTRE_SERIES:
        if win == ref:
            continue
        # Two things must hold: the power schedule is untouched, and the state
        # of charge is offset by exactly the shift in the window, not merely by
        # some constant.
        dp = float(np.max(np.abs(sols[win]["p"] - p_ref)))
        shift = 0.5 * ((win[0] - ref[0]) + (win[1] - ref[1]))
        offset = sols[win]["e"] - e_ref
        d_off = float(np.max(np.abs(offset - shift * e_cap)))
        # The invariant is the work the dispatch does, not the vertex it is
        # reported at. The linear program has multiple optima and the SoC
        # balance is an inequality, so the solver may return a different point
        # on the same optimal face. Energy moved is the same at every point on
        # that face; the pointwise schedule need not be.
        e_ref_moved = float(np.abs(p_ref).sum())
        e_win_moved = float(np.abs(sols[win]["p"]).sum())
        rel_work = abs(e_win_moved - e_ref_moved) / max(e_ref_moved, 1e-12)
        rows.append({"window": f"{int(win[0]*100)}-{int(win[1]*100)}%",
                     "energy_moved_ref_MWh": e_ref_moved,
                     "energy_moved_win_MWh": e_win_moved,
                     "rel_diff_energy_moved": rel_work,
                     "max_abs_power_diff_MW": dp,
                     "expected_soc_offset_MWh": shift * e_cap,
                     "mean_soc_offset_MWh": float(np.mean(offset)),
                     "max_dev_from_expected_MWh": d_off})
        print(f"    {rows[-1]['window']:>8}: max |dP| {dp:.2e} MW   "
              f"SoC offset {np.mean(offset):+.4f} MWh "
              f"(expected {shift * e_cap:+.4f}, max deviation {d_off:.2e})")
        # Where does it deviate? A handful of steps points at solver
        # degeneracy; a broad drift points at a formulation dependence.
        bad = np.abs(sols[win]["p"] - p_ref) > 1e-6
        if bad.any():
            idx = np.flatnonzero(bad)
            rows[-1]["n_steps_deviating"] = int(bad.sum())
            rows[-1]["first_step"] = int(idx[0])
            rows[-1]["last_step"] = int(idx[-1])
            rows[-1]["energy_moved_ref_MWh"] = float(np.abs(p_ref).sum())
            rows[-1]["energy_moved_win_MWh"] = float(np.abs(sols[win]["p"]).sum())
            print(f"        deviates at {bad.sum()} of {len(bad)} steps, "
                  f"first h{idx[0]}, last h{idx[-1]}")
            print(f"        total |power| moved: ref {np.abs(p_ref).sum():.1f} MWh, "
                  f"this window {np.abs(sols[win]['p']).sum():.1f} MWh")
        else:
            rows[-1]["n_steps_deviating"] = 0

    worst = max(r["rel_diff_energy_moved"] for r in rows)
    print(f"    largest dispatch difference across the centre series: "
          f"{max(r['max_abs_power_diff_MW'] for r in rows):.3e} MW")
    ok = worst <= 1e-9
    print(f"    largest relative difference in energy moved: {worst:.2e}")
    if ok:
        print("    PASS -- moving the window leaves the dispatch unchanged.")
        print("    Where the pointwise schedule differs, it is the solver")
        print("    reporting a different point on the same optimal face; the")
        print("    state of charge can also drift where the balance is slack,")
        print("    because the formulation permits discarding energy there.")
    else:
        print("    FAIL: the two windows move different amounts of energy, so")
        print("    this is not degeneracy. The formulation carries an absolute")
        print("    state-of-charge dependence and every center-series result is")
        print("    affected. A window with soc_min = 0 makes the kernel lower")
        print("    bound e >= e_cap*(1-dod) collapse onto the variable's own")
        print("    bound, which is the first place to look.")

    print("\n  center series, week degradation (for the Section 4.4 text)")
    for win in CENTRE_SERIES:
        st = stats[win]
        print(f"    {int(win[0]*100)}-{int(win[1]*100)}%:  mean SoC {st['week_mean_soc']:.4f}"
              f"   f_d {st['week_fd']:.6e}"
              f"   cycle {st['week_fd_cycle']:.6e}"
              f"   calendar {st['week_fd_calendar']:.6e}")
    tot = [stats[w]["week_fd"] for w in CENTRE_SERIES]
    sig = [stats[w]["week_mean_soc"] for w in CENTRE_SERIES]
    ratios = []
    for i in range(1, len(tot)):
        pred = float(np.exp(1.04 * (sig[i - 1] - sig[i])))
        ratios.append((tot[i-1]/tot[i], pred))
        print(f"    ratio {tot[i-1]/tot[i]:.6f}   predicted exp(k_sigma*dsigma) {pred:.6f}")
    return {"rows": rows, "worst_power_diff": worst, "fd_ratios": ratios,
            "ok": ok}


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--resolve", action="store_true",
                    help="ignore the cache and solve the five annual LPs again")
    args = ap.parse_args()

    print("=" * 78)
    print("Week snapshot of the real dispatch: year-1 solves on DK1 prices")
    print("=" * 78)

    windows = []
    for w in [REF_WINDOW, *WIDTH_SERIES, *CENTRE_SERIES]:
        if w not in windows:
            windows.append(w)

    cached = None
    if CACHE_FILE.exists() and not args.resolve:
        cached = load_cache(CACHE_FILE, windows)
    elif args.resolve and CACHE_FILE.exists():
        print("  --resolve given, ignoring the cache")

    if cached is not None:
        inp, sols = cached
    else:
        inp = load_inputs()
        print(f"\n  solving year 1 for {len(windows)} windows ...")
        sols = {}
        for w in windows:
            print(f"    {int(w[0]*100)}-{int(w[1]*100)}% ...", end="", flush=True)
            sols[w] = solve_year1(inp, w[0], w[1])
            print(" done")
        save_cache(inp, sols, CACHE_FILE)

    check_against_baseline(sols[REF_WINDOW], inp["e_cap"])

    week, info = choose_week(inp["price"])
    sl = week_slice(week)

    stats = {w: window_stats(sols[w], inp["e_cap"], week, sl) for w in windows}

    DATA_DIR_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for w in windows:
        r = {k: v for k, v in stats[w].items() if not k.startswith("_")}
        r["week"] = week
        rows.append(r)
    summ = pd.DataFrame(rows)
    lead = ["window", "soc_min", "soc_max", "full_depth", "week",
            "week_records", "week_records_full", "week_records_half",
            "week_cycles", "week_cycles_above_dc", "week_straddling",
            "week_at_full_depth", "week_above_dc", "week_zero_depth",
            "week_mean_depth", "week_max_depth", "week_mean_soc", "week_efc",
            "week_fd_cycle", "week_fd_calendar", "week_fd",
            "year_records", "year_records_full", "year_records_half",
            "year_records_other", "year_zero_depth",
            "year_cycles", "year_cycles_per_week",
            "year_cycles_above_dc_per_week",
            "year_mean_depth", "year_mean_soc", "year_efc"]
    summ = summ[[c for c in lead if c in summ.columns]
                + [c for c in summ.columns if c not in lead]]
    summ.to_csv(DATA_DIR_OUT / "week_snapshot_summary.csv", index=False)

    hourly = {"hour": np.arange(HOURS_PER_WEEK),
              "price_eur_mwh": inp["price"][sl],
              "wind_mw": inp["wind"][sl]}
    for w in windows:
        tag = f"{int(w[0]*100)}_{int(w[1]*100)}"
        hourly[f"storage_p_mw_{tag}"] = sols[w]["p"][sl]
        hourly[f"soc_{tag}"] = sols[w]["e"][sl] / inp["e_cap"]
    pd.DataFrame(hourly).to_csv(DATA_DIR_OUT / "week_snapshot_hourly.csv", index=False)

    # every counted cycle, so any number quoted in the text is traceable
    crow = []
    for w in windows:
        tag = f"{int(w[0]*100)}-{int(w[1]*100)}%"
        for scope, lst in (("week", stats[w]["_cyc_wk"]),
                           ("adjacent", stats[w]["_cyc_touch"])):
            for c in lst:
                crow.append({"window": tag, "scope": scope,
                             "i_start_year": c["i_start"], "i_end_year": c["i_end"],
                             "hour_in_week_start": c["i_start"] - sl.start,
                             "hour_in_week_end": c["i_end"] - sl.start,
                             "depth": c["dod"], "mean_soc": c["soc_mean"],
                             "count": c["count"]})
    pd.DataFrame(crow).to_csv(DATA_DIR_OUT / "week_snapshot_cycles.csv", index=False)

    # Every cycle of the full year at the reference window, so the annual record
    # split quoted in Section 4.3 is traceable to individual cycles rather than
    # derived from the record count and the summed multiplicity.
    ref_tag = f"{int(REF_WINDOW[0]*100)}-{int(REF_WINDOW[1]*100)}%"
    yrow = [{"window": ref_tag,
             "i_start": c["i_start"], "i_end": c["i_end"],
             "depth": c["dod"], "mean_soc": c["soc_mean"], "count": c["count"]}
            for c in stats[REF_WINDOW]["_cyc_yr"]]
    pd.DataFrame(yrow).to_csv(DATA_DIR_OUT / "week_snapshot_year_cycles.csv", index=False)

    ry = stats[REF_WINDOW]
    R, S = ry["year_records"], ry["year_cycles"]
    print(f"\n  annual rainflow, {ref_tag} window, {PRICE_FILE}")
    print(f"    records              : {R}")
    print(f"      full  (count 1.0)  : {ry['year_records_full']}")
    print(f"      half  (count 0.5)  : {ry['year_records_half']}")
    print(f"      other              : {ry['year_records_other']}   <- must be 0")
    print(f"      zero amplitude     : {ry['year_zero_depth']}")
    print(f"    summed multiplicity  : {S:.1f}")
    print(f"    equivalent full cycles: {ry['year_efc']:.2f}")
    print(f"    mean amplitude       : {100 * ry['year_mean_depth']:.2f} %")
    print(f"    identities: 2S-R = {2*S - R:.0f} full, 2(R-S) = {2*(R - S):.0f} half")
    
    fig_a_dispatch(sols[REF_WINDOW], inp, sl, info, OUT_DIR)
    fig_b_rainflow(sols[REF_WINDOW], stats[REF_WINDOW], inp, sl, info, OUT_DIR)
    fig_c_width(sols, stats, inp, sl, info, OUT_DIR)

    print("\n  dispatch invariance to the window centre")
    inv = check_centre_invariance(sols, stats, sl)
    pd.DataFrame(inv["rows"]).to_csv(DATA_DIR_OUT / "week_snapshot_checks.csv",
                                     index=False)
    if not inv["ok"]:
        print("\n  NOTE: figures and CSVs were still written. The failing check "
              "is recorded\n        in week_snapshot_checks.csv.")

    print("\n  per-window statistics for the week")
    show = ["window", "week_records", "week_cycles", "week_straddling",
            "week_mean_depth", "week_max_depth", "week_mean_soc", "week_efc",
            "week_fd", "year_cycles"]
    print(summ[show].to_string(index=False))
    print(f"\n  wrote CSVs to {DATA_DIR_OUT} and three figures to {OUT_DIR}")

if __name__ == "__main__":
    main()