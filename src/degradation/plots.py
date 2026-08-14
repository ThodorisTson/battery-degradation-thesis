"""
WP2 Battery Degradation,  Plots, Reports, and DST Validation
=============================================================

All visualisation and reporting for the Xu et al. (2016) degradation model.
Split from xu.py to keep the physical model file concise.

Imports everything it needs from degradation_xu,  nothing here does any
degradation computation, it only presents results.

Public API
----------
    print_degradation_report(degradation, period_days, enabled) -> None
    plot_degradation_analysis(degradation, storage_e, time_vec,
                              save_path, show, verbose,
                              lifetime_years, eol_thresholds, p) -> List[Figure]
    validate_xu_dst(save_path, show, T_cell_C) -> Figure
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from degradation.xu import (
    XuModelParams,
    XU_LMO,
    s_temp,
    fc_cycle,
    xu_capacity_curve,
)


def print_degradation_report(degradation: Dict,
                              period_days: float,
                              enabled: bool = True) -> None:
    """Print a formatted degradation report based on Xu model results.

    Args:
        degradation: Output of analyze_degradation().
        period_days: Simulation duration [days].
        enabled:     If False, no output is printed.
    """
    if not enabled:
        return

    print("\n" + "=" * 72)
    print("BATTERY DEGRADATION,  Xu et al. (2016) Semi-Empirical Model")
    print("=" * 72)

    meta   = degradation.get("meta", {})
    stats  = degradation.get("xu_cycle_stats", {})
    cycles = float(degradation["total_cycles"])

    print(f"\nSimulation period      : {period_days:.1f} days")
    print(f"Temperature assumption : {meta.get('T_cell_C', 25):.0f}°C  "
          f"(S_T = {s_temp(meta.get('T_cell_C', 25.0)):.4f})")
    print(f"Mean SoC               : {meta.get('sigma_mean', 0)*100:.1f}%")

    print("\nCycle metrics:")
    print(f"  Equivalent full cycles (EFC) : {cycles:.2f}")
    print(f"  EFC per day                  : {cycles/max(period_days, 1):.3f}")
    print(f"  EFC per year (extrapolated)  : {cycles/max(period_days, 1)*365:.1f}")
    if stats:
        print(f"  Rainflow cycles (weighted)   : {stats.get('n_rainflow_cycles', 0):.1f}")
        print(f"  Mean rainflow DoD            : {stats.get('mean_dod', 0)*100:.1f}%")
        print(f"  Mean rainflow SoC            : {stats.get('mean_soc', 0)*100:.1f}%")

    print("\nXu degradation (fd decomposition):")
    fd = float(degradation["fd"])
    print(f"  fd_total    : {fd:.6f}")
    print(f"  fd_cycle    : {degradation['fd_cycle']:.6f}  "
          f"({100*degradation['fd_cycle']/max(fd, 1e-30):.1f}%)")
    print(f"  fd_calendar : {degradation['fd_calendar']:.6f}  "
          f"({100*degradation['fd_calendar']/max(fd, 1e-30):.1f}%)")

    print("\nCapacity state:")
    print(f"  State of Health (SoH)   : {degradation['soh']:.3f}%")
    print(f"  Capacity retention      : {degradation['capacity_retention']:.6f}")
    print(f"  Capacity fade           : {degradation['capacity_fade_percent']:.4f}%")
    print(f"  Degraded energy cap     : {degradation['e_cap_degraded']:.3f} MWh")
    print(f"  Degraded power cap      : {degradation['p_cap_degraded']:.3f} MW")

    eol_years = degradation.get("eol_years", {})
    if eol_years:
        print("\nProjected End-of-Life (same cycling pattern extrapolated):")
        labels = {0.80: "IEC/EV convention",
                  0.70: "warranty typical",
                  0.60: "grid-storage operational"}
        for thr in sorted(eol_years.keys(), reverse=True):
            yr = eol_years[thr]
            note = labels.get(thr, "")
            yr_str = f"{yr:.1f} yr" if yr is not None else "> 200 yr"
            print(f"  SoH \u2265 {thr*100:.0f}%  \u2192  EoL at {yr_str}"
                  + (f"  [{note}]" if note else ""))

    print("\n" + "=" * 72)


# =============================================================================
# Capacity fade curve helper
# =============================================================================

def plot_degradation_analysis(
    degradation: Dict,
    storage_e: List[float],
    time_vec: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
    verbose: bool = False,
    lifetime_years: float = 15.0,
    eol_thresholds: List[float] = None,
    p: XuModelParams = XU_LMO,
) -> List["plt.Figure"]:
    """Produce 4 clean degradation figures, saved individually.

    Figures produced (saved with numbered suffixes before the extension):

        <stem>_1_soc.<ext>           Battery State of Charge time series
        <stem>_2_rainflow_dod.<ext>  Rainflow cycle DoD distribution
        <stem>_3_rainflow_soc.<ext>  Rainflow cycle mean-SoC distribution
        <stem>_4_soh_projection.<ext> SoH projection over design lifetime

    The same 4-plot structure is used for both the Xu and Shi branches.
    Text, labels, and legend entries adapt automatically based on which
    model produced the degradation dict (detected via meta["model"]).

    Args:
        degradation:    Output dict from analyze_degradation() or
                        analyze_degradation_shi().
        storage_e:      SoC time-series [MWh].
        time_vec:       Time axis [days].
        save_path:      Base path for output images.  Suffix and index are
                        inserted automatically before the extension.
                        If None, figures are not saved.
        show:           If True, calls plt.show() on each figure.
        verbose:        If True, prints each saved path.
        lifetime_years: Design lifetime for the SoH projection panel.
        eol_thresholds: List of SoH thresholds for EoL lines.
        p:              Xu model parameters (used for the SEI capacity curve).

    Returns:
        List of 4 matplotlib Figure objects.
    """
    from pathlib import Path as _Path

    if eol_thresholds is None:
        eol_thresholds = list(degradation.get("meta", {}).get(
            "eol_thresholds", [0.80, 0.60]))

    # ------------------------------------------------------------------
    # Shared model metadata
    # ------------------------------------------------------------------
    meta       = degradation.get("meta", {})
    fd_now     = float(degradation["fd"])
    soh_now    = float(degradation["soh"])
    T_C        = float(meta.get("T_cell_C", 25.0))
    t_sim_h    = float(meta.get("t_total_hours", 8760.0))
    pct_cyc    = 100 * degradation["fd_cycle"]    / max(fd_now, 1e-30)
    pct_cal    = 100 * degradation["fd_calendar"] / max(fd_now, 1e-30)
    cycles     = degradation.get("cycle_depth_distribution", [])
    eol_dict   = degradation.get("eol_years", {})

    # Auto-detect model branch
    _is_shi = "Shi" in meta.get("model", "")

    # ------------------------------------------------------------------
    # Suptitle,  one line of context stamped on every figure
    # ------------------------------------------------------------------
    if _is_shi:
        sup = (f"Shi (2018) + Xu SEI (Option B)  |  T = {T_C:.0f}\u00b0C  |  "
               f"fd = {fd_now:.4f}  |  SoH = {soh_now:.2f}%  |  "
               f"cycle-only  (fd_calendar \u2261 0)")
    else:
        sup = (f"Xu (2016) Degradation  |  T = {T_C:.0f}\u00b0C  |  "
               f"fd = {fd_now:.4f}  |  SoH = {soh_now:.2f}%  |  "
               f"Cycle {pct_cyc:.0f}% / Calendar {pct_cal:.0f}%")

    # ------------------------------------------------------------------
    # Text variants,  only these change between Xu and Shi
    # ------------------------------------------------------------------
    if _is_shi:
        _k3 = float(meta.get("k3", 0.0))
        _k4 = float(meta.get("k4", 1.0))
        _dod_subtitle  = (f"Each bar = weighted rainflow cycle count in that DoD bin.  "
                          f"Deeper cycles cost more via \u03a6(\u03b4) = k3\u00b7\u03b4^k4")
        _soc_subtitle  = ("Each bar = weighted rainflow cycle count at that mean SoC.  "
                          "Higher SoC amplifies damage via S\u03c3 = exp(k\u03c3\u00b7(\u03c3\u2212\u03c3_ref))")
        _soh_fd_label  = f"Projected SoH  (fd_shi/yr = {fd_now/max(t_sim_h/8760.0,1e-6):.5f})"
        _fd_note       = (f"fd_shi = {fd_now:.5f}  (cycle-only)\n"
                          f"k3 = {_k3:.4e}   k4 = {_k4:.4f}")
    else:
        _dod_subtitle  = ("Each bar = weighted rainflow cycle count in that DoD bin.  "
                          "Deeper cycles cost more via Xu S_\u03b4 stress factor")
        _soc_subtitle  = ("Each bar = weighted rainflow cycle count at that mean SoC.  "
                          "Higher SoC amplifies damage via Xu S\u03c3 stress factor")
        _soh_fd_label  = f"Projected SoH  (fd/yr = {fd_now/max(t_sim_h/8760.0,1e-6):.5f})"
        _fd_note       = (f"fd_cycle   = {degradation['fd_cycle']:.5f}  ({pct_cyc:.0f}%)\n"
                          f"fd_calendar = {degradation['fd_calendar']:.5f}  ({pct_cal:.0f}%)\n"
                          f"fd_total    = {fd_now:.5f}")

    # EoL threshold colours
    _thr_colours = {0.80: "#e74c3c", 0.70: "#e67e22", 0.60: "#8e44ad"}
    def _thr_colour(thr: float) -> str:
        return _thr_colours.get(thr, "#555555")

    def _save(fig: "plt.Figure", suffix: str) -> None:
        if save_path is not None:
            sp  = _Path(save_path)
            out = sp.with_name(sp.stem + suffix + sp.suffix)
            fig.savefig(out, dpi=200, bbox_inches="tight")
            if verbose:
                print(f"  \u2713 Saved: {out.name}")
        if show:
            plt.show()

    figs: List["plt.Figure"] = []

    # ------------------------------------------------------------------
    # Figure 1,  Battery State of Charge
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_vec, storage_e, linewidth=0.6, color="steelblue")
    ax.axhline(
        degradation["e_cap_degraded"],
        linestyle="--", color="red", linewidth=1.2,
        label=f"Degraded capacity after {t_sim_h/8760:.1f} yr  (SoH = {soh_now:.2f}%)"
    )
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("State of Charge [MWh]")
    ax.set_title("Battery State of Charge") # Note for reader: red dashed line = degraded capacity at end of simulation year (Shi cycle-only SoH)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "_1_soc")
    figs.append(fig)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2,  Rainflow Cycle DoD Distribution
    # ------------------------------------------------------------------
    # Note: deeper cycles cost more via Φ(δ) = k3·δ^k4 (Shi) or S_δ stress factor (Xu)
    # Note: distribution is bimodal — shallow arbitrage cycles (0–5%) + deep full swings (75–80%)
    fig, ax = plt.subplots(figsize=(9, 5))
    if cycles:
        dods     = np.array([c["dod"]   for c in cycles]) * 100
        cnts     = np.array([c["count"] for c in cycles])
        ax.hist(dods, bins=20, weights=cnts,
                color="steelblue", alpha=0.75, edgecolor="white", linewidth=0.5)
        mean_dod = float(np.average(dods, weights=cnts))
        ax.axvline(mean_dod, linestyle="--", color="red", linewidth=1.5,
                   label=f"Weighted mean DoD = {mean_dod:.1f}%")
        ax.legend(fontsize=9)
    ax.set_xlabel("Cycle depth of discharge [% of nominal capacity]")
    ax.set_ylabel("Weighted cycle count")
    ax.set_title("Rainflow Cycle: Depth of Discharge Distribution")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, "_2_rainflow_dod")
    figs.append(fig)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3,  Rainflow Cycle Mean SoC Distribution
    # ------------------------------------------------------------------
    # Note: higher SoC amplifies damage via S_σ stress factor
    # Note: mean SoC ≈ 50% → S_σ ≈ 1.0 → validates Option B (k3_eff = k3, no correction needed)
    fig, ax = plt.subplots(figsize=(9, 5))
    if cycles:
        socs     = np.array([c["soc_mean"] for c in cycles]) * 100
        cnts_s   = np.array([c["count"]    for c in cycles])
        ax.hist(socs, bins=20, weights=cnts_s,
                color="steelblue", alpha=0.75, edgecolor="white", linewidth=0.5)
        mean_soc = float(np.average(socs, weights=cnts_s))
        ax.axvline(mean_soc, linestyle="--", color="red", linewidth=1.5,
                   label=f"Weighted mean SoC = {mean_soc:.1f}%")
        ax.legend(fontsize=9)
    ax.set_xlabel("Mean SoC during the cycle [% of nominal capacity]")
    ax.set_ylabel("Weighted cycle count")
    ax.set_title("Rainflow Cycle:  Mean SoC Distribution")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, "_3_rainflow_soc")
    figs.append(fig)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4,  SoH Projection over Design Lifetime
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    fd_per_yr = fd_now / max(t_sim_h / 8760.0, 1e-6)
    years     = np.linspace(0, lifetime_years, 400)
    soh_proj  = xu_capacity_curve(years * fd_per_yr, p)

    ax.plot(years, soh_proj, linewidth=2.0, color="steelblue", label=_soh_fd_label)

    for thr in sorted(eol_thresholds, reverse=True):
        col  = _thr_colour(thr)
        note = ("IEC/EV" if thr == 0.80 else
                "grid-storage" if thr == 0.60 else f"{thr*100:.0f}%")
        ax.axhline(thr * 100, linestyle="--", color=col, alpha=0.7,
                   label=f"EoL {thr*100:.0f}%  [{note}]")
        yr_eol = eol_dict.get(thr)
        if yr_eol is not None and yr_eol <= lifetime_years:
            ax.axvline(yr_eol, linestyle=":", color=col, alpha=0.5)
            ha = "left" if yr_eol < lifetime_years * 0.75 else "right"
            ax.annotate(
                f"{thr*100:.0f}% EoL\n\u2248 {yr_eol:.1f} yr",
                xy=(yr_eol + (0.15 if ha == "left" else -0.15), thr * 100 + 0.6),
                fontsize=8.5, color=col, ha=ha, fontweight="bold"
            )

    sim_yr = t_sim_h / 8760.0
    ax.scatter([sim_yr], [soh_now], s=120, zorder=5, color="red",
               label=f"End of simulation  ({sim_yr:.1f} yr,  SoH = {soh_now:.2f}%)")

    # fd breakdown note
    ax.annotate(
        f"fd/yr = {fd_now/max(t_sim_h/8760.0,1e-6):.5f}", # Note: fd shown here is Shi cycle-only. Combined fd (incl. Xu calendar) is in the text report.
        xy=(0.02, 0.06), xycoords="axes fraction", fontsize=8, # Note: difference between cycle-only SoH (92.29%) and combined SoH (92.276%) is ~0.01%
        va="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85)
    )

    ax.set_xlim(0, lifetime_years)
    min_thr = min(eol_thresholds) * 100 if eol_thresholds else 60.0
    ax.set_ylim(max(float(soh_proj.min()) - 3, min_thr - 8), 102)
    ax.set_xlabel("Battery age [years]")
    ax.set_ylabel("State of Health [%]")
    ax.set_title(f"SoH Projection — {lifetime_years:.0f}-Year Design Lifetime") # Note: assumes same operating pattern repeats each year (stationary approximation)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "_4_soh_projection")
    figs.append(fig)
    plt.close(fig)

    return figs
