"""
WP2 Battery Degradation, console reporting
==========================================

Formatted console output for the Xu et al. (2016) degradation model. Nothing
here does any degradation computation; it only presents results produced by
analyze_degradation() in xu.py.

This module was previously named plots.py and also held
plot_degradation_analysis, which produced four diagnostic figures from a single
run. Those figures were development diagnostics and nothing in the thesis or
the repository read them, so the function was removed together with the inline
figure code in scripts/run_baseline.py. Every thesis figure is produced by a
script under figures/ or verification/ that reads the frozen results in
results/. The module is named for what it does now, and no longer imports
matplotlib.

Public API
----------
    print_degradation_report(degradation, period_days, enabled) -> None
"""

from __future__ import annotations

from typing import Dict

from degradation.xu import s_temp


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