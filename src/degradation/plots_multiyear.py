"""
plots_multiyear.py
==============================
Standalone plotting module for multi-year battery degradation results.
 
Contains all visualisation functions that consume the `multiyear` result dict produced by `_run_multiyear()` in the run script. Separated from the run script so plots can be edited, iterated, 
or re-run without touching any physics or LP logic.
 
Functions
---------
    _split_battery_segments   — utility: split SoH trajectory at replacements
    plot_gradient_analysis    — 2×2 inner-loop gradient figure
    plot_subgradient_timeseries — 2-panel subgradient vs SoC time series
    plot_multiyear_trajectory — 2×2 multi-year degradation summary figure
 
All three plot functions accept `plots_dir`, `n_years`, `eol_replacement`, and `show` as explicit arguments so this module has no dependency on the CONFIG block of the run script.
"""
 
from __future__ import annotations
 
from pathlib import Path
from typing import Dict, List
 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
 
 
# =============================================================================
# Utility
# =============================================================================

def _split_battery_segments(
    years_full: List[int],
    soh_full: List[float],
    replacement_years: List[int],
) -> List[tuple]:
    """Split SoH trajectory into one segment per battery generation.
    Fully data-driven : works for 0, 1, or N replacements at any year.
    Each replacement resets the segment and anchors a fresh 100% start point.
    """
    repl_set = set(replacement_years)
    segments = []
    cur_x, cur_y = [], []

    for yr, soh in zip(years_full, soh_full):
        cur_x.append(yr)
        cur_y.append(soh)
        if yr in repl_set:
            segments.append((list(cur_x), list(cur_y)))
            cur_x = [yr]      # same x-anchor at replacement year
            cur_y = [100.0]   # fresh battery starts at 100%

    if cur_x:
        segments.append((cur_x, cur_y))

    return segments

# =============================================================================
# Gradient analysis : 2×2
# =============================================================================

def plot_gradient_analysis(
    multiyear: Dict,
    run_label: str,
    plots_dir: Path,
    n_years: int,
    show: bool = False,
) -> None:
    """2×2 inner-loop gradient analysis figure.
 
    Top-left    dDeg/dDoD per year, regime-coloured (red = high, blue = low)
    Top-right   Alignment factor vs dDeg/dDoD scatter (r value annotated)
    Bottom-left Mean |subgradient| bars + mean DoD right-axis overlay
    Bottom-right Scalar components vs gradient, all normalised to mean=1
    """
    # ── Regime colours ────────────────────────────────────────────────────
    C_HIGH = "#e74c3c"
    C_LOW  = "#4C72B0"
    ALIGN_THRESHOLD = 3000.0

    ann_grad = multiyear.get("annual_gradient", [])
    traj     = multiyear["soh_trajectory"]

    years    = [r[0] for r in traj]
    soh_pct  = [r[1] for r in traj]
    years    = [r[0] for r in traj]
    soh_pct  = [r[1] for r in traj]

    # Unpack gradient tuple : None-safe
    # Tuple layout:     
    #  [5] cycle_coverage [6] subgrad_combined array
    dDeg_dDoD        = [g[0] if g else None for g in ann_grad] # [0] dDeg/dDoD
    mean_abs_subgrad = [g[1] if g else None for g in ann_grad] # [1] mean_abs_subgrad
    mean_dod         = [g[2] if g else None for g in ann_grad] # [2] mean_dod
    mean_abs_dual    = [g[3] if g else None for g in ann_grad] # [3] mean_abs_dual
    e_cap_eff_yr     = [g[4] if g else None for g in ann_grad] # [4] e_cap_eff

    # Alignment factor = dDeg/dDoD / (e_cap × |subgrad| × |dual|)
    alignment = []
    for dd, mas, mad, ec in zip(dDeg_dDoD, mean_abs_subgrad, mean_abs_dual, e_cap_eff_yr):
        if all(v is not None for v in (dd, mas, mad, ec)):
            denom = ec * mas * mad
            alignment.append(dd / denom if abs(denom) > 1e-9 else None)
        else:
            alignment.append(None)

    regime_colors = [
        C_HIGH if (al is not None and al >= ALIGN_THRESHOLD) else C_LOW
        for al in alignment
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Inner-Loop Gradient Analysis  |  {n_years} yr",
                 fontsize=11, y=1.01)

    # ── Top-left: dDeg/dDoD per year ─────────────────────────────────────
    ax = axes[0, 0]
    valid_years = [yr for yr, dd in zip(years, dDeg_dDoD) if dd is not None]
    valid_dDeg  = [dd for dd in dDeg_dDoD if dd is not None]
    valid_rcols = [c  for c, dd in zip(regime_colors, dDeg_dDoD) if dd is not None]
    bar_colors  = [
        C_HIGH if yr in multiyear["replacement_years"] else c
        for yr, c in zip(valid_years, valid_rcols)
    ]
    ax.bar(valid_years, valid_dDeg, width=0.7, color=bar_colors, alpha=0.82)
    for yr in multiyear["replacement_years"]:
        ax.axvline(yr + 0.5, linestyle="-.", linewidth=1.4, color="red", alpha=0.8)
    mean_g = np.mean(valid_dDeg) if valid_dDeg else 0.0
    ax.axhline(mean_g, linestyle="--", linewidth=1.2, color="#2c3e50", alpha=0.8)
    ax.set_xlabel("Project year")
    ax.set_ylabel("dDeg/dDoD  [EUR / MWh]")
    ax.set_title("dDeg/dDoD per Year") #red = high-alignment regime, blue = low-alignment regime
    ax.set_xlim(0.5, n_years + 0.5)
    ax.set_xticks(range(2, n_years + 1, 2))
    ax.legend(handles=[
        Patch(facecolor=C_HIGH, alpha=0.82, label="High-alignment"),
        Patch(facecolor=C_LOW,  alpha=0.82, label="Low-alignment"),
        Line2D([0], [0], linestyle="--", color="#2c3e50",
               label=f"Mean: {mean_g:.3e}" if valid_dDeg else "Mean: N/A"),
    ], fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")


    # ── Top-right: alignment vs dDeg/dDoD scatter ─────────────────────────
    ax = axes[0, 1]
    valid_align = [
        (yr, al, dd, c)
        for yr, al, dd, c in zip(years, alignment, dDeg_dDoD, regime_colors)
        if al is not None and dd is not None
    ]
    if valid_align:
        va_yrs, va_al, va_dd, va_cols = zip(*valid_align)
        ax.scatter(va_al, va_dd, color=va_cols, s=65, zorder=3,
                   edgecolors="white", linewidths=0.5, alpha=0.88)
        for yr, al, dd in zip(va_yrs, va_al, va_dd):
            ax.annotate(str(yr), (al, dd), fontsize=6.5,
                        xytext=(3, 3), textcoords="offset points", alpha=0.85)
        if len(va_al) >= 2:
            m, b = np.polyfit(va_al, va_dd, 1)
            xs = np.linspace(min(va_al), max(va_al), 60)
            r  = np.corrcoef(va_al, va_dd)[0, 1]
            ax.plot(xs, m * np.array(xs) + b, "--", color="#2c3e50",
                    linewidth=1.3, alpha=0.7, label=f"r = {r:+.3f}")
            ax.set_title(f"Alignment Factor vs dDeg/dDoD  (r = {r:+.3f})") # alignment = dDeg/dDoD / (e_cap × |subgrad| × |dual|)
        else:
            ax.set_title("Alignment Factor vs dDeg/dDoD")
    ax.axvline(ALIGN_THRESHOLD, linestyle=":", linewidth=1.2,
               color="grey", alpha=0.6, label=f"Regime boundary = {ALIGN_THRESHOLD:.0f}")
    ax.set_xlabel("Alignment factor  =  dDeg/dDoD / (e_cap × |subgrad| × |dual|)")
    ax.set_ylabel("dDeg/dDoD  [EUR / MWh]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # ── Bottom-left: mean |subgrad| bars + mean DoD overlay ───────────────
    ax = axes[1, 0]
    valid_mas_yrs = [yr for yr, m in zip(years, mean_abs_subgrad) if m is not None]
    valid_mas     = [m  for m  in mean_abs_subgrad if m is not None]
    ax.bar(valid_mas_yrs, valid_mas, width=0.7, color="#8e44ad", alpha=0.80)
    for yr in multiyear["replacement_years"]:
        ax.axvline(yr + 0.5, linestyle="-.", linewidth=1.4, color="red", alpha=0.8)
    ax.set_xlabel("Project year")
    ax.set_ylabel("mean |subgrad_combined|  [EUR / MWh]")
    ax.set_xlim(0.5, n_years + 0.5)
    ax.set_xticks(range(2, n_years + 1, 2))
    ax.grid(True, alpha=0.25, axis="y")
 
    valid_dod_yrs = [yr for yr, d in zip(years, mean_dod) if d is not None]
    valid_dod     = [d * 100 for d in mean_dod if d is not None]
    if valid_dod:
        ax_dod = ax.twinx()
        ax_dod.plot(valid_dod_yrs, valid_dod, "s--", color="#e67e22",
                    linewidth=1.3, markersize=4, alpha=0.85, label="mean DoD [%]")
        ax_dod.set_ylabel("mean DoD [%]", color="#e67e22", fontsize=9)
        ax_dod.tick_params(axis="y", colors="#e67e22", labelsize=8)
        ax_dod.set_ylim(0, max(valid_dod) * 2.2)
        ax_dod.legend(fontsize=8, loc="lower right")
    ax.set_title("Mean |Subgradient| and Mean DoD per Year") # neither scalar tracks the gradient oscillation : alignment (dot product structure) does
 

    # ── Bottom-right: scalar components vs gradient, normalised ───────────
    ax = axes[1, 1]
    comp_ecap = [ec for ec in e_cap_eff_yr    if ec is not None]
    comp_mas  = [m  for m  in mean_abs_subgrad if m  is not None]
    comp_mad  = [d  for d  in mean_abs_dual    if d  is not None]
    comp_grad = [dd for dd in dDeg_dDoD        if dd is not None]
    valid_comp_yrs = [yr for yr, ec in zip(years, e_cap_eff_yr) if ec is not None]
 
    if comp_ecap and comp_mas and comp_mad and comp_grad:
        comp_product = [ec * m * d for ec, m, d in zip(comp_ecap, comp_mas, comp_mad)]
 
        def _norm(lst: list) -> list:
            mu = np.mean(lst)
            return [v / mu for v in lst]
 
        for yr in multiyear["replacement_years"]:
            ax.axvspan(yr, n_years + 0.5, alpha=0.07, color=C_HIGH,
                       label="2nd battery lifetime")
 
        ax.plot(valid_comp_yrs, _norm(comp_grad),    "o-",  color="#2c3e50",
                linewidth=2.0, markersize=5, zorder=4, label="dDeg/dDoD  (normalised)")
        ax.plot(valid_comp_yrs, _norm(comp_product), "s--", color="#3498db",
                linewidth=1.3, markersize=4, alpha=0.85,
                label="e_cap × |subgrad| × |dual|  (normalised)")
        ax.plot(valid_comp_yrs, _norm(comp_mas),     "^--", color="#8e44ad",
                linewidth=1.3, markersize=4, alpha=0.85, label="mean |subgrad|  (normalised)")
        ax.plot(valid_comp_yrs, _norm(comp_mad),     "D--", color="#27ae60",
                linewidth=1.3, markersize=4, alpha=0.85, label="mean |dual|  (normalised)")
 
        ax.axhline(1.0, linestyle=":", linewidth=1.0, color="grey", alpha=0.5)
        for yr in multiyear["replacement_years"]:
            ax.axvline(yr + 0.5, linestyle="-.", linewidth=1.4, color="red", alpha=0.8)
 
    ax.set_xlabel("Project year")
    ax.set_ylabel("Value / mean  (normalised to 1.0)")          # |subgrad|, |dual|, scalar product all flat : gradient oscillates 2.5×
    ax.set_title("Scalar Components vs Gradient : Normalised")  # alignment (dot product structure) is the sole driver
    ax.set_xlim(0.5, n_years + 0.5)
    ax.set_xticks(range(2, n_years + 1, 2))
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25, axis="y")
 
    plt.tight_layout()
    save_path = plots_dir / f"gradient_analysis_{run_label}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  ✓ Plot: {save_path.name}")
    if not show:
        plt.close("all")
         
# =============================================================================
# Subgradient time series : 2 panels
# =============================================================================

def plot_subgradient_timeseries(
    multiyear: Dict,
    run_label: str,
    plots_dir: Path,
    show: bool = False,
) -> None:
    """Two-panel subgradient time series: year 1 vs EoL year.
 
    Left axis: per-timestep subgrad_combined [EUR/MWh].
    Right axis: SoC [MWh].
    Panel label includes SoH and dDeg/dDoD for that year.
    """
    ann_grad = multiyear.get("annual_gradient", [])
    ann_soc  = multiyear.get("annual_soc", [])
    traj     = multiyear["soh_trajectory"]
 
    if not ann_grad or not ann_soc:
        print("  ⚠ Skipping subgradient time series : data unavailable.")
        return
 
    repl_years = multiyear["replacement_years"]
    if repl_years:
        eol_yr      = repl_years[0]
        panel_years = [1, eol_yr]
        panel_idxs  = [0, eol_yr - 1]
    else:
        panel_years = [1, traj[-1][0]]
        panel_idxs  = [0, len(traj) - 1]
 
    time_days = np.arange(len(ann_soc[0])) / 24.0
 
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Subgradient Time Series : Year 1 vs Year {panel_years[1]}",   # Note: left axis = per-timestep subgradient [EUR/MWh]; right axis = SoC [MWh]
        fontsize=11, y=1.01,                                            # Note: comparing year 1 (fresh battery) vs EoL year (degraded capacity)
    )

    colours = ["#4C72B0", "#e74c3c"]
    for ax, yr, idx, col in zip(axes, panel_years, panel_idxs, colours):
        if idx >= len(ann_grad) or ann_grad[idx] is None:
            ax.text(0.5, 0.5, f"Year {yr} : gradient unavailable",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        subgrad  = ann_grad[idx][6]   # full subgrad_combined array
        grad_val = ann_grad[idx][0]   # dDeg/dDoD scalar
        soc      = ann_soc[idx]
        soh_p    = traj[idx][1]
        t        = time_days[:len(subgrad)]

        # Left axis,  subgradient
        ax.plot(t, subgrad, linewidth=0.6, color=col, alpha=0.85,
                label=f"yr {yr}  |  SoH = {soh_p:.1f}%  |  dDeg/dDoD = {grad_val:.2e}")
        ax.set_ylabel("subgrad_combined\n[EUR / MWh]", color=col)
        ax.tick_params(axis="y", labelcolor=col)
        ax.axhline(0, linewidth=0.8, color="grey", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.20)
        ax.legend(loc="upper left", fontsize=8)

        # Right axis,  SoC
        ax2 = ax.twinx()
        ax2.plot(t[:len(soc)], soc, linewidth=0.5, color="#95a5a6",
                 alpha=0.60, label="SoC [MWh]")
        ax2.set_ylabel("SoC [MWh]", color="#7f8c8d")
        ax2.tick_params(axis="y", labelcolor="#7f8c8d")
        ax2.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time [days]")
    plt.tight_layout()
    save_path = plots_dir / f"subgradient_timeseries_{run_label}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  ✓ Plot: {save_path.name}")
    if not show:
        plt.close("all")

 
# =============================================================================
# Multi-year degradation trajectory : 2×2
# =============================================================================
 
def plot_multiyear_trajectory(
    multiyear: Dict,
    run_label: str,
    plots_dir: Path,
    n_years: int,
    eol_replacement: float,
    show: bool = False,
) -> None:
    """2×2 multi-year degradation summary figure.
 
    Top-left    SoH trajectory with EoL crossings and replacement events
    Top-right   Annual fd bars (Shi cycles + Xu calendar)
    Bottom-left Year-over-year SoH loss rate
    Bottom-right fd decomposition: cycle vs calendar stacked bars
    """
    traj    = multiyear["soh_trajectory"]
    years   = [r[0] for r in traj]
    soh_pct = [r[1] for r in traj]

    # Unpack tuple annual_fd: (fd_total, fd_cycle, fd_calendar)
    ann_fd_total    = [t[0] for t in multiyear["annual_fd"]]
    ann_fd_cycle    = [t[1] for t in multiyear["annual_fd"]]
    ann_fd_calendar = [t[2] for t in multiyear["annual_fd"]]

    # Prepend year-0 origin
    years_full = [0] + years
    soh_full   = [100.0] + soh_pct

# Effective capacity: E_nominal × (SoH/100), reset at replacement
# e_cap_nominal_approx = multiyear.get("e_cap_nominal", 300.0)


    # EoL crossing interpolator
    def _eol_crossing(soh_list, year_list, threshold):
        for i in range(1, len(soh_list)):
            if soh_list[i-1] >= threshold >= soh_list[i]:
                frac = (soh_list[i-1] - threshold) / max(soh_list[i-1] - soh_list[i], 1e-9)
                return year_list[i-1] + frac * (year_list[i] - year_list[i-1])
        return None

    thr_colours = {80.0: "#e74c3c", 70.0: "#e67e22"}
    thr_labels  = {80.0: "80% [IEC/EV]", 70.0: "70% [warranty]"}
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Multi-year Degradation : Shi (2018) + Xu calendar  |  {n_years} yr",
        fontsize=11, y=1.01,
    )

    # ── Top-left: SoH trajectory ─────────────────────────────────────────
    ax = axes[0, 0]
    
    segments = _split_battery_segments(years_full, soh_full, multiyear["replacement_years"])
    for sx, sy in segments:
        ax.plot(sx, sy, linewidth=2.0, color="#4C72B0", linestyle="-",
                marker="o", markersize=3.5, zorder=3, label="_nolegend_")

    # ── Connector: vertical dotted line between end of Battery i and start of Battery i+1
    for i in range(len(segments) - 1):
        x_conn  = segments[i][0][-1]   # replacement year (same for both endpoints)
        y_bot   = segments[i][1][-1]   # EoL SoH of outgoing battery (~70%)
        y_top   = segments[i+1][1][0]  # 100.0,  fresh battery anchor
        ax.plot([x_conn, x_conn], [y_bot, y_top],
                linewidth=2.0, color="#4C72B0", linestyle="-", alpha=1.0, zorder=4)

    crossing_lines = []
    for thr, col in thr_colours.items():
        ln = ax.axhline(thr, linestyle="--", linewidth=1.2, color=col, alpha=0.85)
        crossing_lines.append((ln, thr_labels[thr]))
        cross = _eol_crossing(soh_full, years_full, thr)
        if cross is not None and cross <= n_years:
            ax.axvline(cross, linestyle=":", linewidth=1.0, color=col, alpha=0.4, zorder=1)
            ha     = "right" if cross > n_years * 0.6 else "left"
            offset = -0.4 if ha == "right" else 0.4
            ax.annotate(
                f"{thr:.0f}%  yr {cross:.1f}",
                xy=(cross, thr), xytext=(cross + offset, thr + 2.0),
                fontsize=7.5, color=col, fontweight="bold", ha=ha,
                arrowprops=dict(arrowstyle="-", color=col, lw=0.6),
            )

    repl_handle = None
    for yr in multiyear["replacement_years"]:
        repl_handle = ax.axvline(yr, linestyle="-.", linewidth=1.5,
                                 color="red", alpha=0.85, zorder=2)
        ax.text(yr - 0.3, min(soh_pct) - 3.5, f"Replace\nyr {yr}",
                fontsize=7, color="red", ha="right", va="top")

    # Build legend manually,  single SoH entry + threshold lines + replacement
    leg_handles = [Line2D([0], [0], color="#4C72B0", lw=2, marker="o",
                          markersize=4, label="Simulated SoH")]
    for ln, lbl in crossing_lines:
        leg_handles.append(Line2D([0], [0], linestyle="--",
                                  color=ln.get_color(), lw=1.2, label=lbl))
    if repl_handle is not None:
        leg_handles.append(Line2D([0], [0], linestyle="-.", color="red",
                                  lw=1.5, label="Battery replacement"))

    min_soh = min(soh_pct) if soh_pct else 55.0
    ax.set_xlim(0, n_years)
    ax.set_xticks(range(0, n_years + 1, 2))
    ax.set_ylim(max(min_soh - 8, 50), 103)
    ax.set_xlabel("Project year")
    ax.set_ylabel("State of Health [%]")
    ax.set_title("State of Health Trajectory") # capacity fade feedback: LP re-solved each year with degraded e_cap_eff
    ax.legend(handles=leg_handles, fontsize=8, loc="lower left")    # Place legend in lower-left where SoH line hasn't reached yet
    ax.grid(True, alpha=0.25)

    # ── Top-right: annual fd bars,  zoomed to show acceleration ──────────
    ax = axes[0, 1]
    ax.bar(years, ann_fd_total, color="#55A868", alpha=0.80, width=0.7,
           label="Annual fd  (Shi cycles + Xu calendar)")
    for yr in multiyear["replacement_years"]:
        ax.axvline(yr + 0.5, linestyle="-.", linewidth=1.4, color="red", alpha=0.8)
    # Note: fd accelerates over the project lifetime : capacity fade → deeper relative cycles
    # Note: pct_acc = 100 * (ann_fd_total[-1] - ann_fd_total[0]) / ann_fd_total[0] : printable if needed

    ax.set_xlabel("Project year")
    ax.set_ylabel("Annual fd (Shi + calendar)")
    ax.set_title("Annual Degradation Rate  (fd)") # Note: fd = Shi cycle accumulation + Xu calendar correction (reporting path only)
    ax.set_xlim(0.5, n_years + 0.5)
    ax.set_xticks(range(2, n_years + 1, 2)) #even numbering 2,4,....,20
    # Zoom y-axis tightly to show the slow acceleration
    fd_min = min(ann_fd_total) * 0.995
    fd_max = max(ann_fd_total) * 1.010
    ax.set_ylim(fd_min, fd_max)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25, axis="y")

    # ── Bottom-left: year-over-year SoH loss rate ────────────────────────
    ax = axes[1, 0]

    post_repl_years = {yr + 1 for yr in multiyear["replacement_years"]} # ΔSoH per year,  derived from the full trajectory including year-0.
    delta_soh = []                                                      # For post-replacement years the reference is 100% (fresh battery), not the previous year's degraded SoH.
    for i, yr in enumerate(years):
        if yr in post_repl_years:
            delta_soh.append(100.0 - soh_full[i + 1])   # loss from fresh battery
        else:
            delta_soh.append(soh_full[i] - soh_full[i + 1])

    # All years are now valid,  post-replacement years use 100% as reference
    loss_years  = years
    loss_values = delta_soh
    bar_colors  = ["#27ae60" if yr in post_repl_years
                   else "#e74c3c" if yr in multiyear["replacement_years"]
                   else "#4C72B0" for yr in loss_years]
    ax.bar(loss_years, loss_values, width=0.7, color=bar_colors, alpha=0.82)

    normal_losses = [d for yr, d in zip(loss_years, loss_values)   # Mean loss rate over normal operating years (exclude the EoL year itself
                     if yr not in multiyear["replacement_years"]]  # as it may be a partial year below threshold, and post-replacement year 1
    if normal_losses:                                              # is included since it represents genuine first-year loss of a fresh battery)
        mean_loss = np.mean(normal_losses)
        ax.axhline(mean_loss, linestyle="--", linewidth=1.3, color="#2c3e50",
                   alpha=0.8, label=f"Mean loss rate: {mean_loss:.3f} %/yr")

    # Mark replacement events
    for yr in multiyear["replacement_years"]:
        ax.axvline(yr + 0.5, linestyle="-.", linewidth=1.4, color="red", alpha=0.8)
        idx = loss_years.index(yr)
        ax.text(yr, loss_values[idx] + 0.05, f"EoL yr {yr}",
                fontsize=7, color="red", ha="center")

    # Add legend entry for post-replacement bar colour
    extra_handles = [
        Patch(facecolor="#27ae60", alpha=0.82,
              label="Post-replacement yr\n(ref = 100% fresh battery)"),
        Patch(facecolor="#e74c3c", alpha=0.82, label="EoL year"),
        Patch(facecolor="#4C72B0", alpha=0.82, label="Normal operating year"),
    ]

    ax.set_xlabel("Project year")
    ax.set_ylabel("Annual SoH loss [%/yr]")
    ax.set_title("Annual SoH Loss Rate") # increasing trend = degradation accelerating as capacity fades and cycles deepen
    ax.set_xlim(0.5, n_years + 0.5)
    ax.set_xticks(range(2, n_years + 1, 2)) #even numbering 2,4,....,20
    ax.set_ylim(0, max(loss_values) * 1.35 if loss_values else 10)
    mean_handle, _ = ax.get_legend_handles_labels()
    mean_handle, _ = ax.get_legend_handles_labels()
    ax.legend(handles=mean_handle + extra_handles, fontsize=7.5, loc="upper right")   
    ax.grid(True, alpha=0.25, axis="y")

    # ── Bottom-right: exact cycle vs calendar stacked bars ────────────────
    ax = axes[1, 1]
    ax.bar(years, ann_fd_cycle, width=0.7, color="#3498db", alpha=0.82,
           label="Cycle fd  (Shi Φ accumulation)")
    ax.bar(years, ann_fd_calendar, width=0.7, color="#e67e22", alpha=0.82,
           bottom=ann_fd_cycle, label="Calendar fd  (Xu ft_calendar)")

    for yr in multiyear["replacement_years"]:
        ax.axvline(yr + 0.5, linestyle="-.", linewidth=1.4, color="red", alpha=0.8)

    # Mean fractions over the run
    mean_cal_pct = 100.0 * np.mean(ann_fd_calendar) / max(np.mean(ann_fd_total), 1e-9)
    mean_cyc_pct = 100.0 - mean_cal_pct
    ax.set_xlabel("Project year")
    ax.set_ylabel("Annual fd contribution")
    ax.set_title(f"fd Decomposition : Cycle {mean_cyc_pct:.0f}%  /  Calendar {mean_cal_pct:.0f}%") # cycle fd from Shi Φ accumulation; calendar fd from Xu ft_calendar (reporting path only)
    ax.set_xlim(0.5, n_years + 0.5)
    ax.set_xticks(range(2, n_years + 1, 2)) #even numbering 2,4,....,20
    ax.set_ylim(0, max(ann_fd_total) * 1.12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    save_path = plots_dir / f"multiyear_trajectory_{run_label}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  ✓ Plot: {save_path.name}")
    if not show:
        plt.close("all")

