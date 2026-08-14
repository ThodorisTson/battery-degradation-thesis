"""
verify_week_audit.py
====================

Independent audit of the week-snapshot verification outputs, and generator of the two LaTeX tables used in Section 4.2 of the thesis.

This script does NOT solve anything. It reads the CSV files written by `verify_week_snapshot.py` and recomputes, from those published numbers alone,
every quantity Section 4.2 asserts. The point is traceability: a reader who has the CSV files can run this and reproduce the reported degradation without
access to the solver, the wind data or the price data.

Inputs (expected in the same folder as this script, or in a subfolder given by --indir):

    week_snapshot_summary.csv
    week_snapshot_cycles.csv
    week_snapshot_hourly.csv
    week_snapshot_checks.csv      (optional; only used for the centre report)

Outputs, written next to this script:

    tab_week_checks_body.tex      body rows of the verification-checks table
    tab_week_stats_body.tex       body rows of the week-statistics table
    week_audit_report.txt         full console report

Usage, from a terminal in VS Code on Windows:

    python verify_week_audit.py
    python verify_week_audit.py --indir "Results/Week Snapshot"

Exit code is 0 if every check passes and 1 otherwise.
"""

from __future__ import annotations
from degradation.paths import RESULTS_DIR

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Constants. These are written out here on purpose, independently of the degradation modules, so that a wrong coefficient in the model shows up as a
# disagreement instead of cancelling on both sides of the comparison.
# ----------------------------------------------------------------------
E_NOM_MWH = 300.0                    # nominal energy capacity
P_RATED_MW = 150.0                   # nominal power rating
ETA_ONEWAY = 0.9539392307945773      # one-way efficiency, symmetric split
K_SIGMA = 1.04                       # Xu mean-SoC stress coefficient
SIGMA_REF = 0.50                     # Xu mean-SoC reference
K_T = 4.14e-10                       # Xu calendar rate, per second
DT_H = 1.0                           # dispatch time step, hours

# Bump this whenever the emitted LaTeX changes, so the console shows at a glance which version produced the tables now sitting in the thesis.
SCRIPT_VERSION = "2026-07-29c  notation aligned: k_tau, p_t^s, Pbar, eta_in"
DELTA_C = 0.1437                     # inflection point of Xu S_delta
WEEK_HOURS = 168
SECONDS_PER_HOUR = 3600.0

# Xu LMO depth stress factor: S_delta = 1 / (a * delta^-b - c)
XU_A, XU_B, XU_C = 1.40e5, 0.501, 1.23e5
DELTA_FLOOR = 1e-6                   # argument clip used by the implementation

# Windows in the order they appear in the thesis tables.
WIDTH_SERIES = ["20-80", "10-90", "0-100"]
CENTRE_SERIES = ["0-80", "10-90", "20-100"]

# Column suffix in week_snapshot_hourly.csv for each window.
HOURLY_SUFFIX = {
    "10-90": "10_90",
    "20-80": "20_80",
    "0-100": "0_100",
    "20-100": "20_100",
    "0-80": "0_80",
}

RTOL = 1e-9      # tolerance for "reproduces the reported value"
ATOL_MWH = 1e-9  # tolerance for bound violations, in MWh


# ----------------------------------------------------------------------
# Xu stress factors, written out from the published coefficients
# ----------------------------------------------------------------------
def s_delta(delta):
    """Depth stress factor of the Xu et al. (2016) LMO model."""
    d = np.maximum(np.asarray(delta, dtype=float), DELTA_FLOOR)
    return 1.0 / (XU_A * d ** (-XU_B) - XU_C)


def s_sigma(mean_soc):
    """Mean state-of-charge stress factor."""
    return np.exp(K_SIGMA * (np.asarray(mean_soc, dtype=float) - SIGMA_REF))


# ----------------------------------------------------------------------
# Small check-recording helper
# ----------------------------------------------------------------------
class Audit:
    def __init__(self):
        self.rows = []
        self.ok = True

    def record(self, name, passed, detail):
        self.rows.append((name, bool(passed), detail))
        if not passed:
            self.ok = False

    def report(self):
        lines = []
        width = max(len(n) for n, _, _ in self.rows)
        for name, passed, detail in self.rows:
            tag = "PASS" if passed else "FAIL"
            lines.append(f"  [{tag}] {name.ljust(width)}  {detail}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Input location
# ----------------------------------------------------------------------
def resolve_indir(user_indir: str | None) -> Path:
    here = Path(__file__).resolve().parent
    candidates = []
    if user_indir:
        candidates.append(Path(user_indir) if Path(user_indir).is_absolute()
                          else here / user_indir)
    candidates += [here, RESULTS_DIR / "week_snapshot"]
    for c in candidates:
        if (c / "week_snapshot_summary.csv").is_file():
            return c
    tried = "\n    ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "week_snapshot_summary.csv not found. Looked in:\n    " + tried
    )


# ----------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------
def check_degradation_recompute(summary, cycles, audit):
    """
    Recompute both degradation terms of every window from the published cycle
    list and the closed-form calendar expression, and compare with the values
    the run reported.
    """
    worst_cyc = 0.0
    worst_cal = 0.0
    for _, row in summary.iterrows():
        win = row["window"]
        sel = cycles[(cycles["window"] == f"{win}%") & (cycles["scope"] == "week")]
        if sel.empty:
            audit.record(f"recompute f_d, {win}", False, "no week rows in cycle list")
            continue

        # Cycle term: sum over counted records of n * S_delta * S_sigma.
        # S_T is 1 at the 25 C reference temperature used throughout.
        fd_cyc = float(
            (sel["count"].values * s_delta(sel["depth"].values)
             * s_sigma(sel["mean_soc"].values)).sum()
        )
        # Calendar term: k_t * t * S_sigma(mean SoC over the week).
        fd_cal = float(
            K_T * WEEK_HOURS * SECONDS_PER_HOUR * s_sigma(row["week_mean_soc"])
        )

        r_cyc = abs(fd_cyc - row["week_fd_cycle"]) / abs(row["week_fd_cycle"])
        r_cal = abs(fd_cal - row["week_fd_calendar"]) / abs(row["week_fd_calendar"])
        worst_cyc = max(worst_cyc, r_cyc)
        worst_cal = max(worst_cal, r_cal)

        audit.record(
            f"cycle term from cycle list, {win}",
            r_cyc < RTOL,
            f"recomputed {fd_cyc:.6e}  reported {row['week_fd_cycle']:.6e}  rel {r_cyc:.1e}",
        )
        audit.record(
            f"calendar term closed form, {win}",
            r_cal < RTOL,
            f"recomputed {fd_cal:.6e}  reported {row['week_fd_calendar']:.6e}  rel {r_cal:.1e}",
        )
    return worst_cyc, worst_cal


def check_bounds_and_rating(summary, hourly, audit):
    """State of charge stays inside its window; power never exceeds the rating."""
    worst_bound_mwh = 0.0
    worst_overrun_mw = 0.0
    for _, row in summary.iterrows():
        win = row["window"]
        suf = HOURLY_SUFFIX[win]
        soc = hourly[f"soc_{suf}"].values
        p = hourly[f"storage_p_mw_{suf}"].values
        lo, hi = float(row["soc_min"]), float(row["soc_max"])

        viol_mwh = max((lo - soc).max(), (soc - hi).max()) * E_NOM_MWH
        overrun = float(np.abs(p).max()) - P_RATED_MW
        worst_bound_mwh = max(worst_bound_mwh, viol_mwh)
        worst_overrun_mw = max(worst_overrun_mw, overrun)

        audit.record(
            f"SoC inside window, {win}",
            viol_mwh < ATOL_MWH,
            f"max violation {viol_mwh:+.3e} MWh of {E_NOM_MWH:.0f} MWh",
        )
        audit.record(
            f"power within rating, {win}",
            overrun < 1e-9,
            f"max |P| {np.abs(p).max():.10f} MW, overrun {overrun:+.2e} MW",
        )
    return worst_bound_mwh, worst_overrun_mw


def check_record_bookkeeping(summary, cycles, audit):
    """Record split and total count in the summary agree with the cycle list."""
    for _, row in summary.iterrows():
        win = row["window"]
        sel = cycles[(cycles["window"] == f"{win}%") & (cycles["scope"] == "week")]
        adj = cycles[(cycles["window"] == f"{win}%") & (cycles["scope"] != "week")]

        n_rec = len(sel)
        n_full = int((sel["count"] == 1.0).sum())
        n_half = int((sel["count"] == 0.5).sum())
        total = float(sel["count"].sum())
        n_dc = int((sel["depth"] > DELTA_C).sum())

        ok = (
            n_rec == int(row["week_records"])
            and n_full == int(row["week_records_full"])
            and n_half == int(row["week_records_half"])
            and abs(total - row["week_cycles"]) < 1e-12
            and n_full + n_half == n_rec
            and n_dc == int(row["week_above_dc"])
            and len(adj) == int(row["week_straddling"])
        )
        audit.record(
            f"record bookkeeping, {win}",
            ok,
            f"{n_rec} records = {n_full} full + {n_half} half, count {total:.1f}, "
            f"{n_dc} above delta_c, {len(adj)} straddling",
        )

def check_annual_record_split(summary, indir, audit):
    """Annual full and half split in the summary agrees with the annual cycle list.

    The split quoted in Section 4.3 is measured here rather than derived. The
    identities F = 2S - R and H = 2(R - S) follow from R = F + H and
    S = F + H/2, and hold only if every count is exactly 1.0 or 0.5. The
    'other' column tests that condition instead of assuming it.
    """
    path = indir / "week_snapshot_year_cycles.csv"
    if not path.exists():
        audit.record("annual record split", False,
                     "week_snapshot_year_cycles.csv not found; re-run the snapshot")
        return
    yc = pd.read_csv(path)
    for _, row in summary.iterrows():
        win = row["window"]
        sel = yc[yc["window"] == f"{win}%"]
        if sel.empty:
            continue
        R = len(sel)
        F = int((sel["count"] == 1.0).sum())
        H = int((sel["count"] == 0.5).sum())
        other = R - F - H
        S = float(sel["count"].sum())
        ok = (
            R == int(row["year_records"])
            and F == int(row["year_records_full"])
            and H == int(row["year_records_half"])
            and other == 0
            and abs(S - row["year_cycles"]) < 1e-12
            and F == round(2 * S - R)
            and H == round(2 * (R - S))
        )
        audit.record(
            f"annual record split, {win}",
            ok,
            f"{R} records = {F} full + {H} half ({other} other), "
            f"summed multiplicity {S:.1f}, "
            f"identities 2S-R = {2*S - R:.0f}, 2(R-S) = {2*(R - S):.0f}",
        )

def check_centre_invariance(summary, hourly, audit):
    """
    Shifting the window centre at fixed width must leave the power schedule unchanged and offset the state of charge by exactly the shift. Degradation
    must then scale by exp(k_sigma * shift).
    """
    ref = "10-90"
    ref_p = hourly[f"storage_p_mw_{HOURLY_SUFFIX[ref]}"].values
    ref_soc = hourly[f"soc_{HOURLY_SUFFIX[ref]}"].values
    s_idx = summary.set_index("window")

    worst_dp = 0.0
    worst_doffset = 0.0
    worst_ratio_err = 0.0

    for win in CENTRE_SERIES:
        if win == ref:
            continue
        shift = float(s_idx.loc[win, "soc_min"] - s_idx.loc[ref, "soc_min"])
        p = hourly[f"storage_p_mw_{HOURLY_SUFFIX[win]}"].values
        soc = hourly[f"soc_{HOURLY_SUFFIX[win]}"].values

        dp = float(np.abs(p - ref_p).max())
        doffset = float(np.abs((soc - ref_soc) - shift).max())
        worst_dp = max(worst_dp, dp)
        worst_doffset = max(worst_doffset, doffset)

        audit.record(
            f"center invariance, dispatch, {win}",
            dp < 1e-6,
            f"max |dP| {dp:.3e} MW over {len(p)} steps",
        )
        audit.record(
            f"center invariance, SoC offset, {win}",
            doffset < 1e-9,
            f"offset {shift:+.2f} held to {doffset:.3e}",
        )

        observed = float(s_idx.loc[win, "week_fd"] / s_idx.loc[ref, "week_fd"])
        predicted = float(np.exp(K_SIGMA * shift))
        err = abs(observed - predicted) / predicted
        worst_ratio_err = max(worst_ratio_err, err)
        audit.record(
            f"degradation ratio vs exp(k_sigma dSoC), {win}",
            err < 1e-4,
            f"observed {observed:.6f}  predicted {predicted:.6f}  rel {err:.1e}",
        )
    return worst_dp, worst_doffset, worst_ratio_err


def check_efc(summary, hourly, audit):
    """
    Equivalent full cycles as the run defines them: discharge throughput at the AC terminal divided by nominal capacity, EFC = sum(p[p>0]) * dt / E_nom.
    """
    worst = 0.0
    for _, row in summary.iterrows():
        win = row["window"]
        p = hourly[f"storage_p_mw_{HOURLY_SUFFIX[win]}"].values
        efc = float(p[p > 0].sum() * DT_H / E_NOM_MWH)
        rel = abs(efc - row["week_efc"]) / abs(row["week_efc"])
        worst = max(worst, rel)
        audit.record(
            f"equivalent full cycles, {win}",
            rel < RTOL,
            f"recomputed {efc:.9f}  reported {row['week_efc']:.9f}  rel {rel:.1e}",
        )
    return worst


def check_rated_hour_positioning(hourly, audit):
    """
    Ahead of the largest price peak of the week the model rests exactly one rated charging hour below the window ceiling, in every width-series window.
    This is a closed form: P_rated * 1 h * eta / E_nom.
    """
    rise = P_RATED_MW * 1.0 * ETA_ONEWAY / E_NOM_MWH
    hi = {"20-80": 0.8, "10-90": 0.9, "0-100": 1.0}
    worst = 0.0
    for win in WIDTH_SERIES:
        soc = hourly[f"soc_{HOURLY_SUFFIX[win]}"].values
        rest = float(soc[99])
        expected = hi[win] - rise
        err = abs(rest - expected)
        worst = max(worst, err)
        audit.record(
            f"pre-peak resting level, {win}",
            err < 1e-9,
            f"h99 SoC {rest:.6f}  ceiling - {rise:.6f} = {expected:.6f}  err {err:.1e}",
        )
    return rise, worst


def report_shared_events(cycles):
    """
    List the intermediate-depth records whose depth is identical in all three width-series windows, and those that are not. Used to support the text.
    """
    per_win = {}
    for win in WIDTH_SERIES:
        sel = cycles[(cycles["window"] == f"{win}%") & (cycles["scope"] == "week")]
        mid = sel[(sel["depth"] > DELTA_C) & (sel["depth"] < sel["depth"].max() - 1e-9)]
        per_win[win] = {round(float(d), 6): int(h) for d, h in
                        zip(mid["depth"], mid["hour_in_week_start"])}
    common = set(per_win[WIDTH_SERIES[0]])
    for win in WIDTH_SERIES[1:]:
        common &= set(per_win[win])
    lines = ["  depths present in all three width-series windows:"]
    for d in sorted(common):
        hrs = {w: per_win[w][d] for w in WIDTH_SERIES}
        lines.append(f"    delta = {d:.4f}   start hour {hrs}")
    lines.append("  depths present in only some windows:")
    for win in WIDTH_SERIES:
        extra = sorted(set(per_win[win]) - common)
        if extra:
            lines.append(f"    {win:6s}: " +
                         ", ".join(f"{d:.4f} (h{per_win[win][d]})" for d in extra))
    return "\n".join(lines)


# ----------------------------------------------------------------------
# LaTeX table bodies
# ----------------------------------------------------------------------
def latex_stats_body(summary) -> str:
    s_idx = summary.set_index("window")
    out = []
    for win in WIDTH_SERIES:
        r = s_idx.loc[win]
        label = win.replace("-", "--") + r"\%"
        out.append(
            f"{label:10s} & {int(r.week_records_full)} + {int(r.week_records_half)} "
            f"& {r.week_cycles:.1f} & {r.week_efc:.3f} & {r.week_mean_depth:.3f} "
            f"& {r.week_fd_cycle * 1e4:.3f} & {r.week_fd_calendar * 1e4:.3f} "
            f"& {r.week_fd * 1e4:.3f} \\\\"
        )
    return "\n".join(out)


def tex_sci(x, sig=1) -> str:
    """Format a number as LaTeX scientific notation, e.g. 4.0e-13 -> 4 \\times 10^{-13}."""
    if x == 0:
        return "0"
    mant, exp = f"{x:.{sig - 1}e}".split("e")
    mant = mant.rstrip("0").rstrip(".") or "1"
    return rf"{mant} \times 10^{{{int(exp)}}}"


def latex_checks_body(worst) -> str:
    """One row per check. Column 2 names the independent reference the check is compared against, as a closed form where one exists. This keeps the
    table inside \\textwidth; the earlier prose column overflowed it."""
    rows = [
        ("Provenance", "Stored baseline run", "$0$"),
        ("Operating limits", "Window bounds",
         rf"${tex_sci(worst['bound_mwh'])}$~MWh"),
        ("Operating limits", "Power rating",
         rf"${tex_sci(worst['overrun'])}$~MW"),
        ("Bookkeeping", "Published cycle list", "exact"),
        ("Cycle term", r"$\sum_i n_i\,S_\delta\,S_\sigma$",
         rf"${tex_sci(worst['cyc'])}$"),
        ("Calendar term", r"$k_\tau\,t\,S_\sigma(\bar{\sigma})$",
         rf"${tex_sci(worst['cal'])}$"),
        ("Center invariance", "Shifted power schedule",
         rf"${tex_sci(worst['dp'])}$~MW"),
        ("Center invariance", "Shifted state of charge",
         rf"${tex_sci(worst['doffset'])}$"),
        ("Center invariance", r"$\exp(k_\sigma \Delta\bar{\sigma})$",
         rf"${tex_sci(worst['ratio'])}$"),
        ("Equivalent full cycles", r"$\sum_t \max(p_t^s,0)\,\Delta t/\bar{E}$",
         rf"${tex_sci(worst['efc'])}$"),
        ("Rated-hour identity", r"$\bar{P}\,\Delta t\,\eta_{\mathrm{in}}/\bar{E}$",
         rf"${tex_sci(worst['rest'])}$"),
    ]
    return "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", default=None,
                    help="folder holding the week_snapshot_*.csv files")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    indir = resolve_indir(args.indir)

    summary = pd.read_csv(indir / "week_snapshot_summary.csv")
    cycles = pd.read_csv(indir / "week_snapshot_cycles.csv")
    hourly = pd.read_csv(indir / "week_snapshot_hourly.csv")

    audit = Audit()
    out = []
    out.append("=" * 78)
    out.append("WEEK SNAPSHOT AUDIT")
    out.append(f"input folder: {indir}")
    out.append(f"windows: {', '.join(summary['window'])}")
    out.append(f"week: {int(summary['week'].iloc[0])}, {len(hourly)} hourly steps")
    out.append(f"script version: {SCRIPT_VERSION}")
    out.append("=" * 78)

    worst_cyc, worst_cal = check_degradation_recompute(summary, cycles, audit)
    worst_bound, worst_overrun = check_bounds_and_rating(summary, hourly, audit)
    check_record_bookkeeping(summary, cycles, audit)
    check_annual_record_split(summary, indir, audit)
    worst_efc = check_efc(summary, hourly, audit)
    worst_dp, worst_doffset, worst_ratio = check_centre_invariance(summary, hourly, audit)
    rated_rise, worst_rest = check_rated_hour_positioning(hourly, audit)

    out.append(audit.report())
    out.append("")
    out.append("SUPPORTING DETAIL")
    out.append(f"  EFC definition: sum(p[p>0])*dt / E_nom, reproduced to {worst_efc:.1e}")
    out.append(f"  one rated charging hour = {rated_rise:.6f} of nominal capacity")
    out.append(report_shared_events(cycles))
    out.append("")
    out.append("  week statistics (week / annual mean per week):")
    for _, r in summary.iterrows():
        out.append(
            f"    {r['window']:7s} cycles {r.week_cycles:5.1f} / {r.year_cycles_per_week:6.3f}"
            f"  ({100 * (r.week_cycles / r.year_cycles_per_week - 1):+5.1f}%)"
            f"   above delta_c {r.week_cycles_above_dc:5.1f} / {r.year_cycles_above_dc_per_week:6.3f}"
            f"  ({100 * (r.week_cycles_above_dc / r.year_cycles_above_dc_per_week - 1):+5.1f}%)"
        )
    out.append("")
    out.append("OVERALL: " + ("PASS" if audit.ok else "FAIL"))
    out.append("=" * 78)

    report = "\n".join(out)
    print(report)
    (here / "week_audit_report.txt").write_text(report + "\n", encoding="utf-8")

    (here / "tab_week_stats_body.tex").write_text(
        latex_stats_body(summary) + "\n", encoding="utf-8")
    (here / "tab_week_checks_body.tex").write_text(
        latex_checks_body({
            "cyc": worst_cyc, "cal": worst_cal, "bound_mwh": worst_bound,
            "dp": worst_dp, "doffset": worst_doffset, "ratio": worst_ratio,
            "rest": worst_rest, "efc": worst_efc, "overrun": max(worst_overrun, 0.0),
        }) + "\n", encoding="utf-8")

    print(f"\nwrote tab_week_stats_body.tex, tab_week_checks_body.tex, "
          f"week_audit_report.txt to {here}")
    return 0 if audit.ok else 1


if __name__ == "__main__":
    sys.exit(main())