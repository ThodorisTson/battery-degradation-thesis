r"""
verify_tier1_degradation.py  --  Tier 1 verification of the degradation chain

WHAT THIS IS
------------
Verification, not validation. It checks that the rainflow counter and the Xu and Shi fade models are implemented correctly, by running them on synthetic
state-of-charge traces whose correct answer is known before the code is run. It says nothing about whether the models describe a real battery; no measured
cell data is used anywhere in this thesis.

No dispatch, no LP, no price or wind data. Runs in about a second.

DESIGN RULE
-----------
Every expected value on the left of a comparison is computed from the published constants written out again at the top of this file. It never calls the
function under test. If the expected column called s_dod() the test could not fail, because both sides would share the same bug.

The rainflow expectations are checked twice: against hand-derived cycle lists written as literals, and against an independent ASTM E1049 counter implemented
below, which is itself checked against the worked example in the standard.

TESTS
-----
  T1.0  reference counter reproduces the ASTM E1049 worked example
  T1.1  constant-amplitude trace: record structure, depths, means, total count
  T1.2  nested small cycles on a large excursion: the small cycles are resolved
  T1.3  degradation.xu and degradation.shi counters agree exactly
  T1.4  Xu cycle and calendar terms against closed form
  T1.5  calendar isolated on a flat trace: rate, mean-SoC factor, duration
  T1.6  linearity: tripling the number of cycles triples the cycle term
  T1.7  Shi surrogate: refit reproduces stored constants, closed-form fade,
        and the extrapolation gap below the fitting range

OUTPUT
------
  console table of expected vs obtained
  Verification/tier1_verification.csv
  Verification/tier1_verification_table.tex   (ready to \input in Chapter 4)
  fig_verify_rainflow_readback.pdf / .png, written beside this script

Reproducible on Windows / VS Code: numpy, pandas, matplotlib, rainflow.
Run from anywhere in the repository; imports resolve through the degradation
package. Produces Figure D.1.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from degradation.style import apply_thesis_style, figsize, TUDELFT, FS_ANNOT, FS_LEGEND

HERE = Path(__file__).resolve().parent

# -- Output ------------------------------------------------------------------ #
OUTPUT = "both"     # "png", "pdf" or "both"
DPI = 300

# --- code under test ---------------------------------------------------------
from degradation.xu import (
    rainflow_cycle_counting as rainflow_xu,
    compute_fd,
    XU_LMO,
)
from degradation.shi import (
    rainflow_cycle_counting as rainflow_shi,
    fit_shi_polynomial,
    compute_fd_shi,
    ShiModelParams,
)

OUT_DIR = HERE

# =============================================================================
# 1.  Reference constants and stress factors, written out independently
#     Source: Xu et al. (2016), Table I, LMO. Do not import these from
#     degradation.xu -- that is the point of the exercise.
# =============================================================================
K_DELTA1, K_DELTA2, K_DELTA3 = 1.40e5, -5.01e-1, -1.23e5   # Eq. 32
K_SIGMA, SIGMA_REF           = 1.04, 0.50                  # Eq. 25
K_T_ARRH, T_REF_C            = 6.93e-2, 25.0               # Eq. 22
K_T_CAL                      = 4.14e-10                    # Eq. 27, per second


def ref_s_delta(delta):
    """S_delta(delta) = 1 / (k1*delta^k2 + k3)."""
    return 1.0 / (K_DELTA1 * np.asarray(delta, float) ** K_DELTA2 + K_DELTA3)


def ref_s_sigma(sigma):
    """S_sigma(sigma) = exp(k_sigma * (sigma - sigma_ref))."""
    return np.exp(K_SIGMA * (np.asarray(sigma, float) - SIGMA_REF))


def ref_s_temp(T_C):
    """S_T(T) = exp(k_T * (T - T_ref) * T_ref / T), temperatures in kelvin."""
    T_K, T_ref_K = np.asarray(T_C, float) + 273.15, T_REF_C + 273.15
    return np.exp(K_T_ARRH * (T_K - T_ref_K) * T_ref_K / T_K)


def ref_f_calendar(t_seconds, sigma_mean, T_C=25.0):
    """f_cal = k_t * t * S_sigma(sigma_mean) * S_T(T)."""
    return K_T_CAL * float(t_seconds) * ref_s_sigma(sigma_mean) * ref_s_temp(T_C)


# =============================================================================
# 2.  Independent ASTM E1049 rainflow counter (the second opinion)
# =============================================================================
def _ref_reversals(series):
    it = iter(series)
    x_last, x = next(it), next(it)
    d_last = x - x_last
    yield 0, x_last
    index = 1
    for x_next in it:
        if x_next == x:
            index += 1
            continue
        d_next = x_next - x
        if d_last * d_next < 0:
            yield index, x
        x_last, x = x, x_next
        d_last = d_next
        index += 1
    yield index, x


def _ref_emit(p1, p2, count):
    (i1, x1), (i2, x2) = p1, p2
    return abs(x2 - x1), 0.5 * (x1 + x2), count, i1, i2


def ref_extract_cycles(series):
    """ASTM E1049 three-point counting, residue closed as half cycles."""
    pts = deque()
    for point in _ref_reversals(series):
        pts.append(point)
        while len(pts) >= 3:
            x1, x2, x3 = pts[-3][1], pts[-2][1], pts[-1][1]
            X, Y = abs(x3 - x2), abs(x2 - x1)
            if X < Y:
                break
            if len(pts) == 3:
                yield _ref_emit(pts[0], pts[1], 0.5)
                pts.popleft()
            else:
                yield _ref_emit(pts[-3], pts[-2], 1.0)
                last = pts.pop()
                pts.pop()
                pts.pop()
                pts.append(last)
    while len(pts) > 1:
        yield _ref_emit(pts[0], pts[1], 0.5)
        pts.popleft()


# =============================================================================
# 3.  Synthetic traces
# =============================================================================
E_CAP   = 300.0        # MWh, the reference battery
SOC_MIN = 0.10
SOC_MAX = 0.90
DT_H    = 1.0
T_CELL  = 25.0

E_LO, E_HI = SOC_MIN * E_CAP, SOC_MAX * E_CAP      # 30, 270 MWh


def triangular_trace(n_periods: int, n_ramp: int = 6) -> np.ndarray:
    """n_periods identical full cycles between the window bounds.

    Each period ramps E_LO -> E_HI -> E_LO in 2*n_ramp samples. The trace closes
    at E_LO so every cycle has depth exactly SOC_MAX - SOC_MIN and mean exactly
    (SOC_MAX + SOC_MIN)/2.
    """
    up = np.linspace(E_LO, E_HI, n_ramp + 1)[:-1]
    dn = np.linspace(E_HI, E_LO, n_ramp + 1)[:-1]
    period = np.concatenate([up, dn])
    return np.concatenate([np.tile(period, n_periods), [E_LO]])


# Turning points of the nested trace, in MWh. Four ripples of 30 MWh depth
# (delta = 0.10) sit on the charging leg of one 30 -> 270 -> 30 excursion.
NESTED_TURNING_POINTS = [30, 90, 60, 120, 90, 150, 120, 180, 150, 270, 30]


def nested_trace(n_interp: int = 3) -> np.ndarray:
    """One large excursion carrying four small ripples, linearly interpolated.

    Interpolation adds points between turning points so the trace looks like an
    hourly series; rainflow reduces to the turning points, so the counted result
    is unchanged by n_interp. That invariance is itself checked in T1.2.
    """
    out = []
    for a, b in zip(NESTED_TURNING_POINTS[:-1], NESTED_TURNING_POINTS[1:]):
        out.extend(np.linspace(a, b, n_interp + 1)[:-1])
    out.append(NESTED_TURNING_POINTS[-1])
    return np.asarray(out, float)


def flat_trace(sigma: float, n_hours: int) -> np.ndarray:
    """Constant state of charge: no reversals, so no cycle aging."""
    return np.full(n_hours, sigma * E_CAP, dtype=float)


# =============================================================================
# 4.  Result collection
# =============================================================================
CHECKS: list[dict] = []


def check(tid: str, quantity: str, expected, obtained, unit: str = "",
          rtol: float = 1e-12, atol: float = 0.0, note: str = "") -> None:
    """Record one expected-vs-obtained comparison and print nothing yet."""
    exp = float(expected)
    obt = float(obtained)
    denom = abs(exp) if abs(exp) > 0 else 1.0
    rel = abs(obt - exp) / denom
    ok = bool(np.isclose(obt, exp, rtol=rtol, atol=atol))
    CHECKS.append({
        "test": tid, "quantity": quantity, "unit": unit,
        "expected": exp, "obtained": obt, "rel_error": rel,
        "rtol": rtol, "status": "PASS" if ok else "FAIL", "note": note,
    })


def cycles_as_tuples(cycles, ndigits: int = 9):
    """(depth_MWh, mean_MWh, count) triples, rounded, for exact comparison."""
    return [(round(c["depth_MWh"], ndigits),
             round(c["mean_MWh"], ndigits),
             round(c["count"], ndigits)) for c in cycles]


# =============================================================================
# T1.0  the reference counter reproduces the ASTM E1049 worked example
# =============================================================================
def test_reference_counter() -> None:
    series = [-2, 1, -3, 5, -1, 3, -4, 4, -2]
    agg: dict[float, float] = {}
    for rng, _mean, cnt, _i, _j in ref_extract_cycles(series):
        agg[rng] = agg.get(rng, 0.0) + cnt
    got = sorted(agg.items())
    # ASTM E1049-85 worked example, as tabulated in the rainflow package docs
    expected = [(3.0, 0.5), (4.0, 1.5), (6.0, 0.5), (8.0, 1.0), (9.0, 0.5)]
    for (r_e, c_e), (r_g, c_g) in zip(expected, got):
        check("T1.0", f"count at range {r_e:.0f}", c_e, c_g, "cycles")
    check("T1.0", "number of distinct ranges", len(expected), len(got), "-")


# =============================================================================
# T1.1  constant-amplitude trace
# =============================================================================
def test_constant_amplitude(n_periods: int = 10) -> list[dict]:
    trace = triangular_trace(n_periods)
    cycles = rainflow_xu(trace, E_CAP)

    # Hand-derived expectation. The three-point algorithm never grows the
    # residue past three points on a constant-amplitude sequence, so it emits
    # 2*n half-cycle records rather than (n-1) full plus 2 half.
    check("T1.1", "rainflow records", 2 * n_periods, len(cycles), "records")
    check("T1.1", "total cycle count", n_periods,
          sum(c["count"] for c in cycles), "cycles")
    check("T1.1", "cycle depth (all records)", SOC_MAX - SOC_MIN,
          max(c["dod"] for c in cycles), "-")
    check("T1.1", "cycle depth spread", 0.0,
          max(c["dod"] for c in cycles) - min(c["dod"] for c in cycles), "-",
          atol=1e-15)
    check("T1.1", "cycle mean SoC (all records)", 0.5 * (SOC_MAX + SOC_MIN),
          max(c["soc_mean"] for c in cycles), "-")
    check("T1.1", "record count of every entry", 0.5,
          max(c["count"] for c in cycles), "cycles")

    # cross-check against the independent counter, record by record
    ref = [(round(r, 9), round(m, 9), round(c, 9))
           for r, m, c, _i, _j in ref_extract_cycles(trace)]
    check("T1.1", "records matching independent counter",
          len(ref), sum(1 for a, b in zip(ref, cycles_as_tuples(cycles)) if a == b),
          "records", note="independent ASTM implementation")
    return cycles


# =============================================================================
# T1.2  nested small cycles
# =============================================================================
def test_nested_small_cycles() -> list[dict]:
    trace = nested_trace()
    cycles = rainflow_xu(trace, E_CAP)

    # Hand-derived from the turning points:
    #   four full cycles of depth 30 MWh at means 75, 105, 135, 165 MWh
    #   two half cycles of depth 240 MWh at mean 150 MWh
    expected_small = [(0.10, 0.25), (0.10, 0.35), (0.10, 0.45), (0.10, 0.55)]
    small = sorted([(round(c["dod"], 9), round(c["soc_mean"], 9))
                    for c in cycles if c["count"] == 1.0])
    large = [c for c in cycles if c["count"] == 0.5]

    check("T1.2", "full cycles resolved", 4, len(small), "cycles")
    check("T1.2", "half cycles (large excursion)", 2, len(large), "records")
    check("T1.2", "total cycle count", 5.0,
          sum(c["count"] for c in cycles), "cycles")
    for k, ((d_e, s_e), (d_g, s_g)) in enumerate(zip(expected_small, small), 1):
        check("T1.2", f"small cycle {k} depth", d_e, d_g, "-")
        check("T1.2", f"small cycle {k} mean SoC", s_e, s_g, "-")
    check("T1.2", "large excursion depth", SOC_MAX - SOC_MIN,
          large[0]["dod"], "-")
    check("T1.2", "large excursion mean SoC", 0.5 * (SOC_MAX + SOC_MIN),
          large[0]["soc_mean"], "-")

    # the counted result must not depend on how finely the ramps are sampled
    coarse = rainflow_xu(np.asarray(NESTED_TURNING_POINTS, float), E_CAP)
    same = sum(1 for a, b in zip(
        sorted([(round(c["dod"], 9), round(c["soc_mean"], 9), c["count"]) for c in coarse]),
        sorted([(round(c["dod"], 9), round(c["soc_mean"], 9), c["count"]) for c in cycles]))
        if a == b)
    check("T1.2", "records invariant to ramp sampling", len(coarse), same,
          "records", note="turning points only vs interpolated")
    return cycles


# =============================================================================
# T1.3  the two module copies of the counter agree
# =============================================================================
def test_counters_agree() -> None:
    for name, trace in [("constant amplitude", triangular_trace(10)),
                        ("nested", nested_trace())]:
        a = cycles_as_tuples(rainflow_xu(trace, E_CAP))
        b = cycles_as_tuples(rainflow_shi(trace, E_CAP))
        check("T1.3", f"identical records, {name}", len(a),
              sum(1 for x, y in zip(a, b) if x == y), "records",
              note="degradation.xu vs degradation.shi")


# =============================================================================
# T1.4  Xu fade against closed form
# =============================================================================
def test_xu_closed_form(n_periods: int = 10) -> None:
    trace = triangular_trace(n_periods)
    n_steps = len(trace)
    t_seconds = n_steps * DT_H * 3600.0
    sigma_mean = float(np.mean(trace)) / E_CAP

    cycles = rainflow_xu(trace, E_CAP)
    fd, fd_cycle, fd_cal = compute_fd(cycles, sigma_mean, t_seconds, T_CELL, XU_LMO)

    # every cycle has the same depth and mean, so the sum collapses:
    #   f_cyc = n * S_delta(0.80) * S_sigma(0.50) * S_T(25)
    exp_cycle = n_periods * ref_s_delta(SOC_MAX - SOC_MIN) \
        * ref_s_sigma(0.5 * (SOC_MAX + SOC_MIN)) * ref_s_temp(T_CELL)
    exp_cal = ref_f_calendar(t_seconds, sigma_mean, T_CELL)

    check("T1.4", "S_delta(0.80)", 2.9797653244e-05,
          ref_s_delta(0.80), "-", rtol=1e-9, note="literal, checked by hand")
    check("T1.4", "f_d cycle term", exp_cycle, fd_cycle, "-")
    check("T1.4", "f_d calendar term", exp_cal, fd_cal, "-")
    check("T1.4", "f_d total", exp_cycle + exp_cal, fd, "-")
    check("T1.4", "S_T at 25 C", 1.0, ref_s_temp(25.0), "-")


# =============================================================================
# T1.5  calendar isolated on a flat trace
# =============================================================================
def test_calendar_isolated() -> None:
    hours = 720
    t_seconds = hours * DT_H * 3600.0

    for sigma in (0.50, 0.70):
        trace = flat_trace(sigma, hours)
        cycles = rainflow_xu(trace, E_CAP)
        fd, fd_cycle, fd_cal = compute_fd(cycles, sigma, t_seconds, T_CELL, XU_LMO)
        # a flat trace has no reversals; the residue closes as one zero-depth
        # half cycle, which the clipped S_delta values at ~7e-9, not exactly 0
        check("T1.5", f"cycle term at sigma={sigma:.2f}", 0.0, fd_cycle, "-",
              atol=1e-8, note="flat trace, no reversals")
        check("T1.5", f"calendar term at sigma={sigma:.2f}",
              ref_f_calendar(t_seconds, sigma, T_CELL), fd_cal, "-")

    # mean-SoC factor: the ratio must be exp(k_sigma * 0.20) exactly
    _, _, cal_50 = compute_fd(rainflow_xu(flat_trace(0.50, hours), E_CAP),
                              0.50, t_seconds, T_CELL, XU_LMO)
    _, _, cal_70 = compute_fd(rainflow_xu(flat_trace(0.70, hours), E_CAP),
                              0.70, t_seconds, T_CELL, XU_LMO)
    check("T1.5", "calendar ratio sigma 0.70 / 0.50",
          np.exp(K_SIGMA * 0.20), cal_70 / cal_50, "-")

    # duration: the calendar term is linear in elapsed time
    t2 = 2 * hours * DT_H * 3600.0
    _, _, cal_2x = compute_fd(rainflow_xu(flat_trace(0.50, 2 * hours), E_CAP),
                              0.50, t2, T_CELL, XU_LMO)
    check("T1.5", "calendar ratio 1440 h / 720 h", 2.0, cal_2x / cal_50, "-")


# =============================================================================
# T1.6  linearity of the cycle term
# =============================================================================
def test_cycle_linearity(n_small: int = 10, factor: int = 3) -> None:
    def fd_cycle_of(n):
        trace = triangular_trace(n)
        t_s = len(trace) * DT_H * 3600.0
        s_mean = float(np.mean(trace)) / E_CAP
        return compute_fd(rainflow_xu(trace, E_CAP), s_mean, t_s, T_CELL, XU_LMO)[1]

    small = fd_cycle_of(n_small)
    large = fd_cycle_of(factor * n_small)
    check("T1.6", f"f_cyc ratio {factor * n_small} / {n_small} periods",
          float(factor), large / small, "-",
          note="identical cycles, so the sum is linear in their number")


# =============================================================================
# T1.7  Shi surrogate
# =============================================================================
def test_shi_surrogate(n_periods: int = 10) -> None:
    fit = fit_shi_polynomial(soc_min=SOC_MIN, soc_max=SOC_MAX, verbose=False)
    p = ShiModelParams.from_fit(fit)

    # stored constants from config/battery.yaml runs, reproduced by the refit
    check("T1.7", "k3 from refit", 3.241779477729208e-05, fit.k3, "-", rtol=1e-9)
    check("T1.7", "k4 from refit", 1.178528617001297, fit.k4, "-", rtol=1e-9)
    check("T1.7", "convexity margin k4 - 1", 0.178528617001297, fit.k4 - 1.0,
          "-", rtol=1e-9, note="k4 > 1 required by Shi Theorem 1")
    check("T1.7", "S_sigma at the window centre", 1.0, ref_s_sigma(0.50), "-",
          note="10-90% window puts sigma_bar at sigma_ref exactly")

    trace = triangular_trace(n_periods)
    cycles = rainflow_shi(trace, E_CAP)
    fd_shi, _ = compute_fd_shi(cycles, p, T_CELL)
    exp_shi = n_periods * fit.k3 * (SOC_MAX - SOC_MIN) ** fit.k4 \
        * ref_s_sigma(0.5 * (SOC_MAX + SOC_MIN)) * ref_s_temp(T_CELL)
    check("T1.7", "f_d Shi, cycle only", exp_shi, fd_shi, "-")

    # Surrogate error inside and outside the fitting range [0.15, max_dod].
    # Expected values are the literals obtained from the Xu constants and the
    # log-log fit; they are regression values, so a change in either the fit
    # range or the Xu coefficients will show up here.
    for delta, expected, tag in [
        (0.80, 0.8363540920, "inside fit range [0.15, 0.80]"),
        (0.10, 0.6892983344, "BELOW fit range, extrapolation"),
    ]:
        ratio = float(fit.k3 * delta ** fit.k4) / float(ref_s_delta(delta))
        check("T1.7", f"Phi_shi / S_delta at delta={delta:.2f}",
              expected, ratio, "-", rtol=1e-8, note=tag)


# =============================================================================
# 5.  Figure: rainflow read-back
# =============================================================================
def make_figure(cyc_tri, cyc_nested, out_dir: Path) -> None:
    pal = apply_thesis_style(palette="brand", usetex=False)
    c_trace, c_full, c_half = TUDELFT["navy"], TUDELFT["darkred"], TUDELFT["blue"]

    fig, axes = plt.subplots(2, 1, figsize=figsize(1.0, aspect=0.72), sharex=False)

    # (a) constant amplitude, first three periods
    tri = triangular_trace(10)[:37]
    ax = axes[0]
    ax.plot(np.arange(len(tri)), tri / E_CAP, color=c_trace, lw=1.2)
    for lvl in (SOC_MIN, SOC_MAX):
        ax.axhline(lvl, color=pal["neutral"], lw=0.7, ls=(0, (5, 4)), alpha=0.8)
    ax.set_ylabel("State of charge  (-)")
    ax.set_xlabel("Time step  (h)")
    ax.set_ylim(0.0, 1.0)
    ax.text(0.5, 0.04, r"20 records, all $\delta = 0.80$, total count 10",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=FS_ANNOT,
            color=pal["neutral"])

    # (b) nested, with the extracted cycles drawn as vertical depth bars
    nst = nested_trace()
    ax = axes[1]
    ax.plot(np.arange(len(nst)), nst / E_CAP, color=c_trace, lw=1.2, zorder=2)
    for lvl in (SOC_MIN, SOC_MAX):
        ax.axhline(lvl, color=pal["neutral"], lw=0.7, ls=(0, (5, 4)), alpha=0.8)
    for c in cyc_nested:
        x = 0.5 * (c["i_start"] + c["i_end"])
        half = 0.5 * c["dod"]
        col = c_full if c["count"] == 1.0 else c_half
        ax.plot([x, x], [c["soc_mean"] - half, c["soc_mean"] + half],
                color=col, lw=1.4, marker="_", ms=5, zorder=3)
    ax.set_ylabel("State of charge  (-)")
    ax.set_xlabel("Time step  (h)")
    ax.set_ylim(0.0, 1.0)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color=c_trace, lw=1.2, label="Synthetic SoC trace"),
        Line2D([0], [0], color=c_full, lw=1.4, label=r"Full cycle, $\delta = 0.10$"),
        Line2D([0], [0], color=c_half, lw=1.4, label=r"Half cycle, $\delta = 0.80$"),
    ], frameon=False, fontsize=FS_LEGEND, loc="upper left", ncol=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    if OUTPUT in ("pdf", "both"):
        fig.savefig(out_dir / "fig_verify_rainflow_readback.pdf")
    if OUTPUT in ("png", "both"):
        fig.savefig(out_dir / "fig_verify_rainflow_readback.png", dpi=DPI)
    plt.close(fig)


# =============================================================================
# 6.  Reporting
# =============================================================================
def report(out_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(CHECKS)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "tier1_verification.csv", index=False)

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
        print(f"{r['test']:<6}{r['quantity']:<{width}}"
              f"{r['expected']:>16.8g}{r['obtained']:>16.8g}"
              f"{r['rel_error']:>12.2e}  {r['status']}")
    print("=" * (width + 58))
    n_fail = int((df["status"] == "FAIL").sum())
    print(f"{len(df)} checks, {len(df) - n_fail} passed, {n_fail} failed")

    # LaTeX fragment for Chapter 4
    rows = []
    for r in CHECKS:
        q = r["quantity"].replace("_", r"\_").replace("%", r"\%")
        rows.append(f"{r['test']} & {q} & {r['expected']:.6g} & "
                    f"{r['obtained']:.6g} & {r['rel_error']:.1e} & {r['status']} \\\\")
    tex = ("\\begin{tabular}{llrrrl}\n\\hline\\noalign{\\smallskip}\n"
           "Test & Quantity & Expected & Obtained & Rel. error & Status \\\\\n"
           "\\noalign{\\smallskip}\\hline\\noalign{\\smallskip}\n"
           + "\n".join(rows)
           + "\n\\noalign{\\smallskip}\\hline\n\\end{tabular}\n")
    (out_dir / "tier1_verification_table.tex").write_text(tex, encoding="utf-8")

    if n_fail:
        raise SystemExit(f"{n_fail} verification check(s) failed.")
    return df


def _check_output() -> None:
    if OUTPUT not in ("png", "pdf", "both"):
        raise ValueError(f'OUTPUT must be "png", "pdf" or "both", not {OUTPUT!r}')


def main() -> None:
    print("Tier 1 verification: rainflow counting and Xu/Shi fade on synthetic traces")
    print(f"  battery      : {E_CAP:.0f} MWh, SoC window {SOC_MIN:.0%}-{SOC_MAX:.0%}")
    print(f"  temperature  : {T_CELL:.0f} C  (S_T = 1 by construction)")

    test_reference_counter()
    cyc_tri = test_constant_amplitude()
    cyc_nested = test_nested_small_cycles()
    test_counters_agree()
    test_xu_closed_form()
    test_calendar_isolated()
    test_cycle_linearity()
    test_shi_surrogate()

    make_figure(cyc_tri, cyc_nested, OUT_DIR)
    report(OUT_DIR)
    print(f"\nwrote CSV, LaTeX table and figure to {OUT_DIR}")


if __name__ == "__main__":
    main()