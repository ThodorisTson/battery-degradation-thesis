"""
Verification of the exact rainflow sub-gradient.   (revision of 29 July 2026)
===========================================================================

Verifies the spanning-set sub-gradient of degradation/subgradient.py, which implements Chapter 2 Equations eq:subgrad_charge and eq:subgrad_discharge. This
is the per-timestep leg of the gradient verification. The capacity gradient is the aggregate leg and is checked by verify_gradient_finite_difference.py; a
single scalar per year cannot see how the sensitivity is distributed over the
horizon, which is the question answered here.

Reads the .npy pair from  Results/RTE Tests/ , which is where run_battery_xu_shi_degradation_v5_6_RTE_test.py writes them (RESULTS_DIR, line 134).
Do NOT point this at Results/ : the pair sitting there is left over from an older routing of the script, and its two files came from different runs.

Checks
------
  GUARD    Are storage_e and e_cap from the same solve? If the trace never reaches soc_max, the two files are out of sync and every normalized depth below
           is wrong. Checked first and loudly, because a stale e_cap silently rescales everything.

  TEST A   Exactness at a reference step, split into two populations:
             smooth : forward and backward differences agree, so the cost is differentiable at that time step.
             kink   : they disagree, so the rainflow map changes combinatorial structure under the perturbation and the cost is not
                      differentiable. A central difference returns the AVERAGE of the two one-sided derivatives and cannot match any
                      sub-gradient. The analytic value must instead coincide with ONE of them, which is what makes it a valid element of the
                      sub-differential. TEST A2 checks that directly.

           Reporting one median against a central difference conflates the two and hides failures. That is how the attribution bug survived: median
           0.000 percent, max 1512 percent.

  TEST A2  Sub-differential bracket at the kinks. Is the analytic value inside [min, max] of the two one-sided derivatives, and does it sit at an
           endpoint rather than in the interior.

  TEST A3  Step-size sweep on the time steps that stay smooth at every step. At one fixed step the residual is dominated by the truncation error of
           the difference, not by any error in the analytic value, so a single number cannot support a claim of machine precision. Same logic as the
           capacity-gradient sweep, applied per timestep.

  TEST B   Efficiency identity, pointwise. |df/dd_t| / |df/dc_t| = 1/(eta_in*eta_out) at EVERY timestep, exactly, because G(t) is common to both. A ratio of
           MEANS over the horizon is not a valid test: it recovers the coefficient ratio only when the charging and discharging depth distributions coincide,
           which is why it read exactly 1/eta_out while the attribution map was collapsed onto a single cycle.

  TEST C   Why S_sigma is excluded from the gradient path. Depth-only analytic against a central difference of the S_sigma-weighted cost, on the
           smooth population only, because a central difference at a kink is not a derivative. The gap is the non-local mean-SoC term. Chapter 2
           subsec:depth_only claims more than two orders of magnitude and cites this section for the number.

Changes in this revision
------------------------
  1. degradation/style.py for fonts, colours and figure width. The previous version hardcoded three hex values and used tight_layout with bbox_inches="tight",
     the opposite of the savefig.bbox = "standard" convention, so its figure was the only one in the thesis at a different scale.
  2. TEST A3 replaces the printed claim "must be machine precision", which sat next to a median of 1.79e-05 % at a fixed 1e-6 step. That is truncation
     error of the difference, not an error in the analytic value. 
  3. Percentiles and a robust error scale. The previous denominator was max(|central difference|, 1e-9), so any step where the derivative passes
     through zero inflated without limit and produced a 4.5e8 relative error. The magnitude of the derivative at that step is now printed alongside, and
     the old normalisation is still computed so the artefact can be reported and dismissed rather than found by someone else.
  4. TEST A2, the bracket test.
  5. TEST C restricted to the smooth population.
  6. Summary CSV, as every other verification artefact in the thesis has. 
  7. Module-family assertion. degradation.xu and degradation.shi both define fit_shi_polynomial, phi_shi and phi_shi_prime. degradation.subgradient
     imports phi_shi_prime from the Xu copy while the cost here uses phi_shi from the Shi copy. They agree today and nothing enforces it. Note their
     positional signatures differ, so every call here uses keywords. 
  8. The SoC window is read from config/battery.yaml when it can be found, and RTE_AC is the value the run script derives rather than a rounded 0.910.
  9. Figures. The old left panel plotted the central difference against the analytic value for both populations, which shows 97 points far off the
     diagonal at the kinks. Those points are meaningless by the section's own argument, so the panel argued against the result. It is replaced by the
     step-size sweep and by a bracket figure that makes the sub-differential claim visible. Say the word and the parity panel comes back.

Reads results/baseline/xu/ through degradation.paths. Figures are written beside this script. Produces Figure C.3, fig_subgrad_bracket_dk2019 and _dk2022.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from degradation.paths import RESULTS_DIR as REPO_RESULTS_DIR, BATTERY_YAML
from degradation.style import apply_thesis_style, figsize, FS_ANNOT, FS_LEGEND

from degradation.xu import (
    XuModelParams,
    XU_LMO,
    fit_shi_polynomial as fit_xu,
    phi_shi_prime as phi_prime_xu,
)
from degradation.shi import (
    rainflow_cycle_counting,
    phi_shi,
    phi_shi_prime as phi_prime_shi,
    fit_shi_polynomial as fit_shi,
    load_soc_window_from_yaml,
    s_soc,
    s_temp,
)
from degradation.subgradient import compute_subgradient

# ------------------------------- configuration --------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPO_RESULTS_DIR / "baseline" / "xu"
OUT_DIR     = SCRIPT_DIR

# Round-trip efficiency as the run script derives it: 0.95 * 0.978721**2.
# The previous 0.910 put the printed identity 6.8e-8 away from the value used
# everywhere else in the thesis.
RTE_AC   = 0.9100000560489498
ETA      = float(np.sqrt(RTE_AC))
ETA_IN   = ETA_OUT = ETA

DT_HOURS = 1.0
B_REPL   = 72.0 * 1000.0        # EUR/MWh, the replacement-energy rate (72 EUR/kWh)
T_CELL_C = 25.0

H_REF      = 1e-6                        # reference step for the smooth / kink split
H_LIST     = np.logspace(-4, -11, 15)    # step-size sweep, fractional SoC
N_SAMPLE   = 200                         # charging time steps in the main population
N_SWEEP    = 25                          # time steps carried through the step sweep
N_BRACKET  = 48                          # ordered subset; the table carries all 97
SMOOTH_TOL = 1e-6                        # |fwd - bwd| / scale below this is smooth
REL_FLOOR  = 1e-3                        # robust scale, as a fraction of median |analytic|
DPI        = 300

FIG_FRAC_WIDE   = 1.00          # \includegraphics[width=\textwidth]
FIG_FRAC_SINGLE = 0.70


# =============================================================================
# Data
# =============================================================================
def find_yaml():
    return BATTERY_YAML if BATTERY_YAML.exists() else None


def load_window():
    """SoC window from config/battery.yaml, with the fallback kept audible.

    The fallback happens to equal the configured window, so a missing file
    would otherwise be invisible. The returned source string says which was
    used and is printed by the caller.
    """
    y = find_yaml()
    if y is None:
        return 0.10, 0.90, f"fallback [0.10, 0.90], {BATTERY_YAML} not found"
    lo, hi, src = load_soc_window_from_yaml(y)
    if not (0.0 < lo < hi <= 1.0):
        return 0.10, 0.90, f"fallback [0.10, 0.90], unusable window from {src}"
    return lo, hi, src


def discover_cases(results_dir: Path):
    """Newest multiyear rte910 .npy per priceset, paired with its npv_summary CSV.

    Same discovery as verify_gradient_finite_difference_v2.py, so the two tests read the same source. Reading annual_soc[0] rather than storage_e_fixed.npy
    makes both price years available without re-running the v5.6 script, which writes only one fixed pair at a time.
    """
    print(f"[discover] results folder : {results_dir}")
    if not results_dir.is_dir():
        raise SystemExit(f"[discover] folder does not exist: {results_dir}")

    all_npy = sorted(results_dir.glob("multiyear_*rte910*.npy"))
    if not all_npy:
        raise SystemExit(f"[discover] no 'multiyear_*rte910*.npy' files in {results_dir}")

    by_label = {}
    for npy in all_npy:
        stem = npy.name[len("multiyear_"):-len(".npy")]
        label = "dk2019" if "dk2019" in stem else "dk2022" if "dk2022" in stem else stem
        by_label.setdefault(label, []).append(npy)

    cases = []
    for label in sorted(by_label):
        group = by_label[label]
        chosen = max(group, key=lambda q: q.name)
        stem = chosen.name[len("multiyear_"):-len(".npy")]
        csv_hits = list(results_dir.glob(f"npv_summary_{stem}.csv"))
        if not csv_hits:
            print(f"[discover] {label}: SKIP (no npv_summary matching {chosen.name})")
            continue
        print(f"[discover] {label}: {chosen.name}")
        for other in group:
            if other is not chosen:
                print(f"[discover]         (ignoring older {other.name})")
        cases.append((label, chosen, csv_hits[0]))
    return cases


def load_case(npy_path: Path, csv_path: Path):
    d = np.load(npy_path, allow_pickle=True).item()
    trace = np.asarray(d["annual_soc"][0], dtype=float)
    e_cap = float(d["e_cap_nominal"])
    par = pd.read_csv(csv_path)
    soc_min = float(par["soc_min"].iloc[0])
    soc_max = float(par["soc_max"].iloc[0])
    src = f"{npy_path.name}  ({len(trace)} steps)"
    return trace, e_cap, soc_min, soc_max, src


def guard(e, e_cap, soc_min, soc_max) -> bool:
    """Refuse to trust the run if the trace and the capacity disagree."""
    lo, hi = e.min() / e_cap, e.max() / e_cap
    print(f"\n{'-'*74}\nGUARD  Are storage_e and e_cap from the same solve?\n{'-'*74}")
    print(f"  e_cap             : {e_cap:.1f} MWh")
    print(f"  SoC range implied : {lo:.4f} to {hi:.4f}")
    print(f"  SoC window        : {soc_min:.2f} to {soc_max:.2f}")
    ok = (hi > soc_max - 0.02) and (lo < soc_min + 0.02)
    if not ok:
        implied = e.max() / soc_max
        print("\n  *** INCONSISTENT ***")
        print(f"  The trace never reaches soc_max. An e_cap of {implied:.1f} MWh")
        print(f"  would put the range at [{e.min()/implied:.3f}, {e.max()/implied:.3f}],")
        print("  which matches the window. The two files are from different runs.")
        print(f"  Every normalized depth below is scaled by {implied/e_cap:.3f}.\n")
    else:
        print("  consistent\n")
    return ok


def check_module_families(soc_min, soc_max) -> None:
    fx = fit_xu(soc_min=soc_min, soc_max=soc_max, source="consistency")
    fs = fit_shi(soc_min=soc_min, soc_max=soc_max, source="consistency")
    dk3 = abs(fx.k3 - fs.k3) / abs(fx.k3)
    dk4 = abs(fx.k4 - fs.k4) / abs(fx.k4)
    d = np.linspace(0.01, 1.0, 200)
    a = phi_prime_xu(d, fx.k3, fx.k4)
    b = phi_prime_shi(d, fs.k3, fs.k4)
    dphi = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
    print(f"{'-'*74}\nMODULE CONSISTENCY  degradation.xu vs degradation.shi\n{'-'*74}")
    print(f"  fit_shi_polynomial : dk3 {dk3:.2e}   dk4 {dk4:.2e}")
    print(f"  phi_shi_prime      : max relative difference {dphi:.2e}")
    if max(dk3, dk4, dphi) > 1e-12:
        raise SystemExit("the two module copies disagree; fix before trusting anything below")
    print("  the two copies agree\n")


# =============================================================================
# Cost and one-sided differences
# =============================================================================
def deg_cost(e, e_cap, k3, k4, p: XuModelParams) -> float:
    """f = E * B * sum_i n_i * Phi(delta_i) * S_sigma(sigma_i) * S_T(T).

    Passing p with k_sigma = 0 and k_T = 0 gives the depth-only cost that Shi
    et al. prove convex and that the gradient path differentiates.
    """
    c = rainflow_cycle_counting(e, e_cap)
    if not c:
        return 0.0
    d = np.array([x["dod"] for x in c])
    q = np.array([x["count"] for x in c])
    s = np.array([x["soc_mean"] for x in c])
    stress = s_soc(s, p.k_sigma, p.sigma_ref) * s_temp(T_CELL_C, p.k_T, p.T_ref_C)
    return float(np.sum(q * phi_shi(d, k3, k4) * stress) * e_cap * B_REPL)


def one_sided(e, e_cap, t, k3, k4, p, f0, h_soc):
    """Forward and backward differences of f with respect to c_t.

    Raising c_t by h raises the energy trace by dt*eta_in*h at every step after t.
    The step is expressed as a fractional SoC shift so it is dimensionless.
    """
    de = h_soc * e_cap
    h = de / (DT_HOURS * ETA_IN)
    ep = e.copy(); ep[t + 1:] += de
    em = e.copy(); em[t + 1:] -= de
    fwd = (deg_cost(ep, e_cap, k3, k4, p) - f0) / h
    bwd = (f0 - deg_cost(em, e_cap, k3, k4, p)) / h
    return fwd, bwd


def robust_scale(analytic, central):
    """Error denominator that does not blow up where the derivative crosses zero."""
    floor = REL_FLOOR * float(np.median(np.abs(analytic))) if analytic.size else 1e-12
    return np.maximum.reduce([np.abs(central), np.abs(analytic),
                              np.full_like(central, max(floor, 1e-12))])


def pct(x, q):
    return float(np.percentile(x, q)) if np.size(x) else float("nan")


# =============================================================================
# Main
# =============================================================================
def run_case(label, e, e_cap, soc_min, soc_max, src, pal) -> dict:
    """One price year end to end. Returns the summary row."""
    n = len(e)

    fit = fit_xu(soc_min=soc_min, soc_max=soc_max, source="verification")
    k3, k4 = fit.k3, fit.k4
    p_depth = XuModelParams(k_sigma=0.0, k_T=0.0)

    print("\n" + "=" * 74)
    print(f"[{label}]  exact rainflow sub-gradient: verification")
    print("=" * 74)
    print(f"  Source : {src}")
    print(f"  Window : [{soc_min:.2f}, {soc_max:.2f}] from the run summary CSV")
    print(f"  Phi    : k3={k3:.6e}  k4={k4:.6f}  R2={fit.r2:.4f}  "
          f"fit=[{fit.fit_lo:.2f},{fit.fit_hi:.2f}]")
    print(f"  eta    : {ETA_IN:.10f} (symmetric, RTE_ac = {RTE_AC:.10f})")

    if not guard(e, e_cap, soc_min, soc_max):
        raise SystemExit("guard failed; nothing below is trustworthy")

    cycles = rainflow_cycle_counting(e, e_cap)
    sg = compute_subgradient(
        e, cycles, dt_hours=DT_HOURS,
        battery_replacement_cost_per_MWh=B_REPL,
        eff_in=ETA_IN, eff_out=ETA_OUT, shi_fit=fit,
    )

    rising = np.diff(e, append=e[-1]) > 0
    idx = np.where(rising)[0]
    idx = idx[(idx > 10) & (idx < n - 10)]
    sample = idx[np.linspace(0, len(idx) - 1, min(N_SAMPLE, len(idx))).astype(int)]
    an = sg["dfdc"][sample]

    # ------------------------------- TEST A -------------------------------
    print(f"{'-'*74}\nTEST A  Exactness at a reference step of {H_REF:.0e} SoC\n{'-'*74}")
    f0_d = deg_cost(e, e_cap, k3, k4, p_depth)
    fwd = np.zeros(len(sample)); bwd = np.zeros(len(sample))
    for i, t in enumerate(sample):
        fwd[i], bwd[i] = one_sided(e, e_cap, int(t), k3, k4, p_depth, f0_d, H_REF)
    ctr = 0.5 * (fwd + bwd)

    scale    = robust_scale(an, ctr)
    kink     = np.abs(fwd - bwd) / scale > SMOOTH_TOL
    err_ctr  = np.abs(an - ctr) / scale
    err_side = np.minimum(np.abs(an - fwd), np.abs(an - bwd)) / scale

    # the previous normalisation, kept so the artefact can be reported and dismissed
    scale_v1    = np.maximum(np.abs(ctr), 1e-9)
    err_side_v1 = np.minimum(np.abs(an - fwd), np.abs(an - bwd)) / scale_v1

    n_smooth, n_kink = int(np.sum(~kink)), int(np.sum(kink))
    print(f"  smooth time steps         : {n_smooth:4d}/{len(sample)}")
    print(f"  rainflow topology changes : {n_kink:4d}/{len(sample)}   "
          f"({100*np.mean(kink):.0f} % of the sampled charging steps)")
    for name, mask, err in (("smooth, vs central   ", ~kink, err_ctr),
                            ("kink,   vs central   ", kink,  err_ctr),
                            ("kink,   vs one-sided ", kink,  err_side)):
        if not np.any(mask):
            continue
        x = err[mask]
        print(f"    {name}: p50 {pct(x,50):.2e}   p90 {pct(x,90):.2e}   "
              f"p99 {pct(x,99):.2e}   max {x.max():.2e}   "
              f"above 1% {int(np.sum(x > 1e-2)):d}/{x.size:d}")

    if np.any(kink):
        j = int(np.argmax(err_side_v1[kink]))
        w = np.where(kink)[0][j]
        print("\n  previous normalisation, divide by max(|central|, 1e-9):")
        print(f"    worst one-sided error {err_side_v1[kink].max():.2e} at step {sample[w]},")
        print(f"    where the central difference is {ctr[w]:.3e} EUR/MW against a median")
        print(f"    analytic magnitude of {np.median(np.abs(an)):.3e}. It is a division by a")
        print(f"    derivative passing through zero. The robust scale gives "
              f"{err_side[kink].max():.2e}.")

    # ------------------------------ TEST A2 -------------------------------
    print(f"\n{'-'*74}\nTEST A2  Sub-differential bracket at the topology changes\n{'-'*74}")
    lo_s = np.minimum(fwd, bwd)
    hi_s = np.maximum(fwd, bwd)
    tol  = SMOOTH_TOL * scale
    inside = (an >= lo_s - tol) & (an <= hi_s + tol)
    width  = hi_s - lo_s
    upos   = np.where(width > 0, (an - lo_s) / np.where(width > 0, width, 1.0), 0.0)
    at_end = np.minimum(upos, 1.0 - upos) < 1e-3
    print(f"  analytic inside [min, max] of the two one-sided derivatives : "
          f"{int(np.sum(inside[kink])):d}/{n_kink:d}")
    print(f"  analytic at an endpoint, within 1e-3 of the bracket width   : "
          f"{int(np.sum(at_end[kink])):d}/{n_kink:d}")
    print("  A central difference returns the midpoint and cannot match either end.")

    # ------------------------- TEST A3, step sweep ------------------------
    print(f"\n{'-'*74}\nTEST A3  Step-size sweep on the persistently smooth population\n{'-'*74}")
    # The population is classified at each step size rather than demanded to be
    # smooth at all of them. Whether a time step sits at a topology change is a
    # property of the step: the previous version required smoothness from 1e-3
    # down to 1e-11 and the intersection was empty, because at 1e-3 the
    # perturbation is 0.3 MWh on a 300 MWh pack and almost every step crosses a
    # change. The shrinking population is itself part of the result.
    cand = sample[~kink]
    if len(cand):
        cand = cand[np.linspace(0, len(cand) - 1, min(N_SWEEP, len(cand))).astype(int)]
    # Two floors close this window from opposite sides, and both are measured
    # rather than argued:
    #   asymmetry   median |fwd - bwd| / scale. Large at coarse steps because the
    #               perturbation crosses a rainflow topology change.
    #   round_floor eps * f0 / (h_MW * median|analytic|), the representation error
    #               of the cost divided by the step. Large at fine steps.
    # A step size is usable only where both sit below SMOOTH_TOL.
    a_h        = sg["dfdc"][cand]
    med_an     = float(np.median(np.abs(a_h))) if len(cand) else 1.0
    eps        = np.finfo(float).eps
    med_err    = np.full(len(H_LIST), np.nan)
    asym       = np.full(len(H_LIST), np.nan)
    round_fl   = np.full(len(H_LIST), np.nan)
    n_smooth_h = np.zeros(len(H_LIST), dtype=int)
    for j, h in enumerate(H_LIST):
        f_h = np.zeros(len(cand)); b_h = np.zeros(len(cand))
        for i, t in enumerate(cand):
            f_h[i], b_h[i] = one_sided(e, e_cap, int(t), k3, k4, p_depth, f0_d, h)
        c_h = 0.5 * (f_h + b_h)
        s_h = robust_scale(a_h, c_h)
        m = np.abs(f_h - b_h) / s_h <= SMOOTH_TOL
        n_smooth_h[j] = int(m.sum())
        asym[j] = float(np.median(np.abs(f_h - b_h) / s_h)) if len(cand) else np.nan
        h_mw = h * e_cap / (DT_HOURS * ETA_IN)
        round_fl[j] = eps * abs(f0_d) / (h_mw * max(med_an, 1e-30))
        if m.any():
            med_err[j] = float(np.median((np.abs(a_h - c_h) / s_h)[m]))

    print(f"  candidate time steps : {len(cand)}  "
          f"(smooth at the {H_REF:.0e} reference step)")
    print(f"  cost magnitude {abs(f0_d):.3e} EUR, median analytic {med_an:.3e} EUR/MW,")
    print(f"  smoothness tolerance {SMOOTH_TOL:.0e}")
    print("      h          smooth    asymmetry  rounding floor   median rel err")
    for h, ns, me, asy, rf in zip(H_LIST, n_smooth_h, med_err, asym, round_fl):
        me_s = "     n/a" if not np.isfinite(me) else f"{me:.2e}"
        print(f"      {h:.2e}  {ns:4d}/{len(cand):<4d}  {asy:.2e}     {rf:.2e}"
              f"       {me_s}")
    if np.isfinite(med_err).any():
        i_min = int(np.nanargmin(med_err))
        print(f"\n  best step {H_LIST[i_min]:.2e}, median relative error "
              f"{med_err[i_min]:.2e} on {n_smooth_h[i_min]} steps")
    print("  Reading: the asymmetry column is the topology floor and the rounding")
    print("  floor is the arithmetic one. A step size is usable only where both are")
    print("  below the tolerance. If that leaves a narrow band or none at all, a")
    print("  central difference cannot be made both differentiable and precise for")
    print("  this cost, which is the per-timestep form of the non-smoothness that")
    print("  stops a smooth NLP solver on the monolithic problem.")

    # ------------------------------- TEST B -------------------------------
    print(f"\n{'-'*74}\nTEST B  Efficiency identity, pointwise\n{'-'*74}")
    nz = np.abs(sg["dfdc"]) > 1e-12
    r = np.abs(sg["dfdd"][nz] / sg["dfdc"][nz])
    tgt = 1.0 / (ETA_IN * ETA_OUT)
    max_dev = float(np.max(np.abs(r - tgt)))
    print(f"  |df/dd| / |df/dc|  : min {r.min():.10f}   max {r.max():.10f}   "
          f"over {int(nz.sum())} time steps")
    print(f"  1/(eta_in*eta_out) : {tgt:.10f}")
    print(f"  max deviation      : {max_dev:.2e}")
    assert max_dev < 1e-9, "efficiency wiring is wrong"
    print("  PASS")

    ns = sg["n_straddled"]
    frac_one = float(np.mean(ns == 1))
    print(f"\n  cycles spanning each time step : min {ns.min()}  median "
          f"{int(np.median(ns))}  max {ns.max()}")
    print(f"  Shi Eqs. 17-18 assume exactly 1. That holds at {100*frac_one:.1f} % of")
    print("  time steps, which is why the spanning-set form replaces the partition.")

    # ------------------------------- TEST C -------------------------------
    print(f"\n{'-'*74}\nTEST C  Why S_sigma is excluded from the gradient path\n{'-'*74}")
    f0_f = deg_cost(e, e_cap, k3, k4, XU_LMO)
    fd_full = np.zeros(len(sample))
    for i, t in enumerate(sample):
        a_, b_ = one_sided(e, e_cap, int(t), k3, k4, XU_LMO, f0_f, H_REF)
        fd_full[i] = 0.5 * (a_ + b_)
    denom = robust_scale(an, fd_full)
    gap   = (fd_full - an) / denom
    ratio = np.abs(an) / np.maximum(np.abs(fd_full), 1e-30)
    sm = ~kink
    print("  depth-only analytic vs central difference of the S_sigma-weighted cost,")
    print("  on the smooth population only, because a central difference at a kink")
    print("  is not a derivative.")
    print(f"    smooth steps               : {int(sm.sum())}")
    print(f"    median relative gap        : {np.median(gap[sm])*100:+7.2f} %")
    med_ratio = float(np.median(ratio[sm])) if sm.any() else float("nan")
    print(f"    median local/weighted      : {med_ratio:.4f}   "
          f"(factor {1.0/max(med_ratio, 1e-30):.0f})")
    print("  Chapter 2 subsec:depth_only claims more than two orders of magnitude.")
    print("  Cycle depths are invariant to a uniform shift of the SoC trajectory.")
    print("  Cycle means are not, so the weighted derivative is not local.")

    # ------------------------------ figures -------------------------------
    # One panel. The step-size sweep is reported as a table in the text: it has
    # at most two usable points, so a plot of it would be misleading.
    fig, ax = plt.subplots(figsize=figsize(FIG_FRAC_SINGLE, aspect=0.62))
    bins = np.logspace(-16, 2, 40)
    if n_kink:
        ax.hist(np.clip(err_side[kink], 1e-16, None), bins=bins, color=pal["fill_a"],
                label="topology change, vs nearest one-sided")
        ax.hist(np.clip(err_ctr[kink], 1e-16, None), bins=bins, color=pal["secondary"],
                alpha=0.55, label="topology change, vs central")
    if n_smooth:
        ax.hist(np.clip(err_ctr[~kink], 1e-16, None), bins=bins, color=pal["primary"],
                alpha=0.75, label="smooth, vs central")
    ax.set_xscale("log")
    ax.set_xlabel("relative error  [-]")
    ax.set_ylabel("time steps  [-]")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)      # headroom so the legend clears the bars
    ax.legend(frameon=False, fontsize=FS_LEGEND, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(OUT_DIR / f"fig_subgrad_exactness_{label}.pdf")
    fig.savefig(OUT_DIR / f"fig_subgrad_exactness_{label}.png", dpi=DPI)
    plt.close(fig)

    # ---- normalised position within the one-sided bracket -------------------
    # Every value is mapped onto a 0 to 1 scale, where 0 is the lower one-sided
    # derivative and 1 is the upper one. This uses the whole non-differentiable
    # population instead of a sample, gives the horizontal axis a meaning, and
    # puts the central difference at exactly 0.5 at every step, because the
    # midpoint of an interval is its midpoint. That is the structural reason a
    # central difference cannot reach an endpoint however small the step.
    wdt = hi_s - lo_s
    okp = kink & (wdt > 0)
    if okp.any():
        u = (an[okp] - lo_s[okp]) / wdt[okp]
        fig, ax = plt.subplots(figsize=figsize(FIG_FRAC_SINGLE, aspect=0.58))
        ax.hist(u, bins=np.linspace(-0.025, 1.025, 22), color=pal["fill_a"],
                label=f"analytic sub-gradient, {int(okp.sum())} time steps")
        ax.axvline(0.5, color=pal["secondary"], lw=1.2, ls="--")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.30)
        ax.text(0.52, ax.get_ylim()[1] * 0.97, "central difference",
                fontsize=FS_ANNOT, color=pal["secondary"], ha="left", va="top")
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("position within the bracket of the two one-sided derivatives  [-]")
        ax.set_ylabel("time steps  [-]")
        ax.legend(frameon=False, fontsize=FS_LEGEND, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(OUT_DIR / f"fig_subgrad_position_{label}.pdf")
        fig.savefig(OUT_DIR / f"fig_subgrad_position_{label}.png", dpi=DPI)
        plt.close(fig)
        print(f"\n  bracket position: inside [0,1] "
              f"{int(np.sum((u >= -1e-9) & (u <= 1 + 1e-9)))}/{int(okp.sum())}, "
              f"at an endpoint {int(np.sum(np.minimum(u, 1 - u) < 1e-3))}, "
              f"at the lower end {int(np.sum(u < 1e-3))}, "
              f"at the upper end {int(np.sum(u > 1 - 1e-3))}")

    # ---- concrete bracket example, kept as an alternative figure ------------
    fig, ax = plt.subplots(figsize=figsize(FIG_FRAC_WIDE, aspect=0.40))
    kidx = np.where(kink)[0]
    if kidx.size:
        # Order by position within the bracket, not by the analytic value. The
        # horizontal axis then carries the argument: the lower-endpoint family,
        # then the interior cases, then the upper-endpoint family, with the
        # central difference mid-bar throughout. With N_BRACKET at or above the
        # population size there is no selection at all.
        wk = np.maximum(hi_s[kidx] - lo_s[kidx], 1e-30)
        uk = (an[kidx] - lo_s[kidx]) / wk
        kidx = kidx[np.argsort(uk)]
        pick = kidx[np.linspace(0, kidx.size - 1, min(N_BRACKET, kidx.size)).astype(int)]
        dense = len(pick) > 50
        ms_pt = 2.0 if dense else 3.2
        lw_pt = 1.0 if dense else 1.6
        x = np.arange(len(pick))
        ax.vlines(x, lo_s[pick], hi_s[pick], color=pal["grid"], lw=lw_pt,
                  label="one-sided derivatives")
        ax.plot(x, an[pick], "o", ms=ms_pt, color=pal["primary"],
                label="analytic sub-gradient")
        ax.plot(x, 0.5 * (fwd[pick] + bwd[pick]), "^", ms=ms_pt, color=pal["secondary"],
                label="central difference")
    ax.set_xticks([])
    ax.set_xlabel("time steps at a topology change, ordered by position within the bracket")
    ax.set_ylabel(r"$\partial f / \partial p_t^{\mathrm{ch}}$  [EUR/MW]")
    ax.legend(frameon=False, fontsize=FS_LEGEND, ncol=3,
              bbox_to_anchor=(0.0, 1.02, 1.0, 0.12), loc="lower left", mode="expand",
              borderaxespad=0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(OUT_DIR / f"fig_subgrad_bracket_{label}.pdf")
    fig.savefig(OUT_DIR / f"fig_subgrad_bracket_{label}.png", dpi=DPI)
    plt.close(fig)

    # -------------------------------- CSV ---------------------------------
    row = {
        "case": label,
        "source": src, "soc_min": soc_min, "soc_max": soc_max,
        "e_cap_MWh": e_cap, "n_steps": n, "n_cycles": len(cycles),
        "k3": k3, "k4": k4, "r2": fit.r2,
        "rte_ac": RTE_AC, "eta_one_way": ETA_IN,
        "h_ref": H_REF, "n_sample": len(sample),
        "n_smooth": n_smooth, "n_kink": n_kink, "kink_fraction": float(np.mean(kink)),
        "n_above_1pct": int(np.sum(err_ctr[kink] > 1e-2)) if n_kink else 0,
        "smooth_vs_central_p50": pct(err_ctr[~kink], 50) if n_smooth else np.nan,
        "smooth_vs_central_max": float(err_ctr[~kink].max()) if n_smooth else np.nan,
        "kink_vs_central_p50": pct(err_ctr[kink], 50) if n_kink else np.nan,
        "kink_vs_onesided_p50": pct(err_side[kink], 50) if n_kink else np.nan,
        "kink_vs_onesided_p99": pct(err_side[kink], 99) if n_kink else np.nan,
        "kink_vs_onesided_max": float(err_side[kink].max()) if n_kink else np.nan,
        "kink_vs_onesided_max_old_norm": float(err_side_v1[kink].max()) if n_kink else np.nan,
        "bracket_inside": int(np.sum(inside[kink])) if n_kink else 0,
        "bracket_at_endpoint": int(np.sum(at_end[kink])) if n_kink else 0,
        "sweep_n_candidates": len(cand),
        "sweep_min_median_relerr": (float(np.nanmin(med_err))
                                    if np.isfinite(med_err).any() else np.nan),
        "sweep_h_at_min": (float(H_LIST[int(np.nanargmin(med_err))])
                           if np.isfinite(med_err).any() else np.nan),
        "sweep_n_smooth_at_min": (int(n_smooth_h[int(np.nanargmin(med_err))])
                                  if np.isfinite(med_err).any() else 0),
        "sweep_h_list": " ".join(f"{h:.3e}" for h in H_LIST),
        "sweep_n_smooth_list": " ".join(str(int(v)) for v in n_smooth_h),
        "sweep_asymmetry_list": " ".join(f"{v:.3e}" for v in asym),
        "sweep_rounding_floor_list": " ".join(f"{v:.3e}" for v in round_fl),
        "cost_magnitude_EUR": abs(f0_d),
        "median_analytic_EUR_per_MW": med_an,
        "eff_ratio_min": float(r.min()), "eff_ratio_max": float(r.max()),
        "eff_ratio_target": tgt, "eff_ratio_max_dev": max_dev,
        "span_min": int(ns.min()), "span_median": int(np.median(ns)),
        "span_max": int(ns.max()), "span_equals_one_fraction": frac_one,
        "testC_median_gap_pct": float(np.median(gap[sm]) * 100) if sm.any() else np.nan,
        "testC_median_ratio": med_ratio,
    }
    print(f"\n  Saved: {OUT_DIR / f'fig_subgrad_bracket_{label}.pdf'}")
    return row


def main() -> None:
    pal = apply_thesis_style(palette="brand", usetex=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    soc_min_y, soc_max_y, win_src = load_window()
    check_module_families(soc_min_y, soc_max_y)
    print(f"  SoC window for the polynomial fit: "
          f"[{soc_min_y:.2f}, {soc_max_y:.2f}] from {win_src}\n")

    cases = discover_cases(RESULTS_DIR)
    if not cases:
        raise SystemExit(f"no rte910 cases found in {RESULTS_DIR}")

    rows = []
    for label, npy_path, csv_path in cases:
        e, e_cap, soc_min, soc_max, src = load_case(npy_path, csv_path)
        rows.append(run_case(label, e, e_cap, soc_min, soc_max, src, pal))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "subgradient_exact_summary.csv", index=False)

    print(f"\n{'='*74}\nacross price years\n{'='*74}")
    print("  case      corners  above 1%%  bracket  span==1   factor")
    for _, r in df.iterrows():
        print(f"  {r['case']:9s} {100*r['kink_fraction']:6.1f}%%  "
              f"{r['n_above_1pct']:7d}  {r['bracket_inside']:3d}/{r['n_kink']:<3d}  "
              f"{100*r['span_equals_one_fraction']:6.1f}%%  "
              f"{1.0/max(r['testC_median_ratio'], 1e-30):7.0f}")
    frac = df["kink_fraction"]
    print(f"\n  corner fraction spans {100*frac.min():.1f} to {100*frac.max():.1f} %.")
    print("  Compare the bracket and span columns, which should be stable across")
    print("  years, against the corner fraction, which need not be.")
    print(f"\n  Saved: {OUT_DIR / 'subgradient_exact_summary.csv'}")


if __name__ == "__main__":
    main()