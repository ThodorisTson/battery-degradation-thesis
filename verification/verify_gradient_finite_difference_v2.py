"""
verify_gradient_finite_difference.py   (revision of 29 July 2026)

Finite-difference verification of the degradation-cost design gradient, in the style of Fig. 6.7 of Martins & Ning (the classic step-size "checkmark").

What it checks
--------------
Raising E_cap with the dispatch held fixed divides the entire normalised SoC trajectory by the scale factor, so every rainflow cycle's depth (delta) and mean
(sigma) scale together. This is the uniform-scale perturbation swept below. Its analytic derivative is

    d f_cyc / ds = sum_i count_i * Phi_i * (k4 + k_sigma * sigma_i)          (complete)

where Phi_i = phi_shi_cycle(delta_i, sigma_i) already carries S_sigma and S_T.
The finite difference converges to this complete derivative, which is the design gradient the outer loop now uses (with the mean-SoC coupling term included).

The scaling direction is not a convenience. The derivative of degradation along that direction is dDegCost/dE_cap by definition, so any other perturbation would
validate a different derivative.

An optional reference line (SHOW_REFERENCE_LINE) marks a gradient that omits the mean-SoC coupling, sum_i count_i * Phi'(delta_i) * S_sigma * delta_i = k4 * f_cyc,
which for this site is ~31% below the complete derivative. It is off by default; turn it on only to illustrate why the coupling term is needed.

Reproducibility
---------------
Reads the v5.6 RTE-test outputs only (no run-script changes):
  - multiyear_*rte910*.npy  -> annual_soc[0], e_cap_nominal
  - npv_summary_*.csv        -> soc_min, soc_max  (the SoC window; single drift source removed)
Re-fits the Shi polynomial deterministically from the window, so k3/k4 are recomputed rather than trusted. Runs on Windows/VS Code; uses the real `rainflow`
package (pulled in transitively by degradation.shi).

Reads results/baseline/xu/ through degradation.paths. Figures are written beside
this script. Produces Figure C.1, fd_verification_dk2019 and fd_verification_dk2022.

Changes in this revision
------------------------
The sweep, the analytic reference, the figure and the two difference forms are unchanged. Everything below is additive.

  1. GUARD on the trace and the capacity, matching verify_subgradient_exact.py.
     It catches a trace and a capacity written by different runs, and doubles as
     a units check, since a trace already in normalised SoC would put the
     implied window three orders of magnitude too low.
  2. k_sigma is read from the model parameters instead of a module constant.
     ShiModelParams carries 1.04, so no number changes; the constant is now a
     fallback that prints a warning if it is ever used.
  3. sigma_Phi, the Phi-weighted mean SoC, and the closed form of the coupling
     share, k_sigma sigma_Phi / (k4 + k_sigma sigma_Phi). This makes the ~31%
     a property of the window centre and the fitted exponent rather than a
     measurement of one dispatch.
  4. REPARTITION_CHECK: the rainflow cycle list is recounted at four points of
     the sweep and compared to the unscaled list with the scale divided out.
     This measures the tie-creation effect identified in the July analysis
     rather than assuming the counting is invariant.
  5. CALENDAR_DIAGNOSTIC: a second sweep on f_cyc + f_cal against
     complete + k_sigma sigma_bar f_cal. Numbers only, no figure. In April the
     calendar term inside the sweep produced an irreducible relative floor of
     about 1.2%, because the slope in use then was the first-order form
     df_cal/ds ~ f_cal. The slope has since been replaced by k_sigma sigma_bar
     f_cal, which is the exact derivative of the Xu calendar expression under
     this scaling. This diagnostic settles whether the floor survived. It does
     not touch the figure either way.
  6. Figure drawn at the width it is included at (0.49 textwidth, inside the
     subfigure pair in Appendix C), per the rule in degradation/style.py. The
     previous 0.74 meant every font in that figure rendered at about two thirds
     of its intended on-page size.
  7. Summary CSV extended.
  8. PRODUCTION_GRADIENT hook, off by default. See step 2 in the handover.
"""

from __future__ import annotations
import importlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from degradation.paths import RESULTS_DIR as REPO_RESULTS_DIR
from degradation.style import apply_thesis_style, figsize, FS_ANNOT, FS_LEGEND

from degradation.shi import (
    fit_shi_polynomial,
    rainflow_cycle_counting,
    compute_fd_shi,
    phi_shi_cycle,
    ShiModelParams,
)
# calendar term for the diagnostic only; the figure stays on f_cyc
from degradation.xu import ft_calendar, XU_LMO

# ------------------------------- configuration --------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
# The baseline run writes multiyear_*_baseline_rte910.npy and its paired
# npv_summary CSV here, one of each per price year.
RESULTS_DIR = REPO_RESULTS_DIR / "baseline" / "xu"
OUT_DIR     = SCRIPT_DIR

DIFF_MODES          = ("central", "forward")  # Jenna has seen both and kept both
H_SWEEP             = np.logspace(-1, -13, 40)  # step sizes, large -> tiny
SHOW_REFERENCE_LINE = False                     # draw the no-coupling reference line (teaching aid only)
SHOW_SLOPE_GUIDES   = True                      # thin O(h) / O(h^2) guides on the truncation branch
REPARTITION_CHECK   = True                      # measure the rainflow tie-creation effect
ZERO_DEPTH_TOL      = 1e-12                     # a record below this depth carries no wear
CALENDAR_DIAGNOSTIC = True                      # second sweep including f_cal, numbers only
T_CELL_C            = 25.0
DT_HOURS            = 1.0
K_SIGMA_FALLBACK    = 1.04                      # only used if the params carry no k_sigma
DPI                 = 300

# Width the figure is included at in Appendix C: two subfigures at 0.49\textwidth.
# Drawing at this width keeps matplotlib points equal to on-page points.
FIG_WIDTH_FRAC      = 0.49
FIG_ASPECT          = 0.88

# Step 2 only. Set to ("degradation.subgradient", "dDegCost_dEcap_terms") after
# extracting the inline block from the run script. Leave as None for step 1.
PRODUCTION_GRADIENT = ("degradation.subgradient", "dDegCost_dEcap_terms")

# --------------------------------- helpers ------------------------------------
def discover_cases(results_dir: Path):
    """Find the newest multiyear rte910 .npy per priceset, paired with its npv_summary CSV.

    Prints the resolved folder and the exact files chosen, so you can confirm the
    right run is being read before the sweep begins. Only names that both start
    with 'multiyear_' and contain 'rte910' are considered, so e_cap_fixed.npy and
    storage_e_fixed.npy are ignored.
    """
    print(f"[discover] results folder : {results_dir}")
    if not results_dir.is_dir():
        raise SystemExit(f"[discover] folder does not exist: {results_dir}")

    all_npy = sorted(results_dir.glob("multiyear_*rte910*.npy"))
    if not all_npy:
        raise SystemExit(f"[discover] no 'multiyear_*rte910*.npy' files in {results_dir}")

    # group by priceset; the filename embeds YYYYMMDD_HHMMSS after 'multiyear_',
    # so the alphabetically largest name is the most recent run
    by_label = {}
    for npy in all_npy:
        stem  = npy.name[len("multiyear_"):-len(".npy")]
        label = "dk2019" if "dk2019" in stem else "dk2022" if "dk2022" in stem else stem
        by_label.setdefault(label, []).append(npy)

    cases = []
    for label in sorted(by_label):
        group  = by_label[label]
        chosen = max(group, key=lambda p: p.name)          # newest run for this priceset
        stem   = chosen.name[len("multiyear_"):-len(".npy")]
        csv_hits = list(results_dir.glob(f"npv_summary_{stem}.csv"))
        if not csv_hits:
            print(f"[discover] {label}: SKIP (no npv_summary matching {chosen.name})")
            continue
        print(f"[discover] {label}: {chosen.name}")
        print(f"[discover]         + {csv_hits[0].name}")
        for other in group:
            if other is not chosen:
                print(f"[discover]         (ignoring older {other.name})")
        cases.append((label, chosen, csv_hits[0]))
    return cases


def load_case(npy_path: Path, csv_path: Path):
    d = np.load(npy_path, allow_pickle=True).item()
    soc   = np.asarray(d["annual_soc"][0], dtype=float)
    e_cap = float(d["e_cap_nominal"])
    par   = pd.read_csv(csv_path)
    soc_min = float(par["soc_min"].iloc[0])
    soc_max = float(par["soc_max"].iloc[0])
    fit = fit_shi_polynomial(soc_min=soc_min, soc_max=soc_max, verbose=False)
    p   = ShiModelParams.from_fit(fit)
    return soc, e_cap, p, fit, soc_min, soc_max


def guard(label, soc, e_cap, soc_min, soc_max):
    """Refuse to trust the run if the trace and the capacity disagree.

    Same failure mode verify_subgradient_exact.py guards against: a trace and a
    capacity written by different runs silently rescale every normalised depth.
    """
    lo, hi = soc.min() / e_cap, soc.max() / e_cap
    print(f"\n{'-'*74}\nGUARD  [{label}]  trace and capacity from the same solve?\n{'-'*74}")
    print(f"  e_cap             : {e_cap:.1f} MWh")
    print(f"  SoC range implied : {lo:.4f} to {hi:.4f}")
    print(f"  SoC window (CSV)  : {soc_min:.2f} to {soc_max:.2f}")
    ok = (hi > soc_max - 0.02) and (lo < soc_min + 0.02)
    if not ok:
        implied = soc.max() / soc_max if soc_max > 0 else float("nan")
        print("\n  *** INCONSISTENT ***")
        print(f"  The trace does not fill the window. A capacity of {implied:.1f} MWh")
        print(f"  would put the range at [{soc.min()/implied:.3f}, {soc.max()/implied:.3f}].")
        print(f"  Every normalised depth below is scaled by {implied/e_cap:.3f}.")
        print("  Either the files are from different runs, or the trace is already")
        print("  normalised and must not be divided by the capacity again.\n")
    else:
        print("  consistent\n")
    return ok


def k_sigma_of(p):
    v = getattr(p, "k_sigma", None)
    if v is None:
        print(f"  WARNING k_sigma not found on the model parameters, "
              f"using the fallback {K_SIGMA_FALLBACK}")
        return float(K_SIGMA_FALLBACK), True
    return float(v), False


def f_cyc_of_scale(soc, e_cap, p, s):
    """Full pipeline: re-run rainflow on the scaled trajectory, return f_cyc."""
    return compute_fd_shi(rainflow_cycle_counting(soc * s, e_cap), p, T_C=T_CELL_C)[0]


def f_cal_of_scale(soc, e_cap, s, t_seconds):
    """Production Xu calendar term at the scaled trajectory.

    Only sigma_bar depends on the scale, and it scales with it, exactly as in the
    run script where sigma_bar = mean(storage_e) / e_cap.
    """
    sigma_bar = float(np.mean(soc * s)) / e_cap
    return float(ft_calendar(t_seconds, sigma_bar, T_CELL_C, XU_LMO))


def analytic_references(soc, e_cap, p, k_sigma, t_seconds):
    """Complete cycle derivative, no-coupling reference, calendar slope, diagnostics."""
    cyc = rainflow_cycle_counting(soc, e_cap)
    cnt = np.array([c["count"]    for c in cyc])
    sig = np.array([c["soc_mean"] for c in cyc])
    phi = np.array([float(phi_shi_cycle(c["dod"], c["soc_mean"], T_CELL_C, p)) for c in cyc])

    w = cnt * phi
    f_cyc       = float(w.sum())
    complete    = float(np.sum(w * (p.k4 + k_sigma * sig)))
    no_coupling = float(p.k4 * f_cyc)
    sigma_phi   = float(np.sum(w * sig) / w.sum()) if w.sum() > 0 else float("nan")
    closed      = k_sigma * sigma_phi / (p.k4 + k_sigma * sigma_phi)

    sigma_bar = float(np.mean(soc)) / e_cap
    f_cal     = float(ft_calendar(t_seconds, sigma_bar, T_CELL_C, XU_LMO))
    cal_slope = k_sigma * sigma_bar * f_cal

    n_full  = int(np.sum(cnt == 1.0))
    n_half  = int(np.sum(cnt == 0.5))

    return {
        "n_cycles": len(cyc), "n_full": n_full, "n_half": n_half,
        "n_other": len(cyc) - n_full - n_half, "sum_count": float(cnt.sum()),
        "f_cyc": f_cyc, "f_cal": f_cal,
        "sigma_bar": sigma_bar, "sigma_phi": sigma_phi,
        "complete": complete, "no_coupling": no_coupling,
        "cal_slope": cal_slope, "total": complete + cal_slope,
        "coupling_measured_pct": 100.0 * abs(complete - no_coupling) / abs(complete),
        "coupling_closed_pct": 100.0 * closed,
    }


def sweep(soc, e_cap, p, reference, t_seconds=None, include_calendar=False):
    """Relative FD error against `reference`, over H_SWEEP.

    include_calendar=False reproduces the established sweep exactly.
    """
    def f_of(s):
        v = f_cyc_of_scale(soc, e_cap, p, s)
        if include_calendar:
            v += f_cal_of_scale(soc, e_cap, s, t_seconds)
        return v

    f0 = f_of(1.0)
    err = {m: [] for m in DIFF_MODES}
    for h in H_SWEEP:
        fp = f_of(1.0 + h)
        if "central" in DIFF_MODES:
            fm = f_of(1.0 - h)
            err["central"].append(abs((fp - fm) / (2 * h) - reference) / abs(reference))
        if "forward" in DIFF_MODES:
            err["forward"].append(abs((fp - f0) / h - reference) / abs(reference))
    return {m: np.array(v) for m, v in err.items()}


# ------------------------- rainflow repartition check -------------------------
def cycle_summary(soc, e_cap, p, s):
    """Scale-invariant summaries of the rainflow record set at scale s.

    Under an exact uniform scaling every cycle maps to itself with delta -> s
    delta, so dividing the scale back out makes each summary below invariant. In
    floating point the mapping is not exact: near-equal reversals can become
    exactly equal at some scales and the counter then emits a different number of
    records. The July analysis measured the resulting movement in wear at about
    5.6e-14.

    The quantities are chosen so that a record carrying no wear cannot move them:

        n_records    number of rainflow records, expected to move
        n_zero       records with depth below ZERO_DEPTH_TOL
        sum_count    total half and full cycle count
        throughput   sum(count * delta) / s           , linear in depth
        wear_proxy   sum(count * delta**k4) / s**k4   , weighted as Phi weights it

    If the extra records are zero-depth, n_records moves while throughput and
    wear_proxy hold to machine precision. That is the statement the appendix
    needs, and it is what an element-wise array comparison cannot express,
    because two record lists of different length do not align.
    """
    cyc = rainflow_cycle_counting(soc * s, e_cap)
    if not cyc:
        return {"n_records": 0, "n_zero": 0, "sum_count": 0.0,
                "throughput": 0.0, "wear_proxy": 0.0}
    d = np.array([c["dod"]   for c in cyc], dtype=float)
    q = np.array([c["count"] for c in cyc], dtype=float)
    return {
        "n_records":  len(cyc),
        "n_zero":     int(np.sum(d < ZERO_DEPTH_TOL)),
        "sum_count":  float(q.sum()),
        "throughput": float(np.sum(q * d)) / s,
        "wear_proxy": float(np.sum(q * d ** p.k4)) / (s ** p.k4),
    }


def repartition_check(soc, e_cap, p, probes):
    """Compare the record set at each probe against the unperturbed one."""
    base = cycle_summary(soc, e_cap, p, 1.0)
    rows = []
    for h in probes:
        for sgn in (+1.0, -1.0):
            cur = cycle_summary(soc, e_cap, p, 1.0 + sgn * h)
            rows.append({
                "h": sgn * h,
                "n_records": cur["n_records"],
                "d_records": cur["n_records"] - base["n_records"],
                "n_zero": cur["n_zero"],
                "sum_count": cur["sum_count"],
                "d_count": cur["sum_count"] - base["sum_count"],
                "throughput_rel": abs(cur["throughput"] - base["throughput"])
                                  / max(abs(base["throughput"]), 1e-30),
                "wear_rel": abs(cur["wear_proxy"] - base["wear_proxy"])
                            / max(abs(base["wear_proxy"]), 1e-30),
            })
    return base, rows

# ------------------------- step 2: the shipped gradient -----------------------
def probe_production_gradient(payload):
    """Compare the shipped capacity gradient against the analytic reference.

    Off unless PRODUCTION_GRADIENT is set. The target is the extraction of the
    inline block at lines 721-750 of the v5.6 run script. It returns the bracket
    per MWh of capacity, before the replacement-cost, annuity and annualisation
    prefactors, so the comparison carries no prefactor.
    """
    if PRODUCTION_GRADIENT is None:
        return None
    mod_name, fn_name = PRODUCTION_GRADIENT
    try:
        fn = getattr(importlib.import_module(mod_name), fn_name)
    except Exception as exc:
        print(f"\n[production gradient] import of {mod_name}.{fn_name} failed: {exc}")
        return None
    sig = inspect.signature(fn)
    kwargs = {k: v for k, v in payload.items() if k in sig.parameters}
    missing = [n for n, prm in sig.parameters.items()
               if n not in kwargs and prm.default is inspect.Parameter.empty]
    if missing:
        print(f"\n[production gradient] {mod_name}.{fn_name}{sig}")
        print(f"[production gradient] cannot call: missing {missing}")
        return None
    try:
        out = fn(**kwargs)
    except Exception as exc:
        print(f"\n[production gradient] {mod_name}.{fn_name}{sig}")
        print(f"[production gradient] call failed: {exc}")
        return None
    return {k: float(v) for k, v in out.items()} if isinstance(out, dict) \
        else {"total": float(out)}


# ---------------------------------- plotting ----------------------------------
def make_figure(label, err, complete, no_coupling, pal, out_dir: Path):
    style = {
        "central": dict(color=pal["primary"],   marker="o", ms=3.2, name="central difference"),
        "forward": dict(color=pal["secondary"], marker="s", ms=3.0, name="forward difference"),
    }
    fig, ax = plt.subplots(figsize=figsize(FIG_WIDTH_FRAC, aspect=FIG_ASPECT))

    for m in DIFF_MODES:
        s = style[m]
        ax.loglog(H_SWEEP, err[m], s["marker"] + "-", color=s["color"],
                  ms=s["ms"], lw=1.2, label=s["name"])
        i = int(np.argmin(err[m]))
        ax.plot(H_SWEEP[i], err[m][i], s["marker"], mfc="none",
                mec=s["color"], ms=9, mew=1.2)  # ring the minimum; values go in the caption

    if SHOW_SLOPE_GUIDES:
        ia    = 2                       # anchor in the truncation regime (h ~ 2e-2)
        hg    = np.array([H_SWEEP[0], H_SWEEP[9]])
        h_lab = H_SWEEP[0] * 0.80       # left-hand anchor (large h) for the labels
        if "central" in DIFF_MODES:
            g2 = err["central"][ia] * 0.5 * (hg / H_SWEEP[ia]) ** 2
            ax.loglog(hg, g2, ":", color=pal["primary"], lw=0.9)
            y2 = err["central"][ia] * 0.5 * (h_lab / H_SWEEP[ia]) ** 2
            ax.text(h_lab, y2 * 0.05, r"$\propto h^{2}$", fontsize=FS_ANNOT,
                    color=pal["primary"], ha="center", va="top")
        if "forward" in DIFF_MODES:
            g1 = err["forward"][ia] * 0.5 * (hg / H_SWEEP[ia]) ** 1
            ax.loglog(hg, g1, ":", color=pal["secondary"], lw=0.9)
            y1 = err["forward"][ia] * 0.5 * (h_lab / H_SWEEP[ia]) ** 1
            ax.text(h_lab, y1 * 0.35, r"$\propto h$", fontsize=FS_ANNOT,
                    color=pal["secondary"], ha="center", va="top")

    if SHOW_REFERENCE_LINE:
        gap = abs(complete - no_coupling) / abs(complete)
        ax.axhline(gap, color=pal["neutral"], lw=1.0, ls="--")
        ax.text(H_SWEEP[1], gap * 1.25,
                f"gradient without mean-SoC coupling ({gap*100:.0f}%)",
                fontsize=FS_ANNOT, color=pal["neutral"], va="bottom", ha="left")

    ax.invert_xaxis()  # Fig. 6.7 orientation: h decreases to the right
    ax.grid(True, which="both", color=pal["grid"], lw=0.5, alpha=0.6)
    ax.set_xlabel(r"finite-difference step size $h$ (fractional SoC scale)")
    ax.set_ylabel(r"relative error $|\,D_{h}-\mathrm{d}f_{\mathrm{cyc}}/\mathrm{d}s\,|\,/\,"
                  r"|\mathrm{d}f_{\mathrm{cyc}}/\mathrm{d}s|$")
    ax.legend(loc="upper right", fontsize=FS_LEGEND, frameon=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"fd_verification_{label}.pdf")
    fig.savefig(out_dir / f"fd_verification_{label}.png", dpi=DPI)
    plt.close(fig)


# ------------------------------------ main ------------------------------------
def main():
    pal = apply_thesis_style(palette="brand", usetex=False)
    cases = discover_cases(RESULTS_DIR)
    if not cases:
        raise SystemExit(f"No rte910 cases found in {RESULTS_DIR}")

    eps = np.finfo(float).eps
    rows = []
    for label, npy_path, csv_path in cases:
        soc, e_cap, p, fit, soc_min, soc_max = load_case(npy_path, csv_path)

        if not guard(label, soc, e_cap, soc_min, soc_max):
            print(f"[{label}] guard failed, case skipped")
            continue

        k_sigma, used_fallback = k_sigma_of(p)
        t_seconds = len(soc) * DT_HOURS * 3600.0

        ref = analytic_references(soc, e_cap, p, k_sigma, t_seconds)
        err = sweep(soc, e_cap, p, reference=ref["complete"])
        make_figure(label, err, ref["complete"], ref["no_coupling"], pal, OUT_DIR)

        line = {
            "case": label, "e_cap_MWh": e_cap, "soc_min": soc_min, "soc_max": soc_max,
            "k3": fit.k3, "k4": fit.k4, "k_sigma": k_sigma,
            "k_sigma_from_fallback": used_fallback,
            "n_cycles": ref["n_cycles"], "n_full": ref["n_full"],
            "n_half": ref["n_half"], "n_other": ref["n_other"],
            "sum_count": ref["sum_count"],
            "f_cyc": ref["f_cyc"], "f_cal": ref["f_cal"],
            "sigma_bar": ref["sigma_bar"], "sigma_phi": ref["sigma_phi"],
            "complete_deriv": ref["complete"], "no_coupling_deriv": ref["no_coupling"],
            "no_coupling_error_pct": ref["coupling_measured_pct"],
            "no_coupling_closed_form_pct": ref["coupling_closed_pct"],
            "calendar_slope": ref["cal_slope"], "total_slope": ref["total"],
        }
        for m in DIFF_MODES:
            i = int(np.argmin(err[m]))
            line[f"{m}_min_relerr"] = err[m][i]
            line[f"{m}_h_at_min"]   = H_SWEEP[i]

        print(f"\n{'='*74}\n[{label}]  cycle-term sweep (the appendix figure)\n{'='*74}")
        print(f"  k3 {fit.k3:.6e}   k4 {fit.k4:.6f}   k_sigma {k_sigma}")
        R, S = ref["n_cycles"], ref["sum_count"]
        print(f"  records {R} = {ref['n_full']} full + {ref['n_half']} half"
              f"   ({ref['n_other']} other, must be 0)")
        print(f"  summed multiplicity {S:.1f}   "
              f"identities 2S-R = {2*S - R:.0f}, 2(R-S) = {2*(R - S):.0f}")
        print(f"  f_cyc {ref['f_cyc']:.6f}   "
              f"f_cal {ref['f_cal']:.6f}   sigma_bar {ref['sigma_bar']:.4f}")
        print(f"  complete {ref['complete']:.6f}   no-coupling {ref['no_coupling']:.6f}")
        print(f"  coupling share: measured {ref['coupling_measured_pct']:.2f} %   "
              f"closed form {ref['coupling_closed_pct']:.2f} %   "
              f"(sigma_Phi = {ref['sigma_phi']:.4f})")
        print("  " + "  ".join(
            f"{m}: {err[m].min():.3e} at h = {H_SWEEP[int(np.argmin(err[m]))]:.3e}"
            for m in DIFF_MODES))
        print(f"  reference steps: sqrt(eps) = {np.sqrt(eps):.3e}   "
              f"eps^(1/3) = {eps ** (1/3):.3e}")

        # ---------------------- repartition diagnostic ----------------------
        if REPARTITION_CHECK:
            base_rec, probe_rows = repartition_check(
                soc, e_cap, p, [H_SWEEP[0], H_SWEEP[19], H_SWEEP[29], H_SWEEP[39]])
            worst_tp = max(r["throughput_rel"] for r in probe_rows)
            worst_we = max(r["wear_rel"] for r in probe_rows)
            n_moved = sum(1 for r in probe_rows if r["d_records"] != 0)
            print("\n  rainflow repartition across the sweep")
            print(f"    records at s = 1 : {base_rec['n_records']}  "
                  f"({base_rec['n_zero']} of them zero-depth)   "
                  f"sum of counts {base_rec['sum_count']:.1f}")
            print(f"    probes where the record count moved : {n_moved}/{len(probe_rows)}")
            print(f"    worst relative change in throughput : {worst_tp:.2e}")
            print(f"    worst relative change in wear proxy : {worst_we:.2e}")
            worst_dc = max(abs(r["d_count"]) for r in probe_rows)
            print(f"    worst change in the total cycle count : {worst_dc:.3f}")
            print("      h             records  d_rec  zero   sum_count  d_count"
                  "   throughput    wear")
            for r in probe_rows:
                print(f"      {r['h']:+.2e}  {r['n_records']:8d}  {r['d_records']:+5d}  "
                      f"{r['n_zero']:4d}  {r['sum_count']:9.1f}  {r['d_count']:+7.3f}"
                      f"   {r['throughput_rel']:.2e}   {r['wear_rel']:.2e}")
            print("    Reading: d_count at zero with d_rec non-zero means the counter")
            print("    reported the same cycles under a different number of records, that")
            print("    is, full cycles split into half-cycle pairs. Throughput and wear are")
            print("    then invariant by construction, which is what the two columns show.")
            line["repartition_n_records"] = base_rec["n_records"]
            line["repartition_n_zero"] = base_rec["n_zero"]
            line["repartition_sum_count"] = base_rec["sum_count"]
            line["repartition_probes_moved"] = n_moved
            line["repartition_worst_d_count"] = worst_dc
            line["repartition_worst_throughput_rel"] = worst_tp
            line["repartition_worst_wear_rel"] = worst_we

        # ----------------------- calendar diagnostic ------------------------
        if CALENDAR_DIAGNOSTIC:
            err_tot = sweep(soc, e_cap, p, reference=ref["total"],
                            t_seconds=t_seconds, include_calendar=True)
            print(f"\n  calendar diagnostic: sweep on f_cyc + f_cal against "
                  f"complete + k_sigma sigma_bar f_cal")
            print(f"    analytic calendar slope {ref['cal_slope']:.6f}   "
                  f"total {ref['total']:.6f}")
            print("    " + "  ".join(
                f"{m}: {err_tot[m].min():.3e} at h = {H_SWEEP[int(np.argmin(err_tot[m]))]:.3e}"
                for m in DIFF_MODES))
            print(f"    cycle-only minimum for comparison: "
                  f"{err['central'].min():.3e}")
            print("    Reaching the same order as the cycle-only sweep means the")
            print("    calendar slope is exact under this perturbation. A floor near")
            print("    1e-2 relative would mean the April approximation error survived.")
            for m in DIFF_MODES:
                i = int(np.argmin(err_tot[m]))
                line[f"withcal_{m}_min_relerr"] = err_tot[m][i]
                line[f"withcal_{m}_h_at_min"]   = H_SWEEP[i]

        # --------------------- step 2: shipped gradient ---------------------
        cyc_list = rainflow_cycle_counting(soc, e_cap)
        prod = probe_production_gradient({
            "dods":        np.array([c["dod"]      for c in cyc_list]),
            "counts":      np.array([c["count"]    for c in cyc_list]),
            "soc_means":   np.array([c["soc_mean"] for c in cyc_list]),
            "storage_e":   soc,
            "fd_calendar": ref["f_cal"],
            "e_cap_cycle": e_cap, "e_cap_cal": e_cap,
            "shi_fit":     fit,
            "T_C":         T_CELL_C,
        })
        if prod:
            # the shipped block returns d/dE_cap, which is -(1/E) times the slope
            shipped_slope = -prod["total"] * e_cap
            rel = abs(shipped_slope - ref["total"]) / abs(ref["total"])
            print(f"\n  shipped gradient, rescaled to the slope convention: "
                  f"{shipped_slope:.6f}")
            print(f"  analytic total slope                              : "
                  f"{ref['total']:.6f}")
            print(f"  relative difference                               : {rel:.3e}")

            # The two sides use different but equivalent expressions for the depth
            # term: k4 * Phi(delta) here, Phi'(delta) * delta in the shipped block.
            # phi_shi_cycle clips delta at 1e-9, so a record of depth exactly zero
            # carries a small cost here and none there. Measure that contribution
            # instead of asserting it explains the residual.
            d_all = np.array([c["dod"]      for c in cyc_list], dtype=float)
            q_all = np.array([c["count"]    for c in cyc_list], dtype=float)
            s_all = np.array([c["soc_mean"] for c in cyc_list], dtype=float)
            zmask = d_all < ZERO_DEPTH_TOL
            if zmask.any():
                phi_z = np.array([float(phi_shi_cycle(dd, ss, T_CELL_C, p))
                                  for dd, ss in zip(d_all[zmask], s_all[zmask])])
                zero_contrib = float(np.sum(q_all[zmask] * phi_z
                                            * (p.k4 + k_sigma * s_all[zmask])))
            else:
                zero_contrib = 0.0
            zero_rel = abs(zero_contrib) / abs(ref["total"])
            print(f"  zero-depth records {int(zmask.sum())}, contributing "
                  f"{zero_contrib:.3e} to the reference here")
            print(f"  and nothing to the shipped block, i.e. {zero_rel:.3e} relative.")
            print("  If that is the same order as the relative difference above, the")
            print("  residual is the clipping of zero-depth records and not a")
            print("  disagreement between the two expressions.")
            line["shipped_slope"] = shipped_slope
            line["shipped_relerr"] = rel
            line["zero_depth_records"] = int(zmask.sum())
            line["zero_depth_contrib"] = zero_contrib
            line["zero_depth_rel"] = zero_rel

        rows.append(line)

    if not rows:
        raise SystemExit("every case failed the guard; nothing written")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "fd_verification_summary.csv", index=False)
    print(f"\nsaved figures + summary to {OUT_DIR}")


if __name__ == "__main__":
    main()