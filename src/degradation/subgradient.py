"""
Exact per-timestep sub-gradient of the rainflow degradation cost.
================================================================

Replaces build_half_cycle_map + compute_subgradient, previously in shi.py and retained there, unused, for the record.
Why the attribution map had to go
---------------------------------
Shi et al. define index sets T_{v_i} and T_{w_j} that partition the horizon into half-cycle legs. The rainflow library reports each cycle as a pair of ADJACENT
reversals (i_start, i_end), so [i_start, i_end] is a single monotone leg. A full cycle has two legs, and the second one lies inside the FOLLOWING segment and is
never reported. Example: 0.2 -> 0.5 -> 0.4 -> 0.6. The full cycle of depth 0.1 is the segment 0.5 -> 0.4 together with the first 0.1 of the 0.4 -> 0.6 rise. No
index interval can express that. An attribution map built from [i_start, i_end] therefore cannot reproduce Shi's partition, and no choice of sort order fixes it.

The exact construction
----------------------
Perturbing the charging power c_t raises the energy trace by dt*eta_in*h at every step AFTER t. A cycle's depth changes if and only if its two turning points
straddle t: the later one moves, the earlier one does not. Cycles entirely after t shift bodily and keep their depth. Cycles entirely before t are untouched.

With f = E * B * sum_i n_i * Phi(delta_i)  and  delta_i = |e[b_i] - e[a_i]| / E,

    df/dc_t = + B * dt * eta_in  * G(t)
    df/dd_t = - B * dt / eta_out * G(t)

    G(t) = sum over cycles i with a_i <= t < b_i  of  n_i * Phi'(delta_i) * s_i
    s_i  = sign(e[b_i] - e[a_i]),   a_i = min(i_start, i_end),  b_i = max(...)

G is a suffix indicator sum, so it is one cumulative sum over a difference array:
O(cycles + timesteps), no per-timestep loop, no ownership, no junction rule.

Two properties fall out for free:

  1. The efficiency identity |df/dd_t| / |df/dc_t| = 1 / (eta_in * eta_out) holds POINTWISE and EXACTLY at every t, for any cycle set, because G(t) is common
     to both. It is a real test of the efficiency wiring, unlike a ratio of means over the horizon, which only recovers the coefficient ratio when the charging
     and discharging depth distributions happen to coincide.

  2. Shi Eqs. 17-18 are the special case in which exactly one half-cycle straddles t with n_i = 0.5. Where half-cycles nest, more than one straddles t and the
     sum has more than one term.

Depth-only by construction
--------------------------
Phi(delta) = k3 * delta^k4 carries no mean-SoC or temperature factor. That is deliberate and structural, not an omission:

  - Convexity. Shi Theorem 1 is stated for a stress function of cycle depth alone.
    Multiplying by S_sigma(sigma_i) puts the cost outside the theorem, because sigma_i is a function of the SoC profile.
  - Locality. Cycle DEPTHS are invariant to a uniform shift of the SoC trajectory.
    Cycle MEANS are not. With an S_sigma factor, df/dc_t acquires a contribution from every cycle downstream of t, and on an annual dispatch that non-local term
    exceeds the local depth term by two orders of magnitude.

S_sigma and S_T stay in the Xu reporting path and in the capacity gradient dDegCost/dE_cap, neither of which carries a convexity claim.

Validation (verify_subgradient_exact.py)
----------------------------------------
    smooth timesteps          : median 1.6e-07 % vs central difference
    rainflow topology changes : matches a one-sided derivative to 8e-05 %
    efficiency identity       : exact at every timestep, to 10 decimals
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from degradation.xu import (
    XuModelParams,
    XU_LMO,
    ShiPolynomialFit,
    _DEFAULT_SHI_FIT,
    phi_shi_prime,
)

# --------------------------------------------------------------------------
# Frozen-dispatch capacity gradient.
#
# Pure extraction of the inline block at lines 721-750 of
# run_battery_xu_shi_degradation_v5_6_RTE_test.py. The arithmetic is unchanged.
# The three helpers are imported from shi rather than xu
# so the call is identical to the run script's. The two module families define
# the same functions with different signatures; they agree numerically today and
# nothing enforces that they keep agreeing.
# --------------------------------------------------------------------------
from degradation.shi import (
    s_soc as _s_soc_shi,
    s_temp as _s_temp_shi,
    phi_shi_prime_with_stress as _phi_prime_stress,
)


def dDegCost_dEcap_terms(
    dods,
    counts,
    soc_means,
    storage_e,
    fd_calendar,
    e_cap_cycle,
    e_cap_cal=None,
    shi_fit=None,
    T_C: float = 25.0,
) -> Dict[str, float]:
    """Frozen-dispatch capacity gradient of annual degradation, per MWh of E_cap.

    Raising E_cap at a fixed energy dispatch rescales the normalised SoCtrajectory: every cycle amplitude and cycle mean falls as 1/E_cap, and so
    does the year's mean SoC. The response has a cycle part and a calendar part. 

        cycle    = sum_i n_i [ Phi'(d_i) S_sigma S_T (-d_i/E)
                               + Phi(d_i) S_sigma S_T k_sigma (-s_i/E) ]
                 = -(1/E) sum_i f_di (k4 + k_sigma s_i)
        calendar = -(1/E) k_sigma sigma_bar f_cal

    Multiply the total by lambda_repl * factor * scale to obtain dDegCost/dE_cap in EUR per MWh of capacity, as the run script does.

    Args:
        dods, counts, soc_means: rainflow arrays counted at e_cap_cycle.
        storage_e:   stored energy trace [MWh].
        fd_calendar: the period calendar term f_cal, before annualisation.
        e_cap_cycle: capacity the cycles were normalised by.
        e_cap_cal:   capacity used for the mean SoC. Defaults to e_cap_cycle.
                     The run script passes e_cap_eff here and e_cap1 above; they
                     should be the same number, and the caller should say so.
        shi_fit:     ShiPolynomialFit supplying k3 and k4.
        T_C:         cell temperature.

    Returns:
        dict with cycle, calendar, total, sigma_bar.
    """
    if shi_fit is None:
        shi_fit = _DEFAULT_SHI_FIT
    if e_cap_cal is None:
        e_cap_cal = e_cap_cycle

    d = np.asarray(dods, dtype=float)
    n = np.asarray(counts, dtype=float)
    s = np.asarray(soc_means, dtype=float)

    if d.size == 0:
        cycle = 0.0
    else:
        phi_full = (shi_fit.k3 * d ** shi_fit.k4
                    * _s_soc_shi(s) * _s_temp_shi(T_C))
        phi_prime = _phi_prime_stress(d, s, T_C, shi_fit.k3, shi_fit.k4)
        dphi = (phi_prime * (-d / e_cap_cycle)
                + phi_full * XU_LMO.k_sigma * (-s / e_cap_cycle))
        cycle = float(np.sum(n * dphi))

    sigma_bar = float(np.mean(np.asarray(storage_e, dtype=float))) / e_cap_cal
    calendar = -XU_LMO.k_sigma * sigma_bar * float(fd_calendar) / e_cap_cal

    return {"cycle": cycle, "calendar": calendar,
            "total": cycle + calendar, "sigma_bar": sigma_bar}

def rainflow_depth_sensitivity(
    storage_e: np.ndarray,
    cycles: List[Dict],
    k3: float,
    k4: float,
) -> np.ndarray:
    """G(t): the summed depth response of every cycle straddling time step t.

    G(t) = sum_{i : a_i <= t < b_i}  n_i * Phi'(delta_i) * sign(e[b_i] - e[a_i])

    Built as a cumulative sum over a difference array, so the cost is O(len(cycles) + len(storage_e)) with no per-timestep Python loop.

    Degenerate cycles (a_i == b_i, or zero depth) contribute nothing and are skipped. Under the old attribution map these were exactly the cycles that
    captured ownership and drove Phi' to zero.
    """
    e = np.asarray(storage_e, dtype=float)
    n = len(e)
    diff = np.zeros(n + 1, dtype=np.float64)

    for c in cycles:
        a = int(min(c["i_start"], c["i_end"]))
        b = int(max(c["i_start"], c["i_end"]))
        a = max(a, 0)
        b = min(b, n - 1)
        if a >= b:
            continue
        rise = e[b] - e[a]
        if rise == 0.0:
            continue
        s = 1.0 if rise > 0.0 else -1.0
        w = (float(c["count"])
             * float(phi_shi_prime(c["dod"], k3, k4))
             * s)
        diff[a] += w
        diff[b] -= w

    return np.cumsum(diff)[:n]


def compute_subgradient(
    storage_e: List[float],
    cycles: List[Dict],
    dt_hours: float,
    battery_replacement_cost_per_MWh: float,
    eff_in: float = 1.0,
    eff_out: float = 0.85,
    shi_fit: Optional[ShiPolynomialFit] = None,
) -> Dict:
    """Exact per-timestep sub-gradient of the rainflow degradation cost.

    Args:
        storage_e: Battery energy [MWh], the trace the cycles were counted on.
        cycles:    Output of rainflow_cycle_counting() on that same trace and the same e_cap. If the two disagree, every normalized depth is
                   wrong and so is everything below.
        dt_hours:  Timestep duration [h].
        battery_replacement_cost_per_MWh: B [EUR/MWh].
        eff_in, eff_out: SHIPP convention, both in (0, 1].
        shi_fit:   ShiPolynomialFit from fit_shi_polynomial(soc_min, soc_max).

    Returns:
        dfdc               (n,) signed  df/dc_t at EVERY timestep
        dfdd               (n,) signed  df/dd_t at EVERY timestep
        subgrad_charge     (n,) dfdc masked to timesteps where the trace rises
        subgrad_discharge  (n,) dfdd masked to timesteps where the trace falls
        subgrad_combined   (n,) subgrad_discharge - subgrad_charge
        G                  (n,) the depth-sensitivity sum, for inspection
        n_straddled        (n,) how many cycles straddle each timestep
        shi_fit            provenance
    """
    e = np.asarray(storage_e, dtype=float)
    n = len(e)
    B = float(battery_replacement_cost_per_MWh)
    tau = float(dt_hours)
    _fit = shi_fit if shi_fit is not None else _DEFAULT_SHI_FIT

    G = rainflow_depth_sensitivity(e, cycles, _fit.k3, _fit.k4)

    dfdc = B * tau * eff_in * G
    dfdd = -B * tau / eff_out * G

    # Direction of the trace, used only to mask the two views. It plays no in the gradient itself.
    slope = np.diff(e, append=e[-1])
    rising = slope > 0
    falling = slope < 0

    subgrad_charge = np.where(rising, dfdc, 0.0)
    subgrad_discharge = np.where(falling, dfdd, 0.0)

    # How many cycles straddle each step. Under Shi's assumptions this is 1.
    # Where half-cycles nest it is larger, which is precisely what the old single-owner attribution could not represent.
    cnt = np.zeros(n + 1, dtype=np.int32)
    for c in cycles:
        a = int(min(c["i_start"], c["i_end"]))
        b = int(max(c["i_start"], c["i_end"]))
        a, b = max(a, 0), min(b, n - 1)
        if a >= b:
            continue
        cnt[a] += 1
        cnt[b] -= 1
    n_straddled = np.cumsum(cnt)[:n]

    return {
        "dfdc": dfdc,
        "dfdd": dfdd,
        "subgrad_charge": subgrad_charge,
        "subgrad_discharge": subgrad_discharge,
        "subgrad_combined": subgrad_discharge - subgrad_charge,
        "G": G,
        "n_straddled": n_straddled,
        "n_cycles": len(cycles),
        "shi_fit": _fit,
    }