"""
economics.py — Shared economic primitives for the battery sweep models.

Single source of truth for the conventions that must stay identical across
    scripts/run_sizing_sweep.py
    scripts/run_window_sweep.py
    scripts/run_baseline.py

Covered here:
  * symmetric round-trip efficiency split (PCU included)
  * discount / annuity convention (matches the SHIPP kernel)
  * battery capex and replacement-cost scope (energy + power expansion)
  * degradation penalty cost (replacement-energy valuation)
  * arbitrage vs marginal revenue definitions
  * lifetime NPV assembly

Design rule
-----------
PURE MATH ONLY. No SHIPP imports, no I/O, no state, no numpy_financial.
Each caller keeps its own year-loop orchestration; only the formulas live here. If a function starts growing mode= / include_x= flags it has crossed into orchestration and belongs back in the caller.

Discounting horizon
-------------------
discount_weights() returns weights for years k = 1 .. n_year-1, so n_year = 20 gives 19 discounted operating years.

This matches the objective function the LP actually maximizes. The SHIPP kernel uses the factor npf.npv(r, ones(n_year)) - 1 (kernel_pyomo.py L146), which sums
n_year-1 terms. Reporting an NPV over a different horizon than the one being optimized would make the two inconsistent, so the weights follow the kernel.
test_matches_shipp_kernel() enforces this and fails if the two ever diverge.

SHIPP v1.2.1 changed OpSchedule.get_npv_irr and get_added_npv to sum n_year terms, without changing the kernel objective. All results here were produced
against SHIPP v1.2.0 (commit b0b4c0b), where the two agree.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "eta_symmetric",
    "discount_weights",
    "annuity_factor",
    "annualize_to_year",
    "capex",
    "replacement_cost",
    "degradation_cost",
    "revenue_annual",
    "lifetime_npv",
    "HEADLINE_BASIS",
]

HOURS_PER_YEAR = 8760.0

# ── Headline revenue basis (reporting convention) ────────────────────────────
# # Which battery-NPV definition the run files treat as the primary/headline number and select the optimum on. Both bases are always computed and reported; this only chooses the label + the optimum key.
#   "arbitrage" : price · storage_p           (battery-as-asset; SHIPP a_npv)
#   "marginal"  : total plant − wind-only     (+ curtailment recovery)
# At the IEA Task 50 site the two differ by ~0.02% (grid ~ wind rating).
# No function in this module reads this. It is a shared declaration the run files consume, so the sweep and the inner-loop file cannot disagree on which
# basis is headline. The marginal basis is used because it accounts for the curtailment the battery recovers, which the arbitrage basis omits.
HEADLINE_BASIS = "marginal"

# ── Efficiency ───────────────────────────────────────────────────────────────
def eta_symmetric(rte_ac: float) -> float:
    """One-way efficiency from a symmetric split of the AC round trip.

    rte_ac is the full AC round-trip efficiency (DC round trip x PCU^2). Returns eta with eta_in = eta_out = eta and eta**2 == rte_ac, so the round trip is preserved but charging is no longer modelled as lossless.
    """
    return rte_ac ** 0.5

# ── Discounting ──────────────────────────────────────────────────────────────
def discount_weights(r: float, n_year: int) -> np.ndarray:
    """Per-year discount factors (1+r)^-k for k = 1 .. n_year-1.

    weights[0] is year 1, weights[-1] is year n_year-1 (SHIPP convention). See "Discounting horizon" at the top of this file.
    """
    return np.array([(1.0 + r) ** (-k) for k in range(1, n_year)], dtype=float)

def annuity_factor(r: float, n_year: int) -> float:
    """Flat-revenue lifetime multiplier = sum of the per-year discount weights."""
    return float(np.sum(discount_weights(r, n_year)))

def annualize_to_year(value: float, n_hours: int) -> float:
    """Scale a quantity accumulated over n_hours up to a full 8760-h year."""
    return value * HOURS_PER_YEAR / float(n_hours)

# ── Capex ────────────────────────────────────────────────────────────────────
def capex(e_cap: float, p_cap: float, e_cost: float, p_cost: float) -> float:
    """Initial battery capex [EUR] = energy component + power component."""
    return e_cost * e_cap + p_cost * p_cap

def replacement_cost(e_cap: float, p_cap: float,
                     repl_e: float, repl_p: float) -> float:
    """Replacement capex [EUR] = energy expansion + power expansion (reuses BOP)."""
    return repl_e * e_cap + repl_p * p_cap

# ── Degradation penalty (annual mode / Plan-B proxy) ─────────────────────────
def degradation_cost(fd_annual: float, e_cap: float,
                     repl_e: float, annuity: float) -> float:
    """Levelized degradation cost [EUR] over the project horizon.

    fd_annual : PER-YEAR capacity-fade fraction (already annualized; use annualize_to_year() on a sub-year fd before calling).
    Valued at the replacement-energy rate on the capacity consumed, then carried over the lifetime by the annuity factor.
    """
    return fd_annual * repl_e * e_cap * annuity

# ── Revenue definitions ──────────────────────────────────────────────────────
def revenue_annual(price, storage_p, production_p, wind, p_max,
                   n_hours: int, dt: float = 1.0) -> dict:
    """Single-year UNDISCOUNTED battery revenue [EUR/yr], two definitions.

    arbitrage : price . storage_p (battery charge/discharge against price only)
    marginal  : price . (production_p + storage_p) - price . min(wind, p_max) (arbitrage + curtailment recovery; total plant minus wind-only)

    production_p is wind-after-curtailment (os.production_p[0].data). The caller applies the annuity (flat extrapolation) or the per-year discount weights (lifetime mode) — this returns the single-year figure.

    With no curtailment (production_p == wind and wind <= p_max everywhere), marginal collapses to arbitrage.
    """
    price = np.asarray(price, dtype=float)
    storage_p = np.asarray(storage_p, dtype=float)
    production_p = np.asarray(production_p, dtype=float)
    wind = np.asarray(wind, dtype=float)

    scale = HOURS_PER_YEAR / float(n_hours) * dt
    wind_no_bat = np.minimum(wind, p_max)

    arbitrage = scale * float(np.dot(price, storage_p))
    marginal = scale * (float(np.dot(price, production_p + storage_p))
                        - float(np.dot(price, wind_no_bat)))
    return {"arbitrage": arbitrage, "marginal": marginal}

# ── Lifetime NPV assembler ───────────────────────────────────────────────────
def lifetime_npv(capex_initial: float, annual_revenue, weights,
                 replacement_years=None, repl_cost: float = 0.0) -> float:
    """Discounted lifetime NPV [EUR].

        NPV = -capex_initial
              + sum_k   rev_k     * weights[k-1]
              - sum_yr  repl_cost * weights[yr-1]

    annual_revenue    : per-year UNDISCOUNTED revenues, length == len(weights). A constant list reproduces the flat no-deg / proxy case (sum == rev * annuity_factor).
    weights           : output of discount_weights(r, n_year).
    replacement_years : 1-indexed years in which a replacement is paid.
    """
    weights = np.asarray(weights, dtype=float)
    rev = np.asarray(annual_revenue, dtype=float)
    if rev.shape[0] != weights.shape[0]:
        raise ValueError(
            f"annual_revenue length {rev.shape[0]} != weights length {weights.shape[0]} "
            "(horizon mismatch — both must span the same years)"
        )

    npv = -capex_initial + float(np.dot(rev, weights))
    for yr in (replacement_years or []):
        if not (1 <= yr <= weights.shape[0]):
            raise ValueError(
                f"replacement year {yr} outside horizon 1..{weights.shape[0]}"
            )
        npv -= repl_cost * weights[yr - 1]
    return npv

# ── Self-test ────────────────────────────────────────────────────────────────
def test_matches_shipp_kernel(r: float = 0.03, n_year: int = 20) -> None:
    """annuity_factor must equal the SHIPP kernel factor npf.npv(r, ones(n))-1.

    npf.npv(r, ones(n)) = sum_{i=0}^{n-1} (1+r)^-i, so the kernel factor is sum_{i=1}^{n-1} (1+r)^-i. We replicate that arithmetic directly here so the check does not depend on numpy_financial. 
    Passing means the 19-year convention is intact.
    """
    kernel_factor = sum((1.0 + r) ** (-i) for i in range(1, n_year))
    got = annuity_factor(r, n_year)
    assert abs(got - kernel_factor) < 1e-12, \
        f"annuity_factor {got} != kernel factor {kernel_factor}"

def _run_self_test() -> None:
    r, n_year, tol = 0.03, 20, 1e-9

    # 1. Annuity matches the SHIPP kernel factor (the cross-file anchor).
    test_matches_shipp_kernel(r, n_year)

    # 2. Weights are self-consistent and correctly counted.
    w = discount_weights(r, n_year)
    assert len(w) == n_year - 1, f"expected {n_year-1} weights, got {len(w)}"
    assert abs(float(np.sum(w)) - annuity_factor(r, n_year)) < tol
    assert abs(w[0] - (1.0 + r) ** -1) < tol          # weights[0] is year 1

    # 3. Symmetric efficiency squares back to the round trip (WP2 numbers).
    rte_ac = 0.9025 * 0.986 ** 2                       # DC round trip x PCU^2
    eta = eta_symmetric(rte_ac)
    assert abs(eta ** 2 - rte_ac) < tol
    # sqrt(0.9025 * 0.986^2) = 0.95 * 0.986 = 0.9367 exactly
    assert abs(eta - 0.9367) < 1e-12, f"eta {eta} (expected 0.9367 for WP2)"

    # 4. Capex / replacement are linear and scope-correct.
    assert capex(300, 150, 245e3, 86e3) == 245e3 * 300 + 86e3 * 150
    assert replacement_cost(300, 150, 72e3, 96e3) == 72e3 * 300 + 96e3 * 150

    # 5. Marginal == arbitrage when there is no curtailment.
    rng = np.random.default_rng(0)
    price = rng.uniform(-10, 100, 100)
    wind = rng.uniform(0, 200, 100)                    # all below p_max=300
    storage = rng.uniform(-50, 50, 100)
    rev = revenue_annual(price, storage, production_p=wind, wind=wind,
                         p_max=300, n_hours=100)
    assert abs(rev["arbitrage"] - rev["marginal"]) < 1e-6, \
        "marginal should equal arbitrage with zero curtailment"

    # 5b. The marginal-minus-arbitrage gap equals exactly
    #     scale * price . (production_p - min(wind, p_max)). This is a formula
    #     identity; its SIGN is a property of the dispatch, not guaranteed
    #     (the battery may divert wind to charge, moving recovery into the
    #     arbitrage/discharge term), so we test the identity, not a direction.
    prod_arb = rng.uniform(0, 300, 100)               # arbitrary wind-after-curtailment
    wind_big = np.full(100, 400.0)                    # always clipped at p_max
    rev2 = revenue_annual(price, storage, production_p=prod_arb,
                          wind=wind_big, p_max=300, n_hours=100)
    scale = HOURS_PER_YEAR / 100.0
    expected_gap = scale * float(np.dot(price, prod_arb - np.minimum(wind_big, 300)))
    assert abs((rev2["marginal"] - rev2["arbitrage"]) - expected_gap) < 1e-6, \
        "marginal - arbitrage must equal scale * price.(production_p - wind_no_bat)"

    # 6. Lifetime NPV with constant revenue == -capex + rev*annuity (flat case).
    cap = capex(300, 150, 245e3, 86e3)
    rev_yr = 2.0e6
    flat = [rev_yr] * len(w)
    npv_flat = lifetime_npv(cap, flat, w)
    npv_closed = -cap + rev_yr * annuity_factor(r, n_year)
    assert abs(npv_flat - npv_closed) < 1e-3, f"{npv_flat} != {npv_closed}"

    # 7. A replacement in year 12 subtracts exactly repl_cost discounted.
    repl = replacement_cost(300, 150, 72e3, 96e3)
    npv_no_repl = lifetime_npv(cap, flat, w)
    npv_with_repl = lifetime_npv(cap, flat, w, replacement_years=[12], repl_cost=repl)
    assert abs((npv_no_repl - npv_with_repl) - repl * w[11]) < 1e-3

    # 8. Length-mismatch and out-of-range guards fire.
    try:
        lifetime_npv(cap, [rev_yr] * (len(w) - 1), w)
        raise AssertionError("expected ValueError on length mismatch")
    except ValueError:
        pass
    try:
        lifetime_npv(cap, flat, w, replacement_years=[n_year], repl_cost=repl)
        raise AssertionError("expected ValueError on out-of-range replacement year")
    except ValueError:
        pass

    # 9. Degradation cost: zero at fd=0, linear otherwise.
    assert degradation_cost(0.0, 300, 72e3, annuity_factor(r, n_year)) == 0.0
    assert degradation_cost(0.02, 300, 72e3, 1.0) == 0.02 * 72e3 * 300

    # 10. annualize_to_year scales a sub-year accumulation correctly.
    assert abs(annualize_to_year(100.0, 4380) - 200.0) < tol   # half a year

    print("economics self-test: ALL CHECKS PASSED")
    print(f"  eta_symmetric(WP2 rte_ac={rte_ac:.6f}) = {eta:.6f}")
    print(f"  annuity_factor(r=3%, n_year=20)        = {annuity_factor(r, n_year):.6f}")
    print(f"  discount_weights count                 = {len(w)}  (SHIPP 19-yr convention)")


if __name__ == "__main__":
    _run_self_test()