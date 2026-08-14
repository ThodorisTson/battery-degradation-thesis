# KNOWN ISSUE: "Failed error check in solve_lp_pyomo"

**Status:** Diagnosed, quantified, benign within a stated envelope (see tripwires).  
**Last verified:** baseline, sizing sweep, and window sweep runs, June 2026.
**Source files:** `kernel_pyomo.py` and `components.py` in SHIPP v1.2.0 (installed dependency, see SHIPP_MODIFICATIONS.md).
---

## TL;DR (read this first, do not re-diagnose from scratch)

This warning is a **residual-check artifact of SHIPP's relaxed LP storage formulation**, not a wrong dispatch and not a solver failure. It fires on essentially every `solve_lp_pyomo` call.

- **Revenue and NPV are correct.** They are computed from dispatch power and prices, which the slack does not touch.
- **fd (degradation) is correct to within < 0.25%.** Verified by direct comparison (see Evidence).
- **Raw cycle count / EFC is inflated** (up to ~16% at small E_cap) and must be reported as an approximate upper bound.
- **The warning is solver-agnostic.** It is not caused by Gurobi, by p_min, by missing price hours, or by efficiency convention. See "Red herrings."

If you are seeing this on a standard WP2 run near the optima, it is the known issue. Stop, confirm the signature, and move on. If you are outside the verified envelope, run the triage checklist before trusting fd.

---

## Console signature

Per solve you get, in order:

```text
Failed error check
Error : <error_in_unit0>   <error_out_unit0>
Error : <error_in_unit1>   <error_out_unit1>
.../kernel_pyomo.py:326: RuntimeWarning: Failed error check in solve_lp_pyomo
  warnings.warn('Failed error check in solve_lp_pyomo', RuntimeWarning)

  ## Upstream status

SHIPP v1.2.1 (`10bd96e`) added a storage model and storage losses check to
`solve_lp_sparse` and a test for energy conservation. Whether that addresses
this warning in `solve_lp_pyomo` has not been verified, since all results here
were produced against v1.2.0.