# Modifications to SHIPP

This repository depends on SHIPP, a hybrid power plant sizing framework by Dr. Jenna Iori. The work in this thesis required four additions to one function in SHIPP. This document records them so the dependency is fully traceable.

## Baseline and pinned version

| | |
|---|---|
| Baseline | SHIPP v1.2.0, commit `b0b4c0b` |
| Pinned in `requirements.txt` | `<COMMIT_SHA>` |
| Files differing from baseline | `src/shipp/kernel_pyomo.py` only |
| Size of difference | 42 insertions, 5 deletions |

Every other file under `src/shipp` is byte-identical to the baseline, verified with:

```bash
git diff --shortstat b0b4c0b <PINNED_SHA> -- src/shipp
```

## Why this baseline rather than the current release

Upstream SHIPP has advanced since `b0b4c0b`. Two of those commits would change the results reported in this thesis:

- `7df20be` alters the interpretation of `n_year` in `components.py`, changing the number of discounted operating years in the net present value calculation.
- `577aa45` changes the initial state-of-charge constraint from an equality to an inequality.

All results in the thesis were produced against `b0b4c0b`. The pin is therefore fixed at that baseline and the version is stated explicitly in the thesis text, rather than rerunning the full parameter sweep against a moving dependency.

## The four additions

All four are optional arguments to `solve_lp_pyomo` with defaults that reproduce the unmodified behaviour exactly. Existing SHIPP code calling this function is unaffected.

### 1. State-of-charge upper bound

New arguments `soc_max1: float = 1.0` and `soc_max2: float = 1.0`.

```python
# before
def rule_e_max1(model, i):
    return model.e_vec1[i] <= model.e_cap1

# after
def rule_e_max1(model, i):
    return model.e_vec1[i] <= model.e_cap1 * soc_max1
```

The same change applies to `rule_e_max2`.

**Note on the lower bound.** No change was required. SHIPP's existing bound, `e_vec1[i] >= e_cap1 * (1 - stor1.dod)`, gives the thesis lower bound directly when the depth-of-discharge parameter is set to `d = 1 - sigma_min`. The operating window is therefore obtained through one code change (upper bound) and one parameter choice (lower bound).

### 2. Inter-year state-of-charge pinning

New argument `e_start1: float = None`. When supplied, the initial state of charge is fixed, allowing consecutive simulated years to be chained without resetting the battery.

```python
model.e_start_end1 = pyo.Constraint(expr = model.e_vec1[0] == model.e_vec1[n])
if e_start1 is not None:
    model.e_fix_start1 = pyo.Constraint(expr = model.e_vec1[0] == e_start1)
```

The periodicity constraint is retained unchanged.

### 3. Dual price extraction

New argument `return_duals: bool = False`. When true, Pyomo `dual` and `rc` suffixes are declared before the solve so the solver populates shadow prices, and the values are collected afterwards into `os_res.dual_prices`.

The dual on the lower state-of-charge bound is the marginal revenue of stored energy at each time step. Combined with the degradation sub-gradient it forms the outer-loop gradient described in the thesis.

Extraction is wrapped in `try`/`except`. On failure a `RuntimeWarning` is issued and `dual_prices` is set to `None`.

### 4. Final state of charge

`os_res.soc_final` is set from the last state-of-charge value, so the caller can chain years without re-reading the full trajectory.

## Upstream status

These changes were offered to the SHIPP maintainer as a pull request against the `feature_degradation` branch: `<PR_URL>`.
