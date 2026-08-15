# Modeling of Battery Degradation for Integrated Design of Storage Systems

Code accompanying the MSc thesis of the same title (Sustainable Energy
Technology, Wind Energy Section, Delft University of Technology, 2026).

The thesis investigates how explicitly accounting for battery degradation
changes the optimal operating strategy and the optimal battery size for a hybrid
wind and battery power plant. The test case is the IEA Wind Task 50 onshore
reference site in western Denmark, dispatched against DK1 day-ahead prices.

This repository contains only the code written for the thesis. The underlying
sizing and dispatch framework, SHIPP, is a separate package by Dr. Jenna Iori
and is installed as a dependency rather than copied in.

## Contents

| Path | Description |
|---|---|
| `src/degradation/` | Degradation models, sub-gradient, site setup, economics, shared paths |
| `scripts/` | Entry points for the three investigations |
| `figures/` | Figure scripts, grouped by what they read |
| `verification/` | Checks reported in the appendices |
| `config/` | Site, turbine, battery, and cable configuration |
| `data/` | Price and wind time series |
| `results/` | Frozen simulation output, tracked in version control |
| `docs/` | Known issues |
| `tools/` | Migration helpers |

## SHIPP dependency

| | |
|---|---|
| Repository | https://github.com/jennaiori/shipp |
| Branch | `feature_degradation` |
| Commit | `d53a657` |

The pin is in `requirements.txt`. Every result in this thesis was produced
against this commit: SHIPP v1.2.0 plus four optional arguments to
`solve_lp_pyomo` (`soc_max1`, `soc_max2`, `e_start1`, `return_duals`) added for
this work and merged upstream as pull request #6 on 14 August 2026. The merged
kernel is byte-identical to the version used for the results.

SHIPP `main` has since moved to v1.2.1, which changes the `n_year` convention in
`components.py` and relaxes the initial state-of-charge constraint to an
inequality. The pin is deliberate. See
[`SHIPP_MODIFICATIONS.md`](SHIPP_MODIFICATIONS.md) for the full list of changes
and the reasoning.

## Installation

```bash
git clone https://github.com/ThodorisTson/battery-degradation-thesis.git
cd battery-degradation-thesis
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Dependency versions are pinned to the environment that produced every result:
Python 3.13, numpy 2.4.1, pandas 2.3.3, scipy 1.17.0, matplotlib 3.10.8,
xarray 2025.12.0, pyomo 6.10.0, py_wake 2.6.18, rainflow 3.2.0, PyYAML 6.0.3,
gurobipy 13.0.1, PyMuPDF 1.28.2, Pillow 12.3.0.

Verify the installation:

```bash
python -c "from degradation import paths; print(paths.REPO_ROOT)"
python -m degradation.economics
```

The second command runs a self-test of the discounting, efficiency, and revenue
conventions and prints ALL CHECKS PASSED.

To change the SHIPP pin, edit `requirements.txt` and reinstall with
      `pip install --force-reinstall --no-deps "shipp @ git+..."`. Omitting
      `--no-deps` upgrades numpy, pandas, scipy and matplotlib past their pins.

### Solver

The dispatch LP is solved with Gurobi through Pyomo. A licence is required; a
free academic licence is available at https://www.gurobi.com/academia/. Place
`gurobi.lic` in your home directory. `gurobipy` ships a size-limited licence
sufficient only for small models, and a full year of hourly dispatch is about
44,000 variables.

Without a licence, set `pyo_solver = "none"` at the top of any run script to
fall back to the SciPy sparse solver. That path does not support the
depth-of-discharge constraint and is limited to roughly six months of
simulation, so it reproduces the structure of the results but not their values.

## Investigative approaches

Chapter 3 defines one shared evaluation pipeline and three approaches built on
it, differing in how the sizing decision couples to that pipeline.

### The shared evaluation pipeline

```bash
python scripts/run_baseline.py
```

Solves the dispatch LP at a fixed design, passes the state-of-charge trajectory
to the Xu and Shi degradation models, accumulates capacity fade over 20 years
re-solving each year at the degraded capacity, and triggers replacement at the
state-of-health threshold. The LP itself carries no degradation term.

This is not one of the three approaches; it is what all three evaluate. Run
alone at the 150 MW / 300 MWh reference design it gives the baseline results of
Section 4.3, and it writes the trajectory and LP duals that Appendix C reads.

### 1. Parameter sweep, in two stages

Stage 1 sweeps energy and power capacity to find the optimal size:

```bash
python scripts/run_sizing_sweep.py
```

Stage 2 fixes that size and sweeps the state-of-charge operating window:

```bash
python scripts/run_window_sweep.py
```

Stage 1 evaluates three scenarios per grid point: no degradation, Xu, and Shi.
Stage 2 runs seven windows at the Xu optimum, all reported on Xu. The 30-70%
window is skipped, because its polynomial fit returns an exponent of 0.908 and
is therefore not convex. Together the stages give the headline result: the
optimum moves from 800 MWh / 250 MW to 550 MWh / 175 MW, about 30%. Runtime is
several hours each.

### 2. Monolithic non-linear program

```bash
python scripts/run_nlp_monolithic.py --year 2022 --slot D43
```

Solves one day of dispatch and degradation as a single nonlinear program,
reported as a negative result. The degradation gradient is discontinuous
wherever the cycle set re-pairs, so across the 356 active days of DK1 2022 the
solver certifies an optimum on 185, or 52%. On 168 of those the dispatch is no
better than the revenue-only LP, and 87% of the positive objective difference
across the year comes from days that never converged. `--all-days` runs the
full year.

### 3. Gradient-based sizing, analytical components only

No run script: the sizing search is not implemented. What is implemented and
verified are the components it would need, namely the frozen-dispatch capacity
gradient, the per-timestep degradation sub-gradient, and the LP duals on the
state-of-charge bounds. All three are checked against finite differences in
`verification/` and reported in Appendix C.

What remains is the re-dispatch term and a step-size rule, both identified as
future work.

## Results

`results/` is tracked in version control rather than ignored. The simulation
runs are frozen and a full sweep takes hours, so the outputs the figures read
from are committed, and every figure can be regenerated from a clean clone
without re-running anything.

| Folder | Produced by |
|---|---|
| `results/baseline/xu/` | `run_baseline.py` |
| `results/sizing_sweep/` | `run_sizing_sweep.py` |
| `results/window_sweep/` | `run_window_sweep.py` |
| `results/nlp_monolithic_all_days/` | `run_nlp_monolithic.py --all-days` |
| `results/week_snapshot/` | `verification/verify_week_snapshot.py` |
| `results/verification/` | `verification/` scripts |

Re-running a script overwrites the committed output. Check `git status` before
committing after a run.

## Figures

Figure scripts are grouped by what they read, since that determines whether they
run from a clean clone. Each writes its output beside itself.

| Folder | Reads |
|---|---|
| `figures/concept/` | nothing: equations, synthetic traces, diagrams |
| `figures/from_data/` | `config/` or `data/` |
| `figures/from_results/` | `results/` |

Filenames do not encode figure numbers, since numbers move when chapters are
edited. The mapping is below.

| Figure | Script |
|---|---|
| _to be completed_ | |

## Verification

`verification/` reproduces the checks reported in the appendices: sub-gradient
exactness against finite differences, per-timestep attribution, energy
conservation over a representative week, and the rainflow read-back on
synthetic traces.

```bash
python verification/verify_week_snapshot.py
python verification/verify_tier1_degradation.py
python verification/verify_tier2_dispatch.py
```

## Data

All input data is included; nothing needs downloading.

| File | Source |
|---|---|
| `data/dk1_prices_2019.csv`, `data/dk1_prices_2022.csv` | ENTSO-E Transparency Platform, DK1 day-ahead prices |
| `data/wind_resource_2022_era5_90m.yaml` | ERA5 reanalysis via the Open-Meteo archive endpoint, grid cell 56.25 N 8.50 E, 100 m series scaled to 90 m hub height with a power-law exponent of 0.17 |
| `config/wind_farm.yaml` | IEA Wind Task 50 reference layout, 65 NREL 5 MW turbines |
| `config/battery.yaml` | Danish Energy Agency Technology Data for Energy Storage, sheet 180 |

Provenance for the wind series is documented in the header of
`config/wind_resource.yaml`.

## Known issues

`docs/KNOWN_ISSUE_failed_error_check_solve_lp_pyomo.md` explains the
"Failed error check in solve_lp_pyomo" warning that appears on every solve. It
is a residual-check artefact of SHIPP's relaxed LP storage formulation.
Revenue, NPV, and degradation are unaffected; raw cycle counts are inflated and
should be treated as an upper bound.

## Citation

_To be completed once the TU Delft repository record exists._

## Licence

Apache 2.0, matching SHIPP, on which this work depends. SHIPP is installed as a
dependency and is not redistributed here.

## Acknowledgements

Supervised by Dr. Jenna Iori, whose SHIPP framework provides the sizing and
dispatch optimization used throughout.
