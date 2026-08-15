# Migration record

What moved, what was renamed, and why. This documents the migration of the
thesis code out of a fork of SHIPP into this standalone repository, carried out
on 14 August 2026.

Source: `ThodorisTson/shipp`, branch `thodoris-degradation-thesis`, commit
`4f090b1`. Everything below lived under
`experiments/PyWake/WP2/Degradation/` unless stated otherwise.

The `WP2` prefix throughout the original code stood for work package 2, an
internal label with no meaning outside the group. It has been removed
everywhere except where it appears in archived comments.

---

## 1. Repository

| | |
|---|---|
| Name | `battery-degradation-thesis` |
| Package (import name) | `degradation` |
| Layout | `src/` with `pyproject.toml`, installed editable |
| SHIPP | dependency pinned at `jennaiori/shipp@d53a657`, not vendored |

Dependency versions are pinned to the environment that produced every result in
the thesis: Python 3.13.14, numpy 2.4.1, pandas 2.3.3, scipy 1.17.0,
matplotlib 3.10.8, xarray 2025.12.0, pyomo 6.10.0, py_wake 2.6.18,
rainflow 3.2.0, PyYAML 6.0.3, gurobipy 13.0.1.

---

## 2. Configuration

| From | To |
|---|---|
| `WP2_HPP.yaml` | `config/hpp.yaml` |
| `WP2_Battery.yaml` | `config/battery.yaml` |
| `WP2_Wind_Farm.yaml` | `config/wind_farm.yaml` |
| `WP2_Wind_Resource.yaml` | `config/wind_resource.yaml` |
| `WP2_Cables.yaml` | `config/cables.yaml` |

Content changes inside those files:

- All four `!include` lines in `hpp.yaml` updated to the new filenames.
- `wind_resource.yaml` now includes `../data/wind_resource_2022_era5_90m.yaml`,
  since the time series moved to `data/`. Verified to resolve: the loader
  resolves includes relative to the including file.
- `site.name` changed from `WP2_Denmark_Onshore` to `DK1_Denmark_Onshore`.
  It is consumed only as a label (`site.py` line 210, with a default).
  `DK1` names the price zone the dispatch runs against, which is more
  informative than the work-package number.
- Informal references to the supervisor by name replaced with references to
  the SHIPP reference HPP schema.
- Comments naming run scripts that no longer exist updated.

---

## 3. Data

| From | To |
|---|---|
| `wind_resource_2022_era5_90m.yaml` | `data/` |
| `dk1_prices_2019.csv` | `data/` |
| `dk1_prices_2022.csv` | `data/` |

About 1 MB in total, so all of it is committed and the repository is runnable
from a clean clone.

---

## 4. Library modules

Fork root `Degradation/` to `src/degradation/`.

| From | To | Import becomes |
|---|---|---|
| `degradation_xu.py` | `xu.py` | `from degradation.xu import ...` |
| `degradation_shi.py` | `shi.py` | `from degradation.shi import ...` |
| `degradation_subgradient.py` | `subgradient.py` | `from degradation.subgradient import ...` |
| `degradation_plots.py` | `plots.py` | `from degradation.plots import ...` |
| `degradation_plots_multiyear.py` | `plots_multiyear.py` | `from degradation.plots_multiyear import ...` |
| `wp2_common.py` | `site.py` | `from degradation.site import quick_setup` |
| `wp2_econ.py` | `economics.py` | `from degradation.economics import ...` |
| `thesis_style.py` | `style.py` | `from degradation.style import ...` |
| — | `paths.py` | new, see section 7 |
| — | `__init__.py` | new |

Twenty-one import statements were rewritten across seven files by
`tools/rename_imports.py`, which prints every change and runs as a dry run by
default.

Two modules located `WP2_Battery.yaml` by searching their own directory and
its parent. Both now use `from degradation.paths import BATTERY_YAML`. Only
the self-tests were affected.

One interactive `input()` prompt in `xu.py` was replaced with a default read
from `config/battery.yaml`, so the self-test runs unattended.

---

## 5. Entry points

| From | To |
|---|---|
| `run_battery_xu_shi_degradation_v5_6_RTE_test.py` | `scripts/run_baseline.py` |
| `run_battery_xu_shi_degradation_v5.4 (+Xu comp_short).py` | `scripts/run_window_sweep.py` |
| `Outer Loop Tests/planB_2d_parameter_sweep_a_npv.py` | `scripts/run_sizing_sweep.py` |
| `Outer Loop Tests/path3.py` | `scripts/run_nlp_monolithic.py` |

The old names encoded version numbers and internal shorthand. The new ones say
what each script does: `run_sizing_sweep` sweeps energy and power capacity;
`run_window_sweep` sweeps the state-of-charge operating window at fixed size.

`FILE_TAG`, which appears in every output filename:

| Script | From | To |
|---|---|---|
| `run_baseline.py` | `v56_rtetest` | `baseline` |
| `run_window_sweep.py` | `v54` | `window` |

Output filename prefixes in `run_sizing_sweep.py`:

| From | To |
|---|---|
| `planB_2d_sweep` | `sizing_sweep` |
| `planB_2d_report` | `sizing_report` |
| `planB_2d_npv_comparison` | `sizing_npv_comparison` |
| `planB_2d_margins` | `sizing_margins` |
| `planB_2d_slices` | `sizing_slices` |
| `planB_lifetime_sweep` | `sizing_lifetime_sweep` |
| `planB_lifetime_report` | `sizing_lifetime_report` |
| `planB_lifetime_npv` | `sizing_lifetime_npv` |

Docstrings were rewritten. The originals recorded version lineage and
differences from superseded files; they now describe what each script does.
Each retains a single line naming its fork origin, so the correspondence stays
traceable.

**Not renamed:** the identifiers `deg_cost_planB`, `npv_bat_planB`, and the
CSV and JSON keys `npv_bat_planB_EUR` and `npv_bat_planB_MEUR`. These are
column headers in committed result files. Renaming them would break the link to
data that cannot be regenerated before the defense.

---

## 6. Results

Simulation runs are frozen. A full sweep takes hours, so the outputs the thesis
figures read from are committed rather than ignored. This is why `.gitignore`
does not exclude `results/`.

| From | To |
|---|---|
| `Results/RTE Tests/Xu model/` | `results/baseline/xu/` |
| `Results/RTE Tests/Xu model/Degradation Plots/` | `results/baseline/xu/plots/` |
| `Outer Loop Tests/Plan B Results/` | `results/sizing_sweep/` |
| `Results/` (the `v54_*` files) | `results/window_sweep/` |
| `Outer Loop Tests/Results_Path3_AllDays/run_20260706_010250/` | `results/nlp_monolithic_all_days/run_20260706_010250/` |
| `Results/Week Snapshot/` | `results/week_snapshot/` |
| `Results/Verification/` | `results/verification/` |

Result files were renamed to match the new tags: `v56_rtetest` to `baseline`,
`v54_` to `window_`, and the `planB_` prefixes as in section 5. Timestamps in
filenames were left untouched, since `find_latest_csv()` sorts on them.

Only current runs were carried over:

- Baseline: `20260812_001610` (DK1 2019) and `20260812_000942` (DK1 2022),
  the most recent per price year.
- Window sweep: `20260813_163037_E550_P175`. The earlier `E475_P150` set
  predates the round-trip efficiency correction that moved the optimum.
- Sizing sweep: the lifetime files only. The annual-mode files predate the same
  correction and were left behind.
- Monolithic NLP: the full 365-day run, 732 files.

About 29 MB in total.

**Caution.** `e_cap_fixed.npy` and `storage_e_fixed.npy` carry no timestamp, so
any new run in the same folder overwrites them. This happened once during
testing and was recovered with `git checkout --`.

---

## 7. Paths

Every script previously defined its own paths from `Path(__file__).parent` or
its parent, so file locations were encoded in eight places. They now come from
`src/degradation/paths.py`:

```python
REPO_ROOT, CONFIG_DIR, DATA_DIR, RESULTS_DIR, FIGURES_DIR
HPP_YAML, BATTERY_YAML, WIND_FARM_YAML, WIND_RESOURCE_YAML, CABLES_YAML
PRICE_CSV_2019, PRICE_CSV_2022
results_dir(name)   # returns results/<name>, created if absent
require(path)       # returns path, or raises with the repository root named
```

`REPO_ROOT` is found by walking up for `pyproject.toml`, so it works from any
working directory.

Output directory names lost their spaces: `Plan B Results`, `Degradation
Plots`, `RTE Tests` and `Outer Loop Tests` became `sizing_sweep`,
`plots`, `baseline` and so on under `results/`.

**Figures are the exception.** Figure scripts write their output beside
themselves rather than to a central directory, so `Path(__file__).parent` is
correct there and was left in place.

---

## 8. Figures

Organised by what each script reads, since that determines whether it runs from
a clean clone.

| Folder | Contents |
|---|---|
| `figures/concept/` | self-contained: equations, synthetic traces, diagrams |
| `figures/from_data/` | reads `config/` or `data/` |
| `figures/from_results/` | reads `results/` |

Every thesis figure has a generating script. The two XDSM diagrams (Figures 3.5
and 3.12) are produced by `figures/concept/make_lp_nlp_xdsm.py`, so the
originally planned `figures/external/` folder was not needed and was removed.

Filenames do not encode figure numbers. Numbers move when chapters are edited,
and a filename that names the wrong figure is worse than one that is vague. Two
scripts were renamed on copy for this reason: `fig34_xu_sdelta_d2.py` became
`xu_sdelta_second_derivative.py` (Figure 3.7) and `fig35_phi_extrapolation.py`
became `phi_extrapolation.py` (Figure 3.8).

The mapping from thesis figure to script belongs in the README.

---

## 9. Not migrated

Left in the fork at `4f090b1`, which remains available:

- Superseded run script versions (v5.1 through v5.4 variants)
- `Old Attempts/`, `Older Attempts/`, `Greenlight/`, `Midterm/`
- Path 1 and Path 2 outer-loop experiments
- Result files predating the round-trip efficiency correction
- Exploratory and diagnostic scripts with no thesis output
- Presentation figures with no thesis counterpart

---

## 10. Verification performed

- All modules import; `python -m degradation.economics`, `-m degradation.xu`
  and `-m degradation.shi` pass their self-tests.
- `quick_setup(HPP_YAML)` loads 65 turbines, 300 MWh / 150 MW,
  SoC window 0.1 to 0.9, confirming the relative `!include` resolves.
- A 30-day `run_baseline.py` completed end to end: PyWake, the SHIPP LP through
  Gurobi with `soc_max1` and `return_duals`, rainflow, subgradient, and CSV and
  report output. Test output was removed afterwards.
- `solve_lp_pyomo` exposes `soc_max1`, `soc_max2`, `e_start1` and
  `return_duals`, confirming the pinned SHIPP commit carries the kernel changes.
- The four kernel changes were merged upstream on 14 August 2026 (pull request
  #6). The merged commit `d53a657` was diffed against the pull request head
  `16fd925` and found byte-identical, so the pin was repointed from the fork to
  Jenna's repository.

Not yet done: a full-year run reproducing the committed results number for
number.