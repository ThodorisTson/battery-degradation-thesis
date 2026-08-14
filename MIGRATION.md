# Migration checklist

Working list for moving code out of the SHIPP fork into this repository.
Delete this file once the migration is complete.

## Source repository

`ThodorisTson/shipp`, branch `thodoris-degradation-thesis`, commit `4f090b1`.
Everything below lives under `experiments/PyWake/WP2/Degradation/`.

## File moves

Library modules keep their names. Every import in the codebase is a flat
top-level import, so renaming them would force edits across eight files for
no functional gain.

| From (fork) | To (this repo) | Rename |
|---|---|---|
| `degradation_xu.py` | `src/degradation_xu.py` | no |
| `degradation_shi.py` | `src/degradation_shi.py` | no |
| `degradation_subgradient.py` | `src/degradation_subgradient.py` | no |
| `degradation_plots.py` | `src/degradation_plots.py` | no |
| `degradation_plots_multiyear.py` | `src/degradation_plots_multiyear.py` | no |
| `wp2_common.py` | `src/wp2_common.py` | no |
| `wp2_econ.py` | `src/wp2_econ.py` | no |
| `thesis_style.py` | `src/thesis_style.py` | no |
| (new) | `src/paths.py` | new file |

Entry points are renamed. Nothing imports them, so this costs no edits.

| From (fork) | To (this repo) |
|---|---|
| `run_battery_xu_shi_degradation_v5_6_RTE_test.py` | `scripts/run_baseline.py` |
| `run_battery_xu_shi_degradation_v5.4 (+Xu comp_short).py` | `scripts/run_xu_shi_comparison.py` |
| `Outer Loop Tests/planB_2d_parameter_sweep_a_npv.py` | `scripts/run_sweep_2d.py` |
| `Outer Loop Tests/path3.py` | `scripts/run_nlp_monolithic.py` |
| `Outer Loop Tests/plot_sweep_thesis_v2.py` | `figures/plot_sweep.py` |
| `Outer Loop Tests/plot_path3_active.py` | `figures/plot_nlp_failure.py` |
| `verify_*.py` | `verification/` (names kept) |
| `plot_*.py` | `figures/` (names kept) |

## Configuration and data

| From | To |
|---|---|
| `WP2_HPP.yaml`, `WP2_Battery.yaml`, `WP2_Wind_Farm.yaml`, `WP2_Wind_Resource.yaml`, `WP2_Cables.yaml` | `config/` |
| `wind_resource_2022_era5_90m.yaml` | `data/` |
| `dk1_prices_2019.csv`, `dk1_prices_2022.csv` | `data/` |

Total data payload is about 1 MB, so all of it is committed and the
repository is runnable from a clean clone.

Note: `WP2_Wind_Resource.yaml` ends with `!include wind_resource_2022_era5_90m.yaml`,
and `WP2_HPP.yaml` includes the other four. Once the YAMLs move to `config/`
and the time series moves to `data/`, that include path becomes
`../data/wind_resource_2022_era5_90m.yaml`. Verify the loader in
`wp2_common.py` resolves includes relative to the including file. If it
resolves relative to the working directory instead, keep the time series in
`config/` alongside the others.

## Path edits required

Each entry point currently defines its own paths from `SCRIPT_DIR`. Replace
those blocks with imports from `paths`.

`scripts/run_baseline.py`, around line 130:

```python
# before
SCRIPT_DIR   = Path(__file__).parent
HPP_YAML     = SCRIPT_DIR / "WP2_HPP.yaml"
PRICE_CSV    = SCRIPT_DIR / "dk1_prices_2019.csv"
RESULTS_ROOT = SCRIPT_DIR / "Results" / "RTE Tests"

# after
from paths import HPP_YAML, PRICE_CSV_2019 as PRICE_CSV, results_dir
RESULTS_ROOT = results_dir("baseline")
```

`scripts/run_xu_shi_comparison.py`, around line 70: same pattern, with
`PRICE_CSV_2022` and `results_dir("xu_shi_comparison")`.

`scripts/run_sweep_2d.py`, around line 43:

```python
# before
SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "Plan B Results"

# after
from paths import results_dir
OUTPUT_DIR = results_dir("sweep_2d")
```

`scripts/run_nlp_monolithic.py`, around lines 62 and 989: replace
`_SCRIPT_DIR` / `_PARENT_DIR` with `results_dir("nlp_monolithic")`, and the
`HPP_YAML` search loop at line 1016 with the `paths.HPP_YAML` constant. The
`sys.path` insertion at line 65 becomes unnecessary once the package is
installed, but is harmless if left.

`figures/plot_sweep.py`, line 485: `Path(__file__).parent / "Plan B Results"`
becomes `results_dir("sweep_2d")`.

`figures/plot_nlp_failure.py`: the `thesis_style.py` search loop at lines
33-40 becomes a plain `from thesis_style import ...` once installed.

`src/degradation_xu.py` line 1036 and `src/degradation_shi.py` line 804 both
search for `WP2_Battery.yaml` next to and above themselves. Replace with
`from paths import BATTERY_YAML`.

## Output directory names

Folder names containing spaces (`Plan B Results`, `Degradation Plots`,
`RTE Tests`, `Outer Loop Tests`) become lowercase, no spaces, under
`results/`. Anything reading those folders must be updated in the same pass.

## Verification before the first push

1. Fresh clone into a new folder.
2. `python -m venv .venv` and `.venv\Scripts\activate`.
3. `pip install -r requirements.txt`.
4. `python -c "import degradation_xu, wp2_common, paths; print(paths.REPO_ROOT)"`.
5. `python -c "import shipp.kernel_pyomo as k; import inspect; print('soc_max1' in inspect.signature(k.solve_lp_pyomo).parameters)"` returns `True`.
6. `python scripts/run_baseline.py` and compare the output against the same
   run from the fork. The numbers must match exactly.
7. `python -m wp2_econ` runs the built-in self-test and prints ALL CHECKS PASSED.

Step 6 is the one that matters. Do not push before it passes.
