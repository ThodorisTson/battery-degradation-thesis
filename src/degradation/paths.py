"""Single source of truth for every path in the repository.

Import from here instead of building paths relative to a script's own
location. Scripts can then be moved between folders without breaking, and
the directory layout is described in one file rather than in eight.

Usage:
    from paths import CONFIG_DIR, DATA_DIR, results_dir

    hpp_yaml  = CONFIG_DIR / "hpp.yaml"
    price_csv = DATA_DIR / "dk1_prices_2022.csv"
    out       = results_dir("baseline")     # created if absent
"""
from __future__ import annotations

from pathlib import Path

# Walk up from this file until the repository root is found, identified by
# pyproject.toml. Works whether the package is installed editable or the
# repository is used in place.
def _find_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(
        "Repository root not found: no pyproject.toml above " f"{start}"
    )


REPO_ROOT = _find_root(Path(__file__).resolve())

CONFIG_DIR = REPO_ROOT / "config"        # site, turbine, battery, cable configuration
DATA_DIR = REPO_ROOT / "data"            # price and wind time series
RESULTS_DIR = REPO_ROOT / "results"      # run output, not tracked by git
FIGURES_DIR = REPO_ROOT / "figures"      # figure scripts

# Configuration files, named so a typo fails at import rather than at runtime.
HPP_YAML = CONFIG_DIR / "hpp.yaml"
BATTERY_YAML = CONFIG_DIR / "battery.yaml"
WIND_FARM_YAML = CONFIG_DIR / "wind_farm.yaml"
WIND_RESOURCE_YAML = CONFIG_DIR / "wind_resource.yaml"
CABLES_YAML = CONFIG_DIR / "cables.yaml"

PRICE_CSV_2019 = DATA_DIR / "dk1_prices_2019.csv"
PRICE_CSV_2022 = DATA_DIR / "dk1_prices_2022.csv"


def results_dir(name: str) -> Path:
    """Return results/<name>, creating it if it does not exist."""
    d = RESULTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def require(path: Path) -> Path:
    """Return path, or raise with a message that says what is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Expected relative to repository root "
            f"{REPO_ROOT}. See README for the required data files."
        )
    return path
