r"""
plot_price_comparison_v2.py
===========================
Produces two standalone thesis figures from DK1 2019 and 2022 price CSVs.

  fig_dk1_timeseries.pdf     -- hourly prices over the year, both regimes
  fig_dk1_ecdf_spread.pdf    -- ECDF of within-day price spread (arbitrage proxy)

CHANGES vs plot_price_comparison.py (green-light version)
---------------------------------------------------------
1. Colours now use the thesis_style semantic slots instead of hardcoded hexes.
   Two-line comparisons (2019 vs 2022) use PALETTE["primary"]/["secondary"]
   = navy / dark red, as thesis_style prescribes. The old blue/orange are the
   fill slots reserved for the cycle-vs-calendar stacked bars, so the previous
   version collided with those figures. 2022 = primary (focus year),
   2019 = secondary.
2. Time-series x-axis label changed from "Month of year  (-)" to "Month".
   Months are calendar categories, not a dimensionless quantity, so no unit.
3. Time-series plotting order swapped so 2022 is drawn first (underneath) and
   2019 second (on top). This makes the legend order match the ECDF (2022, 2019)
   and keeps the thinner 2019 line visible where the two series overlap.

In-figure legends are kept (minimal style, consistent with the rest of the
thesis). Captions in the .tex refer to the series by year, not colour.

4. Ported to the standalone repository: inputs resolve through
   degradation.paths, figures are written beside this script, and the format
   is selected by OUTPUT below.
5. The runtime printout now covers every row of Table 4.1, so the table and
   the two figures come from one pass over the same CSVs.

Reproducible on Windows / VS Code: matplotlib + numpy + pandas, DejaVu font,
no LaTeX toolchain.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from degradation.paths import PRICE_CSV_2019, PRICE_CSV_2022, require
from degradation.style import apply_thesis_style, figsize, FS_ANNOT

# -- Output ------------------------------------------------------------------ #
OUTPUT = "png"     # "png", "pdf" or "both"
DPI = 300

P = apply_thesis_style(palette="brand", usetex=False)

# ===========================================================================
# CONFIGURATION
# ===========================================================================
HERE      = Path(__file__).resolve().parent
FILE_2019 = require(PRICE_CSV_2019)
FILE_2022 = require(PRICE_CSV_2022)

# Two-line comparison -> primary / secondary (navy / dark red) per thesis_style.
C_2022 = P["primary"]     # focus year (sizing sweep)
C_2019 = P["secondary"]   # low-volatility reference year

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_TICKS  = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016]


# ===========================================================================
# DATA HELPERS
# ===========================================================================
def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col, price_col = df.columns[0], df.columns[1]
    df = df.rename(columns={ts_col: "ts", price_col: "price"})
    df["ts"]   = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df         = df.dropna(subset=["ts"]).reset_index(drop=True)
    df["date"] = df["ts"].dt.date
    df["hour"] = np.arange(len(df), dtype=float)
    return df

def daily_spread(df: pd.DataFrame) -> pd.Series:
    return df.groupby("date")["price"].agg(lambda s: s.max() - s.min())

def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    v = np.sort(values.to_numpy())
    f = np.arange(1, v.size + 1) / v.size
    return v, f

def _save(fig, stem: str) -> None:
    """Write the formats OUTPUT asks for, beside this script."""
    if OUTPUT in ("pdf", "both"):
        fig.savefig(HERE / f"{stem}.pdf")
        print(f"  wrote {stem}.pdf")
    if OUTPUT in ("png", "both"):
        fig.savefig(HERE / f"{stem}.png", dpi=DPI)
        print(f"  wrote {stem}.png  ({DPI} dpi)")
    plt.close(fig)


# ===========================================================================
# FIG 1 -- HOURLY TIME SERIES (full text width)
# ===========================================================================
def plot_timeseries(d19: pd.DataFrame, d22: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.45))

    # 2022 first (underneath), 2019 second (on top, stays visible).
    ax.plot(d22["hour"], d22["price"],
            color=C_2022, lw=0.55, alpha=0.85, label="DK1 2022")
    ax.plot(d19["hour"], d19["price"],
            color=C_2019, lw=0.55, alpha=0.85, label="DK1 2019")

    ax.axhline(0, color=P["neutral"], lw=0.7, ls=(0, (4, 4)), alpha=0.6)
    ax.set_xticks(MONTH_TICKS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=FS_ANNOT)
    ax.set_xlim(0, 8760)
    ax.set_xlabel("Month")
    ax.set_ylabel("Day-ahead price  (EUR / MWh)")
    ax.legend(frameon=False, fontsize=FS_ANNOT, loc="upper left")

    _save(fig, "fig_dk1_timeseries")


# ===========================================================================
# FIG 2 -- ECDF OF DAILY PRICE SPREAD (about 0.6 text width)
# ===========================================================================
def plot_ecdf_spread(ds19: pd.Series, ds22: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=figsize(0.58, aspect=0.88))

    v22, f22 = ecdf(ds22)
    v19, f19 = ecdf(ds19)

    ax.plot(v22, f22, color=C_2022, lw=1.5, label="DK1 2022")
    ax.plot(v19, f19, color=C_2019, lw=1.5, label="DK1 2019")

    # Median markers with value annotation.
    for ds, col in [(ds22, C_2022), (ds19, C_2019)]:
        med = float(np.median(ds))
        ax.axvline(med, color=col, lw=0.8, ls=(0, (2, 3)), alpha=0.80)
        ax.text(med + (v22.max() * 0.015), 0.06,
                f"{med:.0f} EUR/MWh",
                color=col, fontsize=FS_ANNOT, va="bottom", rotation=90)

    ax.set_xlabel("Daily price spread  (EUR / MWh)")
    ax.set_ylabel("Cumulative fraction of days  (-)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=FS_ANNOT, loc="lower right")

    _save(fig, "fig_dk1_ecdf_spread")


# ===========================================================================
# MAIN
# ===========================================================================
def print_table_41(d19, d22, ds19, ds22) -> None:
    """Every row of Table 4.1, so the table and the figures share one pass."""
    rows = [
        ("Mean price (EUR/MWh)",         lambda p, s: p.mean()),
        ("Median price (EUR/MWh)",       lambda p, s: p.median()),
        ("Standard deviation (EUR/MWh)", lambda p, s: p.std()),
        ("Minimum (EUR/MWh)",            lambda p, s: p.min()),
        ("Maximum (EUR/MWh)",            lambda p, s: p.max()),
        ("5th percentile (EUR/MWh)",     lambda p, s: p.quantile(0.05)),
        ("95th percentile (EUR/MWh)",    lambda p, s: p.quantile(0.95)),
        ("Negative-price hours",         lambda p, s: float((p < 0).sum())),
        ("Mean daily spread (EUR/MWh)",  lambda p, s: s.mean()),
        ("Median daily spread (EUR/MWh)", lambda p, s: s.median()),
    ]
    print("\n-- Table 4.1 -- (copy into .tex)")
    print(f"  {'Metric':32}{'DK1 2019':>10}{'DK1 2022':>10}")
    for label, fn in rows:
        a = fn(d19["price"], ds19)
        b = fn(d22["price"], ds22)
        print(f"  {label:32}{a:10.1f}{b:10.1f}")


def main() -> None:
    if OUTPUT not in ("png", "pdf", "both"):
        raise ValueError(f'OUTPUT must be "png", "pdf" or "both", not {OUTPUT!r}')

    print("Loading price data...")
    d19 = load_prices(FILE_2019)
    d22 = load_prices(FILE_2022)
    ds19 = daily_spread(d19)
    ds22 = daily_spread(d22)

    for name, df in [("2019", d19), ("2022", d22)]:
        if len(df) != 8760:
            print(f"  WARNING: DK1 {name} has {len(df)} rows, expected 8760")

    print_table_41(d19, d22, ds19, ds22)

    print("")
    plot_timeseries(d19, d22)
    plot_ecdf_spread(ds19, ds22)


if __name__ == "__main__":
    main()