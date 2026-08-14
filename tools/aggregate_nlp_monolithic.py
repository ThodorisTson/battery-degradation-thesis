"""
aggregate_nlp_monolithic.py - collate monolithic NLP daily-run JSONs into one
results table.

Globs a run folder, keeps the LATEST run per (case, e_cap, p_cap, wind_mode),
and writes a console table + CSV + a LaTeX table (thesis style: \\hline +
\\noalign{\\smallskip}, open sides, no booktabs).

The CSV it writes is the input to figures/from_results/nlp_monolithic_figures.py,
which draws Figure 4.15 from it. The LaTeX table is the source for Table E.3.
Ported from aggregate_path3.py in the fork; the run script is
scripts/run_nlp_monolithic.py, formerly path3.py.

This lives in tools/ rather than scripts/ because scripts/ holds the four
simulation entry points that take hours to run, and rather than results/ because
that directory is frozen output that a run can overwrite.

REGIME (Option A): each day is classified by its own DAILY price spread, using
per-year relative tertiles (33rd / 67th percentile of that year's daily spreads).
Cuts are computed over the FULL year's price file (stable regardless of which
subset of days is in the results folder). The computed regime takes precedence
over the stale period-string label; if no price file is found for a run's year,
the old parse_regime(period) behavior is used as a fallback so nothing breaks.

Spread metric defaults to max-min (matches the weekly-dispatch figure); P95-P5 is
available via --spread-metric and agrees with max-min on ~88% of 2022 days.

Note: regime is a WITHIN-YEAR rank. 2019 cuts ~19/30 EUR/MWh, 2022 cuts ~127/210;
a "High" 2019 day is calm by 2022 standards. Do not read it as an absolute margin.

Usage (from anywhere in the repository):
    python tools/aggregate_nlp_monolithic.py
    python tools/aggregate_nlp_monolithic.py --dir results/nlp_monolithic_all_days/run_...
    python tools/aggregate_nlp_monolithic.py --spread-metric p95p5
    python tools/aggregate_nlp_monolithic.py --all   # include battery-only runs too

With no --dir it takes the newest run_* folder under
results/nlp_monolithic_all_days/.
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
import numpy as np

from degradation.paths import DATA_DIR, RESULTS_DIR, require

NLP_RESULTS_DIR = RESULTS_DIR / "nlp_monolithic_all_days"


def latest_run(root: Path = NLP_RESULTS_DIR) -> Path:
    """Newest run_* folder. The timestamp in the name is what orders them."""
    runs = sorted(root.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"no run_* folder under {root}")
    return runs[-1]

REGIME_ORDER = {"High": 0, "Mid": 1, "Low": 2, "?": 3}

def parse_case(fname: str) -> str:
    # ..._dk2022_D166_trust-constr_scaled_results.json  -> D166
    m = re.search(r"_dk\d{4}_([A-Za-z0-9]+)_", fname)
    return m.group(1) if m else "?"

def parse_year(fname: str):
    # ..._dk2022_...  -> 2022
    m = re.search(r"_dk(\d{4})_", fname)
    return int(m.group(1)) if m else None

def parse_ts(fname: str) -> str:
    m = re.match(r"(\d{8}_\d{6})_", fname)
    return m.group(1) if m else "0"

def parse_regime(period: str) -> str:
    # legacy fallback only: extract High/Mid/Low from the period label string
    m = re.search(r"\b(High|Mid|Low)\b", period)
    return m.group(1) if m else "?"

# ----------------------------------------------------------------------------
# Option A: per-year daily-spread tertile classifier
# ----------------------------------------------------------------------------
def daily_spread_classifier(price_csv: Path, metric: str = "maxmin"):
    """Return (regime_of_day, (q1, q2), spread_of_day, n_days) for one year.

    Tertile cuts are computed over ALL days in the file, so a day's regime does
    not depend on which other days happen to be in the results folder.
    """
    rows = list(csv.DictReader(open(price_csv, encoding="utf-8")))
    key = "price_eur_mwh" if rows and "price_eur_mwh" in rows[0] else list(rows[0])[-1]
    a = np.array([float(r[key]) for r in rows], dtype=float)
    nd = a.size // 24
    a = a[:nd * 24].reshape(nd, 24)
    if metric == "p95p5":
        spread = np.percentile(a, 95, 1) - np.percentile(a, 5, 1)
    else:  # "maxmin"
        spread = a.max(1) - a.min(1)
    q1, q2 = (float(x) for x in np.quantile(spread, [1 / 3, 2 / 3]))

    def regime_of_day(d: int) -> str:           # d is the 1-indexed day number
        i = d - 1
        if not (0 <= i < nd):
            return "?"
        s = spread[i]
        return "Low" if s <= q1 else ("Mid" if s <= q2 else "High")

    def spread_of_day(d: int):
        i = d - 1
        return float(spread[i]) if 0 <= i < nd else float("nan")

    return regime_of_day, (q1, q2), spread_of_day, nd


def get_classifier(year, prices_dir: Path, metric: str, cache: dict, reported: set):
    """Lazily build/cache a classifier for `year`; None if its price file is absent."""
    if year in cache:
        return cache[year]
    pcsv = prices_dir / f"dk1_prices_{year}.csv"
    if year is None or not pcsv.exists():
        if year not in reported:
            print(f"  [regime] no price file {pcsv} - falling back to period label "
                  f"for {year} runs")
            reported.add(year)
        cache[year] = None
        return None
    clf = daily_spread_classifier(pcsv, metric)
    print(f"  [regime] {year}: daily {metric} tertile cuts = "
          f"{clf[1][0]:.0f} / {clf[1][1]:.0f} EUR/MWh  ({clf[3]} days)")
    cache[year] = clf
    return clf
# ----------------------------------------------------------------------------

def collect(results_dir: Path, want_wind: bool, e_cap, p_cap, include_all: bool,
            prices_dir: Path, metric: str):
    rows = {}
    clf_cache, reported = {}, set()
    for jp in results_dir.glob("*_results.json"):
        try:
            s = json.load(open(jp, encoding="utf-8"))["summary"]
        except Exception:
            continue
        if not include_all:
            if bool(s.get("wind_mode", False)) != want_wind:
                continue
            if e_cap is not None and abs(float(s["e_cap"]) - e_cap) > 1e-6:
                continue
            if p_cap is not None and abs(float(s["p_cap"]) - p_cap) > 1e-6:
                continue
        case = parse_case(jp.name)
        year = parse_year(jp.name)
        key = (case, s["e_cap"], s["p_cap"], bool(s.get("wind_mode", False)))
        ts = parse_ts(jp.name)
        if key not in rows or ts > rows[key]["_ts"]:
            # --- Option A regime: classify by this day's own spread (per-year tertiles)
            regime, spread = parse_regime(s.get("period", "")), float("nan")
            dm = re.fullmatch(r"D(\d+)", case)
            clf = get_classifier(year, prices_dir, metric, clf_cache, reported)
            if dm and clf is not None:
                d = int(dm.group(1))
                regime = clf[0](d)        # computed regime takes precedence
                spread = clf[2](d)
            lp_rev = float(s["lp_revenue"]); lp_deg = float(s["lp_deg_cost"])
            rows[key] = dict(
                case=case,
                regime=regime,
                daily_spread=spread,
                deg_rev_pct=100.0 * lp_deg / lp_rev if lp_rev else float("nan"),
                sacrifice=float(s["revenue_sacrifice"]),
                saving=float(s["degradation_saving"]),
                ratio=float(s["ratio"]),
                net_gain=float(s["nlp_net"]) - float(s["lp_net"]),
                iters=int(s["nlp_iters"]),
                optimality=float(s["final_optimality"]),
                converged=bool(s.get("converged", False)),
                wind=bool(s.get("wind_mode", False)),
                _ts=ts,
            )
    out = list(rows.values())
    out.sort(key=lambda r: (REGIME_ORDER.get(r["regime"], 9), r["case"]))
    return out

def print_console(rows):
    h = (f'{"Case":<6}{"Reg":<5}{"Spread":>8}{"deg/rev":>8}{"Sacr":>9}{"Save":>9}'
         f'{"Ratio":>7}{"NetGain":>9}{"Iters":>7}{"Opt":>11}')
    print(h); print("-" * len(h))
    for r in rows:
        sp = r.get("daily_spread", float("nan"))
        sp_s = f'{sp:>8.0f}' if sp == sp else f'{"-":>8}'   # nan-safe
        print(f'{r["case"]:<6}{r["regime"]:<5}{sp_s}{r["deg_rev_pct"]:>7.2f}%'
              f'{r["sacrifice"]:>9.2f}{r["saving"]:>9.2f}{r["ratio"]:>7.2f}'
              f'{r["net_gain"]:>9.2f}{r["iters"]:>7d}{r["optimality"]:>11.2e}')

def write_csv(rows, path: Path):
    cols = ["case","regime","daily_spread","deg_rev_pct","sacrifice","saving","ratio",
            "net_gain","iters","optimality","converged","wind"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)

def write_latex(rows, path: Path):
    # thesis style: \hline + \noalign{\smallskip}, open sides, no booktabs
    lines = [r"\begin{tabular}{l l r r r r r r}",
             r"\hline\noalign{\smallskip}",
             r"Case & Regime & deg/rev & Sacrifice & Saving & Ratio & Net gain & Optimality \\",
             r"\noalign{\smallskip}\hline\noalign{\smallskip}"]
    for r in rows:
        lines.append(
            f'{r["case"]} & {r["regime"]} & {r["deg_rev_pct"]:.2f}\\% & '
            f'{r["sacrifice"]:.2f} & {r["saving"]:.2f} & {r["ratio"]:.2f} & '
            f'{r["net_gain"]:.2f} & {r["optimality"]:.2e} \\\\')
    lines += [r"\noalign{\smallskip}\hline", r"\end{tabular}"]
    path.write_text("\n".join(lines), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None,
                    help="run folder (default: newest run_* under "
                         "results/nlp_monolithic_all_days)")
    ap.add_argument("--wind", action="store_true", default=True)
    ap.add_argument("--battery-only", dest="wind", action="store_false")
    ap.add_argument("--all", action="store_true", help="include every run, no filter")
    ap.add_argument("--e-cap", type=float, default=550.0)
    ap.add_argument("--p-cap", type=float, default=175.0)
    ap.add_argument("--prices-dir", default=None,
                    help="folder with dk1_prices_<year>.csv (default: data/)")
    ap.add_argument("--spread-metric", choices=["maxmin", "p95p5"], default="maxmin")
    args = ap.parse_args()

    d = require(Path(args.dir)) if args.dir else latest_run()
    prices_dir = Path(args.prices_dir) if args.prices_dir else DATA_DIR
    print(f"  run folder : {d}")
    print(f"  prices dir : {prices_dir}")
    rows = collect(d, args.wind, args.e_cap, args.p_cap, args.all,
                   prices_dir, args.spread_metric)
    if not rows:
        print(f"No matching JSONs in {d.resolve()}"); return
    print_console(rows)
    write_csv(rows, d / "path3_summary_table.csv")
    write_latex(rows, d / "path3_summary_table.tex")
    print(f"\nWrote {d/'path3_summary_table.csv'} and .tex  ({len(rows)} cases)")

if __name__ == "__main__":
    main()