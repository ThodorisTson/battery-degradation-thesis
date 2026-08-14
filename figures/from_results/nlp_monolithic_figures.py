"""
nlp_monolithic_figures.py - figures for the monolithic NLP results.

  Figure 4.15   fig:nlp_scatter    optimality_vs_netgain
  Figures E.1-E.4  fig:monolithic_A..D   day_D049 / D179 / D084 / D363_lp_vs_nlp

Ported from plot_path3.py in the fork. The run script is scripts/run_nlp_monolithic.py and the summary table comes from
tools/aggregate_nlp_monolithic.py.

Two products:
  1. SCATTER   - net_gain vs final optimality (log x), converged vs not. Reads path3_summary_table.csv. The chapter's overview figure.
  2. DAY PANELS - per-day LP-vs-NLP dispatch (price on top, SoC below), for the A-E failure-taxonomy exemplars (or any days you pass). Reads the per-day *_results.npz + *_results.json from an all-days run.

Verified schema (path3.py np.savez_compressed + JSON summary):
  npz : lp_e, nlp_e  (energy state, length T+1 = 25), prices (T = 24)   [also *_p, *_curtailed, wind]
  json: summary.{e_cap, soc_min, soc_max, lp_net, nlp_net, revenue_sacrifice, degradation_saving, ratio, final_optimality, nlp_status}

CONVENTION NOTE: SoC is shown as a FRACTION (e / e_cap) with the [soc_min, soc_max] window, matching the weekly-dispatch figure. path3.py and compare_path3_v54.py plot
raw energy in MWh instead; pass --mwh to match those two files.

Usage (from anywhere in the repository):
  python figures/from_results/nlp_monolithic_figures.py --scatter --annotate-scatter
  python figures/from_results/nlp_monolithic_figures.py --scatter --all-days
  python figures/from_results/nlp_monolithic_figures.py --days     # the four exemplars
  python figures/from_results/nlp_monolithic_figures.py --days --mwh
  python figures/from_results/nlp_monolithic_figures.py --days D283 D264 D134
  python figures/from_results/nlp_monolithic_figures.py --inspect D283

With no --results it takes the newest run_* folder under
results/nlp_monolithic_all_days/. Figures are written beside this script.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from degradation.paths import RESULTS_DIR, require
from degradation.style import apply_thesis_style, figsize, TUDELFT, FS_ANNOT

# -- Output ------------------------------------------------------------------ #
# Figures E.1 to E.4 are included as PDF in the document, so "png" alone is not
# enough to rebuild the appendix.
OUTPUT = "png"     # "png", "pdf" or "both"
DPI = 300

HERE = Path(__file__).resolve().parent
NLP_RESULTS_DIR = RESULTS_DIR / "nlp_monolithic_all_days"
SUMMARY_CSV_NAME = "path3_summary_table.csv"


def latest_run(root: Path = NLP_RESULTS_DIR) -> Path:
    """Newest run_* folder. The timestamp in the name is what orders them."""
    runs = sorted(root.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"no run_* folder under {root}")
    return runs[-1]


def save(fig, stem: Path) -> None:
    """Write the formats OUTPUT asks for."""
    if OUTPUT in ("pdf", "both"):
        fig.savefig(stem.with_suffix(".pdf"))
    if OUTPUT in ("png", "both"):
        fig.savefig(stem.with_suffix(".png"), dpi=DPI)
    plt.close(fig)

# ---- default failure-taxonomy exemplars (one clean day per cell) -----------
# Four-cell taxonomy for the 550/175, RTE 0.910 run (run_20260706_010250). 
# B and C merged into one "converged loss" cell; the day shown is a B-type (worse on both axes). C-behavior (D217, trajectories coincide) is described
# in text only. Old five-cell set (D114/D349/D217/D097/D363) is retired: D114 and D097 moved to the diverged band under the corrected setup.
TAXONOMY = {
     49: "1 | converged gain | one low-value cycle removed (net +87, forgone 28, saved 115, ratio 4.1)",   # largest of class: D189 +136
    179: "2 | converged loss | worse on BOTH axes: degradation rises (net -45, saved -24, opt 2e-7)",       # C-behavior alt: D217
     84: "3 | stalled | step collapse; gain the diagnostics dissolve (net +107, opt 3.9e-3)",               # alt D312 (+286, reads like diverged)
    363: "4 | diverged | flat NLP, fictional saving (net +986, opt 8.6e9)",                                 # alt D099 (+958)
}

# Per-exemplar label offsets (points) for the annotated scatter. Default is (4, 4), up-right. D179 sits inside the dense converged cluster, so it is
# pushed right and down into the gap. Tune here if a label overlaps a point.
SCATTER_LABEL_OFFSETS = {49: (4, 4), 84: (4, 4), 179: (20, -5), 363: (4, 4)}

# Days on which the price spread was too small to support a charge cycle, so the battery stayed idle. Excluded from the 356 active days reported in
# Section 4.5. Plotted only with --all-days.
IDLE_DAYS = {"D024", "D027", "D045", "D094", "D098",
             "D279", "D315", "D320", "D321"}

# What Section 4.5.1 states about the committed run. Checked rather than
# assumed, so a regenerated or partial summary table is visible in the console
# instead of silently redrawing Figure 4.15 with different counts than the text.
EXPECTED = dict(total=365, active=356, converged=185, not_converged=171)

def term_reason(msg: str) -> str:
    m = (msg or "").lower()
    if "gtol" in m:     return "gtol (1st-order optimum)"
    if "xtol" in m:     return "xtol (step collapse)"
    if "maxim" in m:    return "maxiter"
    if "callback" in m: return "callback stop"
    return (msg or "?")[:40]

def find_day_files(results_dir: Path, day: int):
    for pat in (f"*_D{day:03d}_*_results.json", f"*_D{day}_*_results.json"):
        hits = sorted(results_dir.glob(pat))
        if hits:
            jp = hits[-1]
            npz = jp.with_name(jp.name.replace("_results.json", "_results.npz"))
            return jp, (npz if npz.exists() else None)
    return None, None

def load_day(jp: Path, npz: Path) -> dict:
    s = json.load(open(jp, encoding="utf-8"))["summary"]
    z = np.load(npz, allow_pickle=True)
    need = ("prices", "lp_e", "nlp_e")                 # only what the panel plots
    missing = [k for k in need if k not in z.files]
    if missing:
        raise KeyError(f"{npz.name} missing {missing}; has {list(z.files)}")
    return dict(
        prices=np.asarray(z["prices"], float),
        lp_e=np.asarray(z["lp_e"], float),             # raw MWh, length T+1
        nlp_e=np.asarray(z["nlp_e"], float),
        e_cap=float(s.get("e_cap", 1.0)) or 1.0,
        soc_min=float(s.get("soc_min", 0.1)), soc_max=float(s.get("soc_max", 0.9)),
        net=float(s.get("nlp_net", 0)) - float(s.get("lp_net", 0)),
        sac=float(s.get("revenue_sacrifice", 0)), save=float(s.get("degradation_saving", 0)),
        ratio=float(s.get("ratio", 0)), opt=float(s.get("final_optimality", np.nan)),
        term=term_reason(s.get("nlp_status", "")),
    )

def plot_day(day: int, results_dir: Path, out: Path, label=None, annotate=True, mwh=False, clean=False):
    jp, npz = find_day_files(results_dir, day)
    if jp is None or npz is None:
        print(f"  [D{day}] results not found in {results_dir} - skipped"); return
    P = apply_thesis_style(palette="brand", usetex=False)
    navy, darkred = TUDELFT["navy"], TUDELFT["darkred"]
    d = load_day(jp, npz)
    n = len(d["prices"])                                # T hourly prices
    ec = d["e_cap"] if not mwh else 1.0                 # divide by e_cap unless --mwh
    lp_y, nlp_y = d["lp_e"] / ec, d["nlp_e"] / ec
    lo_line, hi_line = d["soc_min"] * (1 if not mwh else d["e_cap"]), \
                       d["soc_max"] * (1 if not mwh else d["e_cap"])
    t_soc = np.arange(len(lp_y))                        # energy state is T+1 (boundary states)
    pr_x = np.arange(n + 1); pr_y = np.append(d["prices"], d["prices"][-1])

    # drawn at 0.85 textwidth (the appendix include width), so on-page text is
    # the nominal thesis 9/8/7 pt.
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=figsize(0.85, aspect=0.72),
                                   sharex=True, gridspec_kw={"height_ratios": [1.0, 1.6]})
    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)
    ax1.set_xlim(0, n)

    # price (top)
    ax0.fill_between(pr_x, pr_y, step="post", color=P["shade"], alpha=0.9)
    ax0.step(pr_x, pr_y, where="post", color=P["neutral"], lw=1.0)
    ax0.set_ylabel("price\n(EUR/MWh)")

    # SoC (bottom): LP navy solid, NLP dark red dashed
    ax1.plot(t_soc, lp_y,  color=navy,    lw=1.6, label="LP")
    ax1.plot(t_soc, nlp_y, color=darkred, lw=1.5, ls=(0, (5, 2)), label="NLP")
    ax1.axhline(lo_line, color=P["grid"], ls=":", lw=0.8)
    ax1.axhline(hi_line, color=P["grid"], ls=":", lw=0.8)
    if not mwh:
        ax1.set_ylim(0, 1); ax1.set_ylabel("SoC fraction")
    else:
        ax1.set_ylabel("stored energy (MWh)")
    ax1.set_xlabel("hour of day")
    ax1.legend(frameon=False, loc="upper right", ncol=2)

    if annotate and not clean:
        txt = (f"net {d['net']:+.1f} EUR   forgone {d['sac']:.1f}   "
               f"saved {d['save']:.1f}   opt {d['opt']:.1e}")
        ax1.text(0.015, 0.04, txt, transform=ax1.transAxes, fontsize=FS_ANNOT,
                 va="bottom", ha="left",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=P["grid"], alpha=0.9))
        if label:
            ax0.text(0.015, 0.92, label, transform=ax0.transAxes, fontsize=FS_ANNOT,
                     va="top", ha="left", color=P["neutral"])

    fig.align_ylabels([ax0, ax1])
    stem = out / f"day_D{day:03d}_lp_vs_nlp"
    save(fig, stem)
    if clean:
        # echo the values for the caption, since they are no longer drawn on the figure
        print(f"  wrote {stem}.png   [caption data] net {d['net']:+.1f} EUR | "
              f"forgone {d['sac']:.1f} | saved {d['save']:.1f} | ratio {d['ratio']:.2f} | "
              f"opt {d['opt']:.1e} | {d['term']}")
    else:
        print(f"  wrote {stem.name} ({OUTPUT})")

def plot_scatter(csv: Path, out: Path, annotate_days=None, net_clip: float = 150.0, active_only: bool = True):
    import csv as _csv
    P = apply_thesis_style(palette="brand", usetex=False)
    all_rows = list(_csv.DictReader(open(csv, encoding="utf-8")))
    rows = [r for r in all_rows if r["case"] not in IDLE_DAYS] if active_only else all_rows
    opt = np.array([float(r["optimality"]) for r in rows])
    net = np.array([float(r["net_gain"]) for r in rows])
    conv = opt < 1e-6
    lo, hi = -net_clip, net_clip                        # objective change clipped for display
    n_hi, n_lo = int((net > hi).sum()), int((net < lo).sum())
    print(f"  days plotted: {len(rows)} "
          f"({'active only' if active_only else 'all days, idle included'}); "
          f"converged {int(conv.sum())}, not converged {int((~conv).sum())}")
    print(f"  clipped at +/-{net_clip:g} EUR: {n_hi} above, {n_lo} below "
          f"(range {net.min():+.1f} to {net.max():+.1f})")
    if active_only:
        got = dict(total=len(all_rows), active=len(rows),
                   converged=int(conv.sum()), not_converged=int((~conv).sum()))
        bad = {k: (got[k], v) for k, v in EXPECTED.items() if got[k] != v}
        if bad:
            print("  WARNING: counts differ from Section 4.5.1 of the thesis")
            for k, (g, e) in bad.items():
                print(f"    {k}: got {g}, thesis states {e}")
    navy, darkred = TUDELFT["navy"], TUDELFT["darkred"]

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.52))
    # not converged first (behind), converged on top
    ax.scatter(opt[~conv], np.clip(net[~conv], lo, hi), s=15, c=darkred, alpha=0.55,
               edgecolors="none",
               label=fr"not converged ($\geq 10^{{-6}}$, n={int((~conv).sum())})")
    ax.scatter(opt[conv], np.clip(net[conv], lo, hi), s=15, c=navy, alpha=0.85,
               edgecolors="none",
               label=fr"converged ($< 10^{{-6}}$, n={int(conv.sum())})")
    ax.set_xscale("log")
    ax.axhline(0, color=P["neutral"], lw=0.7)
    ax.axvline(1e-6, color=P["neutral"], ls="--", lw=0.8)     # convergence line
    ax.axvline(1e-2, color=P["grid"],    ls=":",  lw=0.9)     # divergence line
    ax.set_xlabel("final optimality (log scale)")
    ax.set_ylabel("Objective difference, NLP $-$ LP  (EUR/day)")
    ax.set_ylim(max(lo, net.min() * 1.08) - 4, hi + 4)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    if annotate_days:
        by = {r["case"]: r for r in rows}
        for day in annotate_days:
            r = by.get(f"D{day:03d}") or by.get(f"D{day}")
            if r:
                ax.annotate(f"D{day:03d}",
                            (float(r["optimality"]), np.clip(float(r["net_gain"]), lo, hi)),
                            fontsize=FS_ANNOT, color=P["neutral"],
                            xytext=SCATTER_LABEL_OFFSETS.get(day, (4, 4)),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec="none", alpha=0.75))
    stem = out / "optimality_vs_netgain"
    save(fig, stem)
    print(f"  wrote {stem.name} ({OUTPUT})")

def inspect(day: int, results_dir: Path):
    jp, npz = find_day_files(results_dir, day)
    if jp is None:
        print(f"no results.json for D{day} in {results_dir}"); return
    s = json.load(open(jp, encoding="utf-8")).get("summary", {})
    print(f"json: {jp.name}\n  summary keys:", sorted(s.keys()))
    if npz:
        z = np.load(npz, allow_pickle=True)
        print(f"npz : {npz.name}\n  arrays:", {k: np.shape(z[k]) for k in z.files})
    else:
        print("npz : NOT FOUND next to the json")

def main():
    if OUTPUT not in ("png", "pdf", "both"):
        raise ValueError(f'OUTPUT must be "png", "pdf" or "both", not {OUTPUT!r}')
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=str(HERE),
                    help="output folder (default: beside this script)")
    ap.add_argument("--scatter", action="store_true")
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--inspect", default=None)
    ap.add_argument("--no-annot", action="store_true")
    ap.add_argument("--clean", action="store_true",
                    help="panels only (no annotation box, no cell label); echo values to console for captions")
    ap.add_argument("--mwh", action="store_true", help="plot raw MWh (match path3/compare) instead of SoC fraction")
    ap.add_argument("--annotate-scatter", action="store_true")
    ap.add_argument("--net-clip", type=float, default=150.0,
                    help="clip the objective change to +/- this value on the scatter (default 150)")
    ap.add_argument("--all-days", action="store_true",
                    help="include the nine idle days on the scatter (default: 356 active days only)")
    args = ap.parse_args()

    results_dir = require(Path(args.results)) if args.results else latest_run()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"  run folder : {results_dir}")
    to_int = lambda s: int(re.sub(r"\D", "", str(s)))

    if args.inspect:
        inspect(to_int(args.inspect), results_dir); return
    if args.scatter:
        csv = Path(args.csv) if args.csv else (results_dir / SUMMARY_CSV_NAME)
        if not csv.exists():
            print(f"  scatter skipped: {csv} not found (pass --csv)")
        else:
            plot_scatter(csv, out, annotate_days=list(TAXONOMY) if args.annotate_scatter else None,
                         net_clip=args.net_clip, active_only=not args.all_days)
    if args.days is not None:
        days = [to_int(x) for x in args.days] if args.days else list(TAXONOMY)
        print(f"  results dir: {results_dir}  ({'MWh' if args.mwh else 'SoC fraction'})")
        for dnum in days:
            plot_day(dnum, results_dir, out, label=TAXONOMY.get(dnum),
                     annotate=not args.no_annot, mwh=args.mwh, clean=args.clean)
    if not (args.scatter or args.days is not None or args.inspect):
        ap.print_help()

if __name__ == "__main__":
    main()