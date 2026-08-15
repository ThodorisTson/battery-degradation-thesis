r"""
plot_sweep_thesis.py
====================
NPV landscapes and slices from the two-dimensional sizing sweep.

Produces Figures 4.9 to 4.12:

    Figure 4.9   fig_sweep_full_no_deg   coarse grid, no-degradation scenario
                 fig_sweep_full_xu       coarse grid, Xu degradation scenario
    Figure 4.10  fig_sweep_zoom_xu       refined grid, Xu scenario
                 fig_sweep_zoom_shi      refined grid, Shi scenario
    Figure 4.11  fig_sweep_npv_vs_E_...  NPV against E at fixed P, pooled
    Figure 4.12  fig_sweep_npv_vs_P_...  NPV against P at fixed E, pooled

Figure 4.13, the revenue and cost decomposition, comes from a separate script.

SIZING
------
The four heatmap panels are placed two to a row, so each occupies about half
the text width, and the two slice figures are placed one per row at full text
width. Each figure is drawn at the width it is included at. Following the rule
stated in style.py, a figure drawn at its include width receives no scaling
from LaTeX, so the 8 pt ticks and 9 pt axis labels set in style.py are also the
sizes that appear on the page. Drawing wider than the include width and letting
LaTeX shrink the result is what makes the text too small to read.

Set HEATMAP_FRAC and SLICE_FRAC to whatever the \includegraphics widths in the
.tex actually are. If they disagree, the on-page text size will not be 8 and
9 pt.

POOLING
-------
The two slice figures are drawn from the coarse and refined sweeps combined.
The two runs are separate evaluations of the same deterministic model, so a
design point present in both must return the same NPV. That is checked before
pooling; if it fails, the two files are not comparable and a pooled line has no
meaning. Both slices come from one shared plotting function, so line width,
marker size, font sizes, figure size and legend style cannot drift apart.

INPUTS
------
Discovered in results/sizing_sweep/. Every file matching CSV_GLOB carries a run
timestamp in its name, in the form YYYYMMDD_HHMMSS. The two most recent runs are
selected, then classified as the coarse and the refined sweep by measuring the
extent of the E and P grid each one covers. Classification is by measured extent
rather than by timestamp order, so a rerun of either sweep alone cannot silently
swap the two roles. Set COARSE_TAG and ZOOM_TAG to pin specific runs.

Outputs are written beside this script, in figures/from_results/.

Values printed at runtime for the captions:
    the two files selected, with their run timestamps
    grid optimum (E*, P*, NPV*) for each scenario in each run
    quadratic-fit optimum with R-squared, refined grid only
    pooled colour range shared by the two refined panels
    overlap and pooled point counts, and the point count on each slice
    Xu optimum along each slice
    canvas and plot box in inches for every figure

Note on Figure 4.9: the two panels do not share a colour scale. The
no-degradation NPV range and the Xu NPV range do not overlap, so a shared scale
would put both panels in a single band. The caption must state that the colour
scales differ, otherwise a reader comparing colours across the two panels draws
the wrong conclusion.

Note on Figure 4.12: E = 550 MWh is not a point on the coarse grid, so pooling
adds nothing to that slice and its P range is set by the refined sweep alone.
The script reports this at runtime. State it in the caption.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from degradation.style import (apply_thesis_style, figsize, TUDELFT,
                               TEXTWIDTH_IN, FS_BASE, FS_LABEL,
                               FS_LEGEND, FS_ANNOT)
from degradation.paths import RESULTS_DIR, require

PALETTE = apply_thesis_style(palette="brand", usetex=False)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
OUTPUT = "png"         # "png", "pdf" or "both"
DPI    = 300

# Fraction of \textwidth each figure is included at.
HEATMAP_FRAC = 0.48    # Figures 4.9 and 4.10, two panels per row
SLICE_FRAC   = 1.00    # Figures 4.11 and 4.12, one per row

# Canvas height over width. The coarse grid spans 1000 MWh and the refined grid
# 300 MWh, so the coarse panel is drawn taller.
ASPECT_FULL  = 0.80
ASPECT_ZOOM  = 0.72
ASPECT_SLICE = 0.50

# Slice positions. Both are the Stage 1 optimum of the Xu scenario.
FIXED_P_MW  = 175      # Figure 4.11, NPV against E at this P
FIXED_E_MWH = 550      # Figure 4.12, NPV against P at this E

# Output stems for the slice figures. These match the filenames the .tex
# currently loads. The suffix records settings that are now the only behaviour,
# so it is a candidate for the filename cleanup pass, together with the .tex.
STEM_SLICE_E = "fig_sweep_npv_vs_E_p{fix:.0f}_pooled_noshade"
STEM_SLICE_P = "fig_sweep_npv_vs_P_e{fix:.0f}"

# Shaded band between the no-degradation and Xu curves on the slice figures.
SHADE_DEG_COST = False

# Tick steps on the heatmaps, set explicitly. An automatic locator picks round
# numbers that are not sweep points, which invites a reader to assume a
# configuration was run when it was not.
TICK_P_FULL, TICK_E_FULL = 50, 200    # MW, MWh
TICK_P_ZOOM, TICK_E_ZOOM = 50, 50     # MW, MWh

# Marker areas, in points squared, at the heatmap panel width above.
S_STAR, S_DIAMOND, S_CIRCLE = 170, 46, 56
S_DOTS_FULL, S_DOTS_ZOOM = 3, 4

SWEEP_DIR = RESULTS_DIR / "sizing_sweep"
CSV_GLOB  = "*lifetime_sweep*.csv"

# All six figures are written here regardless of where this file sits, so a
# copy run from another folder cannot scatter outputs.
OUT_DIR = RESULTS_DIR.parent / "figures" / "from_results"

# Run selection. Leave both as None to discover the two most recent runs.
# Set either to a timestamp substring, for example "20260703_221934", to pin
# that run and bypass discovery.
COARSE_TAG = None
ZOOM_TAG   = None

# Timestamp embedded in every run filename: YYYYMMDD_HHMMSS
TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")

SCENARIO = "xu"        # degraded model shown in the primary panels

# Scenarios drawn on the coarse grid. Figure 4.9 uses these two only. Adding
# "shi" here writes a fig_sweep_full_shi that the thesis does not include.
FULL_SCENARIOS = ("xu", "no_deg")

EUR_TO_MEUR = 1e-6

# White-space padding on the refined panels. The no-degradation diamond sits on
# the grid corner closest to the axes; without padding the frame clips it.
ZOOM_PAD_P = 8       # MW
ZOOM_PAD_E = 14      # MWh

# Expected extent of the refined grid, checked against the data at runtime.
# A mismatch is reported but does not stop the run: the figures always use the
# measured extent, so the dashed box and the fit domain cannot drift from the
# data they describe.
EXPECTED_ZOOM_E = (450.0, 750.0)    # MWh
EXPECTED_ZOOM_P = (125.0, 250.0)    # MW

# ── Colour assignments ──────────────────────────────────────────────────────
C_NODEG = TUDELFT["navy"]
C_XU    = TUDELFT["darkred"]
C_SHI   = TUDELFT["blue"]
C_SCENARIO = {"no_deg": C_NODEG, "xu": C_XU, "shi": C_SHI}

CMAP_NPV = "RdYlGn"   # low NPV red, high NPV green

SCENARIO_LABEL = {
    "no_deg": "No degradation",
    "xu":     "Xu et al. (2016)",
    "shi":    "Shi et al. (2018)",
}
NPV_COL = {
    "no_deg": "npv_no_deg",
    "xu":     "npv_with_xu",
    "shi":    "npv_with_shi",
}


# ═══════════════════════════════════════════════════════════════════════════
# RUN DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

def _run_timestamp(path: Path) -> datetime:
    """Parse the YYYYMMDD_HHMMSS stamp out of a run filename."""
    m = TIMESTAMP_RE.search(path.name)
    if m is None:
        raise ValueError(f"No YYYYMMDD_HHMMSS timestamp in {path.name}")
    return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")


def _list_runs(sweep_dir: Path) -> list[tuple[datetime, Path]]:
    """Every sweep CSV in `sweep_dir`, newest first, by embedded timestamp.

    File modification time is deliberately not used. Git sets it to the clone
    time in arbitrary order, so an ordering based on it would differ between
    machines. The timestamp in the filename is written by the run itself.
    """
    runs = []
    for p in sweep_dir.glob(CSV_GLOB):
        try:
            runs.append((_run_timestamp(p), p))
        except ValueError:
            print(f"  skipped, no timestamp in name: {p.name}")
    runs.sort(key=lambda t: t[0], reverse=True)
    return runs


def _pin(runs: list[tuple[datetime, Path]], tag: str) -> tuple[datetime, Path]:
    """Select the one run whose filename contains `tag`."""
    hits = [r for r in runs if tag in r[1].name]
    if len(hits) == 1:
        return hits[0]
    listing = "\n    ".join(p.name for _, p in runs) or "(none)"
    if not hits:
        raise FileNotFoundError(
            f"No sweep CSV matching '{tag}'.\n  Runs found:\n    {listing}")
    raise ValueError(
        f"'{tag}' matches {len(hits)} runs. Use a longer tag.\n"
        f"  Matches:\n    " + "\n    ".join(p.name for _, p in hits))


def _extent(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Bounding box of the sampled grid: (E_min, E_max, P_min, P_max)."""
    return (float(df.e_cap.min()), float(df.e_cap.max()),
            float(df.p_cap.min()), float(df.p_cap.max()))


def _area(df: pd.DataFrame) -> float:
    e_lo, e_hi, p_lo, p_hi = _extent(df)
    return (e_hi - e_lo) * (p_hi - p_lo)


def _read(path: Path) -> pd.DataFrame:
    """Read one sweep CSV and convert every NPV column from EUR to MEUR."""
    df = pd.read_csv(path)
    for col in NPV_COL.values():
        if col in df.columns:
            df[col] = df[col] * EUR_TO_MEUR
    return df


def select_runs(sweep_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (coarse, refined) sweep frames.

    With both tags unset, the two most recent runs are taken and the one
    covering the larger area of the E-P plane is the coarse sweep.
    """
    runs = _list_runs(sweep_dir)
    if not runs:
        raise FileNotFoundError(f"No file matching '{CSV_GLOB}' in {sweep_dir}")

    if COARSE_TAG is not None and ZOOM_TAG is not None:
        picked = [_pin(runs, COARSE_TAG), _pin(runs, ZOOM_TAG)]
        classify = False
    else:
        if len(runs) < 2:
            raise FileNotFoundError(
                f"Need two sweep runs in {sweep_dir}, found {len(runs)}: "
                f"{runs[0][1].name}")
        picked = runs[:2]
        classify = True
        if len(runs) > 2:
            print(f"  {len(runs)} runs present, taking the two most recent. "
                  f"Set COARSE_TAG and ZOOM_TAG to pin specific runs.")

    frames = [(ts, p, _read(p)) for ts, p in picked]
    for ts, p, df in frames:
        print(f"  {ts:%Y-%m-%d %H:%M:%S}  {p.name}  ({len(df)} rows)")

    if classify:
        frames.sort(key=lambda f: _area(f[2]), reverse=True)
    (ts_c, path_c, df_coarse), (ts_z, path_z, df_zoom) = frames

    print(f"  coarse  : {path_c.name}")
    print(f"  refined : {path_z.name}")
    if ts_z < ts_c:
        print("  note: the refined run predates the coarse run, which is the "
              "reverse of the usual order. Confirm the two files are the "
              "intended pair.")
    return df_coarse, df_zoom


# ═══════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def grid_optimum(df: pd.DataFrame, scenario: str) -> tuple[float, float, float]:
    """Return (E, P, NPV in MEUR) at the grid optimum for `scenario`."""
    col = NPV_COL[scenario]
    v = df.dropna(subset=[col])
    row = v.loc[v[col].idxmax()]
    return float(row["e_cap"]), float(row["p_cap"]), float(row[col])


def pool_sweeps(df_coarse: pd.DataFrame, df_zoom: pd.DataFrame,
                rtol: float = 1e-9) -> pd.DataFrame:
    """Merge the coarse and refined sweeps into one sample set.

    The two runs are separate evaluations of the same deterministic model, so a
    design point evaluated in both must return the same NPV. That is checked
    here before pooling; if it fails, the two files are not comparable and a
    pooled line has no meaning.

    Returns one row per unique (e_cap, p_cap), sorted, with a `src` column
    recording which sweep supplied the row.
    """
    key = ["e_cap", "p_cap"]
    both = df_coarse.merge(df_zoom, on=key, suffixes=("_c", "_z"))
    for c in NPV_COL.values():
        a, b = both[c + "_c"].values, both[c + "_z"].values
        ok = np.isclose(a, b, rtol=rtol, equal_nan=True)
        if not ok.all():
            worst = np.nanmax(np.abs(a - b))
            raise ValueError(
                f"Sweeps disagree on shared points for '{c}': "
                f"{(~ok).sum()} of {len(ok)} points differ, "
                f"max |diff| = {worst:.6g} MEUR. Pooling is only valid if the "
                "shared points are identical.")
    print(f"  overlap check: {len(both)} shared (E,P) points, identical in all "
          f"{len(NPV_COL)} NPV columns")

    # The refined pass takes precedence on shared points.
    c = df_coarse.assign(src="coarse")
    z = df_zoom.assign(src="refined")
    pooled = (pd.concat([z, c], ignore_index=True)
                .drop_duplicates(subset=key, keep="first")
                .sort_values(key)
                .reset_index(drop=True))
    print(f"  pooled: {len(df_coarse)} + {len(df_zoom)} - {len(both)} "
          f"= {len(pooled)} unique (E,P) points")
    return pooled


def fit_quadratic(df: pd.DataFrame, scenario: str,
                  e_range: tuple, p_range: tuple
                  ) -> tuple[float, float, float, bool, float]:
    """Fit z = b0 + b1 E + b2 P + b3 E^2 + b4 E P + b5 P^2 on the local box.

    Returns (E*, P*, NPV*, is_max, R-squared).
    """
    col = NPV_COL[scenario]
    m = df.dropna(subset=[col])
    m = m[m.e_cap.between(*e_range) & m.p_cap.between(*p_range)]
    E, P_, z = m.e_cap.values.astype(float), m.p_cap.values.astype(float), m[col].values
    A = np.column_stack([np.ones_like(E), E, P_, E**2, E * P_, P_**2])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    b0, b1, b2, b3, b4, b5 = coef
    H = np.array([[2 * b3, b4], [b4, 2 * b5]])
    is_max = bool((b3 < 0) and (np.linalg.det(H) > 0))
    try:
        star = np.linalg.solve(H, [-b1, -b2])
        Es = float(np.clip(star[0], *e_range))
        Ps = float(np.clip(star[1], *p_range))
        zs = float(np.array([1, Es, Ps, Es**2, Es * Ps, Ps**2]) @ coef)
    except np.linalg.LinAlgError:
        Es = Ps = zs = float("nan")
        is_max = False
    z_hat = A @ coef
    ss_res = float(np.sum((z - z_hat) ** 2))
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return Es, Ps, zs, is_max, r2


def _report_size(fig, ax, stem: str) -> None:
    """Print the canvas and the plot box in inches."""
    fig.canvas.draw()
    w, h = fig.get_size_inches()
    bb = ax.get_position()
    print(f"  {stem:40s} canvas {w:.2f} x {h:.2f} in, "
          f"plot box {bb.width * w:.2f} x {bb.height * h:.2f} in")


def _save(fig, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if OUTPUT in ("pdf", "both"):
        fig.savefig(out_dir / f"{stem}.pdf")
        written.append("pdf")
    if OUTPUT in ("png", "both"):
        fig.savefig(out_dir / f"{stem}.png", dpi=DPI)
        written.append("png")
    plt.close(fig)
    print(f"  wrote {stem}.{{{','.join(written)}}}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4.9 — COARSE-GRID LANDSCAPE WITH REFINEMENT BOX
# ═══════════════════════════════════════════════════════════════════════════

def plot_full_heatmap(df: pd.DataFrame, scenario: str, grid_opt_all: dict,
                      zoom_box: tuple, out_dir: Path) -> None:
    """Coarse-grid NPV landscape with a dashed box marking the refined region.

    Star marks the grid optimum of the focal scenario, omitted when the focal
    scenario is the no-degradation baseline because the diamond already marks
    that point. Diamond marks the no-degradation optimum in every panel, so the
    displacement between star and diamond is the sizing effect of degradation.
    """
    e_lo, e_hi, p_lo, p_hi = zoom_box
    col = NPV_COL[scenario]
    m = df.dropna(subset=[col])
    tri = mtri.Triangulation(m.p_cap.values, m.e_cap.values)

    fig, ax = plt.subplots(figsize=figsize(HEATMAP_FRAC, aspect=ASPECT_FULL))

    cf = ax.tricontourf(tri, m[col].values, levels=18, cmap=CMAP_NPV)
    ax.tricontour(tri, m[col].values, levels=18,
                  colors="k", linewidths=0.20, alpha=0.35)
    cb = fig.colorbar(cf, ax=ax, shrink=0.88)
    cb.set_label("Lifetime NPV  (MEUR)")

    ax.scatter(m.p_cap, m.e_cap, s=S_DOTS_FULL, c="k", alpha=0.30,
               linewidths=0, zorder=3)

    if scenario != "no_deg":
        Eo, Po, _ = grid_opt_all[scenario]
        ax.scatter([Po], [Eo], marker="*", s=S_STAR, c=C_SCENARIO[scenario],
                   edgecolors="k", linewidths=0.7, zorder=6)

    En, Pn, _ = grid_opt_all["no_deg"]
    ax.scatter([Pn], [En], marker="D", s=S_DIAMOND, c=C_NODEG,
               edgecolors="k", linewidths=0.6, zorder=6)

    rect = Rectangle((p_lo, e_lo), p_hi - p_lo, e_hi - e_lo,
                     linewidth=1.0, edgecolor=TUDELFT["navy"],
                     facecolor="none", linestyle="--", zorder=8)
    ax.add_patch(rect)

    ax.set_xlabel("Power capacity  P  (MW)")
    ax.set_ylabel("Energy capacity  E  (MWh)")
    ax.xaxis.set_major_locator(MultipleLocator(TICK_P_FULL))
    ax.yaxis.set_major_locator(MultipleLocator(TICK_E_FULL))
    # Marker legend omitted; the caption explains star, diamond and box.

    _report_size(fig, ax, f"fig_sweep_full_{scenario}")
    _save(fig, f"fig_sweep_full_{scenario}", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4.10 — REFINED-GRID LANDSCAPE
# ═══════════════════════════════════════════════════════════════════════════

def plot_zoom_heatmap(df: pd.DataFrame, scenario: str, grid_opt_all: dict,
                      quad: tuple, zoom_box: tuple, out_dir: Path,
                      levels=None) -> None:
    """Refined-grid NPV landscape.

    Star marks the grid optimum, open circle the quadratic-fit optimum, diamond
    the no-degradation optimum from the same grid. The two panels share the
    colour levels passed in `levels`, so their colorbars are identical and the
    surfaces can be compared directly.
    """
    e_lo, e_hi, p_lo, p_hi = zoom_box
    col = NPV_COL[scenario]
    m = df.dropna(subset=[col])
    tri = mtri.Triangulation(m.p_cap.values, m.e_cap.values)

    fig, ax = plt.subplots(figsize=figsize(HEATMAP_FRAC, aspect=ASPECT_ZOOM))

    lv = 16 if levels is None else levels
    cf = ax.tricontourf(tri, m[col].values, levels=lv, cmap=CMAP_NPV)
    ax.tricontour(tri, m[col].values, levels=lv,
                  colors="k", linewidths=0.20, alpha=0.35)
    cb = fig.colorbar(cf, ax=ax, shrink=0.88)
    cb.set_label("Lifetime NPV  (MEUR)")
    # The colorbar locator is left alone. Matplotlib labels every other one of
    # the 17 contour levels; any override breaks the correspondence between a
    # label and a band of colour.

    ax.scatter(m.p_cap, m.e_cap, s=S_DOTS_ZOOM, c="k", alpha=0.30,
               linewidths=0, zorder=3)

    Eo, Po, _ = grid_opt_all[scenario]
    ax.scatter([Po], [Eo], marker="*", s=S_STAR, c=C_SCENARIO[scenario],
               edgecolors="k", linewidths=0.7, zorder=6)

    Es, Ps, Ns, is_max, r2 = quad
    if is_max:
        ax.scatter([Ps], [Es], marker="o", s=S_CIRCLE, facecolors="none",
                   edgecolors=C_SCENARIO[scenario], linewidths=1.2, zorder=7)
        print(f"  [zoom {scenario}] quadratic optimum: E*={Es:.0f} MWh, "
              f"P*={Ps:.0f} MW, NPV*={Ns:.2f} MEUR, R2={r2:.4f}")
    else:
        print(f"  [zoom {scenario}] quadratic stationary point is not a "
              f"maximum; circle omitted")

    En, Pn, _ = grid_opt_all["no_deg"]
    ax.scatter([Pn], [En], marker="D", s=S_DIAMOND, c=C_NODEG,
               edgecolors="k", linewidths=0.6, zorder=6)

    ax.set_xlabel("Power capacity  P  (MW)")
    ax.set_ylabel("Energy capacity  E  (MWh)")
    ax.set_xlim(p_lo - ZOOM_PAD_P, p_hi + ZOOM_PAD_P)
    ax.set_ylim(e_lo - ZOOM_PAD_E, e_hi + ZOOM_PAD_E)
    ax.xaxis.set_major_locator(MultipleLocator(TICK_P_ZOOM))
    ax.yaxis.set_major_locator(MultipleLocator(TICK_E_ZOOM))
    # Marker legend omitted; the caption explains star, circle and diamond.

    _report_size(fig, ax, f"fig_sweep_zoom_{scenario}")
    _save(fig, f"fig_sweep_zoom_{scenario}", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES 4.11 AND 4.12 — NPV ALONG ONE DESIGN AXIS, ONE SHARED PLOTTER
# ═══════════════════════════════════════════════════════════════════════════

SLICE_SPEC = {
    "E": dict(x="e_cap", fix="p_cap", sym="E", unit="MWh",
              xlabel="Energy capacity  E  (MWh)",
              fix_label="P", fix_unit="MW", stem=STEM_SLICE_E),
    "P": dict(x="p_cap", fix="e_cap", sym="P", unit="MW",
              xlabel="Power capacity  P  (MW)",
              fix_label="E", fix_unit="MWh", stem=STEM_SLICE_P),
}


def plot_npv_slice(pooled: pd.DataFrame, axis: str, fix_val: float,
                   out_dir: Path, shade: bool = SHADE_DEG_COST,
                   legend_loc: str = "lower left") -> None:
    """Lifetime NPV along one design axis at a fixed value of the other axis.

    `axis` is "E", meaning x is energy capacity and P is held fixed, or "P",
    meaning x is power capacity and E is held fixed. Sample points from the
    coarse and refined sweeps are pooled, so each scenario is one continuous
    line and the marker spacing shows where the sampling is denser.
    """
    sp = SLICE_SPEC[axis]
    x, fix = sp["x"], sp["fix"]
    sl = pooled[pooled[fix] == fix_val].sort_values(x)
    if sl.empty:
        raise ValueError(f"No sampled points at {fix} = {fix_val} in either "
                         f"sweep. Choose a value present on at least one grid.")

    n_c = int((sl.src == "coarse").sum())
    n_z = int((sl.src == "refined").sum())
    print(f"  [NPV vs {sp['sym']} at {sp['fix_label']}={fix_val:.0f} "
          f"{sp['fix_unit']}] {len(sl)} points on the slice "
          f"({n_c} coarse, {n_z} refined)")
    if n_c == 0:
        print(f"  [warning] the coarse sweep has no points at "
              f"{sp['fix_label']}={fix_val:.0f}, so pooling adds nothing here "
              f"and the {sp['sym']} range is set by the refined sweep alone")

    fig, ax = plt.subplots(figsize=figsize(SLICE_FRAC, aspect=ASPECT_SLICE))

    series = {}
    for sc, colr in [("no_deg", C_NODEG), ("xu", C_XU), ("shi", C_SHI)]:
        s = sl.dropna(subset=[NPV_COL[sc]])
        series[sc] = s
        if len(s):
            ax.plot(s[x], s[NPV_COL[sc]], color=colr, lw=1.5,
                    ls="solid", marker="o", markersize=4)

    if shade and len(series["no_deg"]) and len(series["xu"]):
        nd, xu = series["no_deg"], series["xu"]
        common = sorted(set(nd[x]) & set(xu[x]))
        nd_v = nd.set_index(x).loc[common, NPV_COL["no_deg"]].values
        xu_v = xu.set_index(x).loc[common, NPV_COL["xu"]].values
        ax.fill_between(common, xu_v, nd_v, color=C_XU, alpha=0.10,
                        linewidth=0, zorder=0)

    # Optimum of the Xu scenario on this slice
    xu_s = series["xu"]
    x_star = float(xu_s.loc[xu_s[NPV_COL["xu"]].idxmax(), x])
    n_star = float(xu_s[NPV_COL["xu"]].max())
    ax.axvline(x_star, color=C_XU, lw=0.9, ls=":", alpha=0.7, zorder=1)
    print(f"      Xu optimum {sp['sym']}*={x_star:.0f} {sp['unit']}, "
          f"NPV*={n_star:.2f} MEUR")

    handles = [
        Line2D([0], [0], color=C_NODEG, lw=1.5, marker="o", markersize=4,
               label="No degradation"),
        Line2D([0], [0], color=C_XU, lw=1.5, marker="o", markersize=4,
               label="Xu et al. (2016)"),
        Line2D([0], [0], color=C_SHI, lw=1.5, marker="o", markersize=4,
               label="Shi et al. (2018)"),
    ]
    if shade:
        handles.append(Patch(facecolor=C_XU, alpha=0.18,
                             label=r"Degradation cost  (no-deg $-$ Xu)"))
    ax.legend(handles=handles, frameon=False, fontsize=FS_LEGEND,
              loc=legend_loc)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.text(x_star + 0.012 * (x1 - x0), n_star + 0.045 * (y1 - y0),
            f"{sp['sym']}*={x_star:.0f} {sp['unit']}",
            fontsize=FS_ANNOT, color=C_XU, va="bottom")

    ax.set_xlabel(sp["xlabel"])
    ax.set_ylabel("Lifetime NPV  (MEUR)")
    ax.axhline(0, color=PALETTE["neutral"], lw=0.7, ls="--", alpha=0.5)

    stem = sp["stem"].format(fix=fix_val)
    _report_size(fig, ax, stem)
    _save(fig, stem, out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    if OUTPUT not in ("png", "pdf", "both"):
        raise ValueError(f"OUTPUT must be 'png', 'pdf' or 'both', got {OUTPUT!r}")

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir = require(SWEEP_DIR)

    print("\nSelecting sweep runs")
    df_coarse, df_zoom = select_runs(sweep_dir)

    # Refinement box, fit domain and axis limits all come from the refined grid
    # itself, so they cannot drift from the data they describe.
    e_lo, e_hi, p_lo, p_hi = _extent(df_zoom)
    zoom_box = (e_lo, e_hi, p_lo, p_hi)
    print(f"  refined grid extent: E {e_lo:.0f} to {e_hi:.0f} MWh, "
          f"P {p_lo:.0f} to {p_hi:.0f} MW")
    if (e_lo, e_hi) != EXPECTED_ZOOM_E or (p_lo, p_hi) != EXPECTED_ZOOM_P:
        print(f"  note: extent differs from the expected "
              f"E {EXPECTED_ZOOM_E[0]:.0f} to {EXPECTED_ZOOM_E[1]:.0f}, "
              f"P {EXPECTED_ZOOM_P[0]:.0f} to {EXPECTED_ZOOM_P[1]:.0f}. "
              f"Figures 4.9 and 4.10 will differ from the committed versions.")

    print("\nGrid optima")
    grid_coarse, grid_zoom = {}, {}
    for sc in ("no_deg", "xu", "shi"):
        grid_coarse[sc] = grid_optimum(df_coarse, sc)
        grid_zoom[sc] = grid_optimum(df_zoom, sc)
        Ec, Pc, Nc = grid_coarse[sc]
        Ez, Pz, Nz = grid_zoom[sc]
        print(f"  {SCENARIO_LABEL[sc]:<22}  "
              f"coarse: E={Ec:.0f}/P={Pc:.0f} {Nc:.2f} MEUR  |  "
              f"refined: E={Ez:.0f}/P={Pz:.0f} {Nz:.2f} MEUR")

    print("\nQuadratic fit on the refined grid")
    quad = {}
    for sc in ("xu", "shi"):
        Es, Ps, Ns, is_max, r2 = fit_quadratic(
            df_zoom, sc, (e_lo, e_hi), (p_lo, p_hi))
        quad[sc] = (Es, Ps, Ns, is_max, r2)
        tag = "maximum" if is_max else "not a maximum"
        print(f"  {SCENARIO_LABEL[sc]:<22}  E*={Es:.0f}  P*={Ps:.0f}  "
              f"NPV*={Ns:.2f} MEUR  R2={r2:.4f}  [{tag}]")

    # Shared colour levels for the two refined panels, pooled over both models.
    other_sc = "shi" if SCENARIO == "xu" else "xu"
    zoom_vals = np.concatenate([
        df_zoom.dropna(subset=[NPV_COL["xu"]])[NPV_COL["xu"]].values,
        df_zoom.dropna(subset=[NPV_COL["shi"]])[NPV_COL["shi"]].values,
    ])
    lo, hi = float(zoom_vals.min()), float(zoom_vals.max())
    pad = 0.02 * (hi - lo)
    zoom_levels = np.linspace(lo - pad, hi + pad, 17)
    print(f"\nShared colour range on the refined panels: "
          f"data {lo:.2f} to {hi:.2f} MEUR, "
          f"level edges {zoom_levels[0]:.2f} to {zoom_levels[-1]:.2f} MEUR")

    print("\nPooling the two sweeps for the slice figures")
    pooled = pool_sweeps(df_coarse, df_zoom)

    print(f"\nHeatmaps drawn at {HEATMAP_FRAC:.2f} x textwidth = "
          f"{HEATMAP_FRAC * TEXTWIDTH_IN:.2f} in, slices at "
          f"{SLICE_FRAC:.2f} x textwidth = {SLICE_FRAC * TEXTWIDTH_IN:.2f} in. "
          f"Include each at the matching width to put ticks at {FS_BASE:.0f} pt "
          f"and axis labels at {FS_LABEL:.0f} pt on the page.")

    print(f"\nWriting figures to {out_dir}")
    print("-" * 78)

    # Figure 4.9
    for sc in FULL_SCENARIOS:
        plot_full_heatmap(df_coarse, sc, grid_coarse, zoom_box, out_dir)

    # Figure 4.10
    plot_zoom_heatmap(df_zoom, SCENARIO, grid_zoom, quad[SCENARIO],
                      zoom_box, out_dir, levels=zoom_levels)
    plot_zoom_heatmap(df_zoom, other_sc, grid_zoom, quad[other_sc],
                      zoom_box, out_dir, levels=zoom_levels)

    # Figures 4.11 and 4.12
    plot_npv_slice(pooled, "E", FIXED_P_MW, out_dir)
    plot_npv_slice(pooled, "P", FIXED_E_MWH, out_dir)

    print("-" * 78)
    print("Done. Caption values are printed above.")


if __name__ == "__main__":
    main()