"""
plot_wind_resource_final.py
---------------------------
Final wind-resource figures for the WP2 HPP, from the site ERA5 series at 90 m.

  FIG 1 (body)     : omnidirectional wind-speed histogram + omnidirectional Weibull fit
  FIG 2 (body)     : directional wind rose, speed bins = turbine operating states
  FIG 3 (appendix) : 12 directional Weibull curves, distinct colours (qualitative map)

Numbers (sector fits) replicate wp2_common.fit_weibull_sectors exactly so the figures match the resource the wake model consumes.

Thesis style: vector PDF, no bbox_inches='tight', constrained_layout, no titles, TU Delft palette for the body figures, fonts >= 9 pt. Captions carry all text.
Data = ERA5 reanalysis at the nearest grid cell (NOT mast data) -> state in caption.
Deps: numpy, scipy, matplotlib, pyyaml. Reproducible in VS Code on Windows.
"""
from pathlib import Path
import numpy as np
import yaml
from scipy.stats import weibull_min
import matplotlib.pyplot as plt

HERE = Path(__file__).parent if "__file__" in globals() else Path(".")
RESOURCE_YAML = HERE / "wind_resource_2022_era5_90m.yaml"
N_SECTORS = 12

# turbine operating-state breakpoints (NREL 5 MW): cut-in 3, rated ~11.4, cut-out 25
CUT_IN, MID, RATED, CUT_OUT = 3.0, 7.0, 11.4, 25.0

NAVY, CYAN, RED, ORANGE, PURPLE = "#0C2340", "#00A6D6", "#A50034", "#EC6842", "#6E2585"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
})
TW = 5.5  # text width [in]

# ---------- load ----------
wr = yaml.safe_load(open(RESOURCE_YAML))["wind_resource"]
ws = np.asarray(wr["wind_speed"], float)
wd = np.asarray(wr["wind_direction"], float) % 360.0
ok = np.isfinite(ws) & np.isfinite(wd) & (ws > 0)
ws, wd = ws[ok], wd[ok]
xx = np.linspace(0, ws.max(), 400)

# ---------- sector fits (loader logic) ----------
def sector_mask(wd, c, sw):
    lo, hi = (c - sw/2) % 360, (c + sw/2) % 360
    return (wd >= lo) & (wd < hi) if lo < hi else (wd >= lo) | (wd < hi)
sw = 360.0 / N_SECTORS
centers = np.arange(sw/2, 360, sw)
A, K, F = [], [], []
for c in centers:
    s = ws[sector_mask(wd, c, sw)]; F.append(len(s)/len(ws))
    k, _, a = weibull_min.fit(s, floc=0) if len(s) > 20 else weibull_min.fit(ws, floc=0)
    A.append(float(a)); K.append(float(k))
A, K, F = np.array(A), np.array(K), np.array(F); F /= F.sum()
k_omni, _, A_omni = weibull_min.fit(ws, floc=0)
print(f"omni A={A_omni:.2f} k={k_omni:.2f} mean={ws.mean():.2f} | sector-mean k={np.average(K,weights=F):.2f}")

# ============ FIG 1: omnidirectional (BODY) ============
fig1, ax = plt.subplots(figsize=(TW, TW*0.60), constrained_layout=True)
bins = np.arange(0, np.ceil(ws.max())+1, 1.0)
ax.hist(ws, bins=bins, density=True, color=CYAN, alpha=0.55,
        edgecolor="white", linewidth=0.5, label="ERA5 hourly 2022")
ax.plot(xx, weibull_min.pdf(xx, k_omni, 0, A_omni), color=RED, lw=2.2,
        label=f"Weibull fit (A = {A_omni:.2f} m/s, k = {k_omni:.2f})")
ax.axvline(ws.mean(), color=NAVY, lw=1.2, ls="--", label=f"mean = {ws.mean():.2f} m/s")
ax.set_xlabel("Wind speed at 90 m hub height  [m/s]")
ax.set_ylabel("Probability density  [-]")
ax.set_xlim(0, np.ceil(ws.max()))
ax.legend(frameon=False, loc="upper right")
fig1.savefig(HERE/"wind_weibull_omni.pdf"); fig1.savefig(HERE/"wind_weibull_omni.png", dpi=150)

# ============ FIG 2: wind rose, turbine-state bins (BODY) ============
speed_bins   = [0, CUT_IN, MID, RATED, CUT_OUT, np.inf]
speed_labels = [f"0-{CUT_IN:.0f} (below cut-in)",
                f"{CUT_IN:.0f}-{MID:.0f} (partial load)",
                f"{MID:.0f}-{RATED:.1f} (approaching rated)",
                f"{RATED:.1f}-{CUT_OUT:.0f} (at/above rated)",
                f">{CUT_OUT:.0f} (above cut-out)"]
rose_colors  = [CYAN, NAVY, ORANGE, RED, PURPLE]
idx = (((wd + sw/2) % 360) // sw).astype(int)
freq = np.zeros((N_SECTORS, len(speed_bins)-1))
for s in range(N_SECTORS):
    sub = ws[idx == s]
    for b in range(len(speed_bins)-1):
        freq[s, b] = np.sum((sub >= speed_bins[b]) & (sub < speed_bins[b+1]))
freq = 100.0 * freq / len(ws)
theta = np.deg2rad(np.arange(N_SECTORS) * sw)
fig2 = plt.figure(figsize=(TW*0.92, TW*0.92), constrained_layout=True)
ax2 = fig2.add_subplot(111, projection="polar")
ax2.set_theta_zero_location("N"); ax2.set_theta_direction(-1)
bottom = np.zeros(N_SECTORS)
for b in range(len(speed_bins)-1):
    if freq[:, b].sum() == 0:   # skip empty bins (e.g. above cut-out) but keep colour mapping
        continue
    ax2.bar(theta, freq[:, b], width=np.deg2rad(sw)*0.9, bottom=bottom,
            color=rose_colors[b], edgecolor="white", linewidth=0.4, label=speed_labels[b])
    bottom += freq[:, b]
ax2.set_xticks(np.deg2rad([0,45,90,135,180,225,270,315]))
ax2.set_xticklabels(["N","NE","E","SE","S","SW","W","NW"])
ax2.set_yticks(np.arange(0, bottom.max()+2, 4))
ax2.tick_params(labelsize=8)
ax2.legend(loc="lower left", bbox_to_anchor=(-0.22, -0.16), frameon=False, fontsize=8)
fig2.savefig(HERE/"wind_rose_turbinebins.pdf"); fig2.savefig(HERE/"wind_rose_turbinebins.png", dpi=150)
print(f"rose: above cut-out hours = {(ws>CUT_OUT).sum()} (bin shown only if non-empty)")

# ============ FIG 3: 12 sector Weibulls, DISTINCT colours (APPENDIX) ============
# qualitative palette: 12 visually distinct hues (tab20 evens + a couple extras)
distinct = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b",
            "#e377c2","#7f7f7f","#bcbd22","#17becf","#393b79","#a55194"]
ls_cycle = ["-","-","-","-","-","-","--","--","--","--","--","--"]
fig3, ax = plt.subplots(figsize=(TW, TW*0.66), constrained_layout=True)
for i, c in enumerate(centers):
    ax.plot(xx, weibull_min.pdf(xx, K[i], 0, A[i]), color=distinct[i], ls=ls_cycle[i],
            lw=1.3, label=f"{c:.0f}\u00b0 ({F[i]*100:.0f}%)")
ax.plot(xx, weibull_min.pdf(xx, k_omni, 0, A_omni), color="k", lw=2.6, label="omnidirectional")
ax.set_xlabel("Wind speed at 90 m hub height  [m/s]")
ax.set_ylabel("Probability density  [-]")
ax.set_xlim(0, np.ceil(ws.max()))
ax.legend(frameon=False, ncol=2, fontsize=8, title="sector centre (freq.)", title_fontsize=8)
fig3.savefig(HERE/"weibull_appendix_12.pdf"); fig3.savefig(HERE/"weibull_appendix_12.png", dpi=150)
print("saved: wind_weibull_omni, wind_rose_turbinebins, weibull_appendix_12 (pdf+png)")
