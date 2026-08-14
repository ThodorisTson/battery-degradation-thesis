"""
WP2 Common Functions
====================
Single source of truth for all data loading and physics calculations.
No calculations live in any run script or YAML file — everything is here.

Design principle
----------------
YAML files store DATA only (raw ERA5 time series, turbine Cp/Ct tables,
battery specs, cable topology).  This module computes everything else:

  1. Weibull fitting       – fit 2-parameter Weibull per directional sector from raw ERA5 hourly wind speeds
  2. Shear scaling         – scale Weibull A from h_ref to hub height
  3. Power curve           – derive P(v) from Cp curve: P = Cp × ½ρ(πD²/4)v³, capped at rated_power
  4. PyWake objects        – XRSite + WindTurbine ready to hand to any model

Entry point
-----------
    setup = quick_setup("WP2_HPP.yaml")
    # setup['site'], setup['windturbine'], setup['x'], setup['y'], ...

Author : Thodoris
Date   : 2026-02-17
"""

from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.stats import weibull_min
from py_wake.site import XRSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

# =============================================================================
# CONSTANTS
# =============================================================================

RHO = 1.225       # Air density [kg/m³] — sea level, 15 °C
N_SECTORS = 12    # Default Weibull sector count
SECTOR_WIDTH = 360.0 / N_SECTORS

# =============================================================================
# DEFAULT SIMULATION CONFIGURATION
# (override any key by passing config={"key": value} to quick_setup)
# =============================================================================

DEFAULT_CONFIG = {
    'interp_n':   2000,   # Dense curve interpolation points (use 10000 for final)
    'wd_step':    5,      # Wind direction bin width for AEP [deg] (use 1 for final)
    'ws_step':    1,      # Wind speed bin width for AEP [m/s]
    'ws_min':     4,      # First wind speed bin [m/s]
    'ws_max':     24,     # Last wind speed bin [m/s]
}

# =============================================================================
# YAML LOADER — resolves !include directives
# =============================================================================

def load_yaml(path: Path | str) -> dict:
    """
    Load a YAML file and resolve !include directives relative to its directory.
    Identical to Jenna's HOPP loader pattern.
    """
    path = Path(path)

    def _include(loader, node):
        target = Path(loader.construct_scalar(node))
        if not target.is_absolute():
            target = path.parent / target
        with open(target, 'r') as fh:
            return yaml.safe_load(fh)

    yaml.add_constructor('!include', _include, Loader=yaml.SafeLoader)
    with open(path, 'r') as fh:
        return yaml.safe_load(fh)

# =============================================================================
# WEIBULL FITTING  (absorbs calculate_wp2_weibull_with_scaling.py)
# =============================================================================

def _sector_mask(wd_deg: np.ndarray, center: float, width: float) -> np.ndarray:
    """Boolean mask for all hours whose wind direction falls in this sector."""
    lo = (center - width / 2) % 360
    hi = (center + width / 2) % 360
    if lo < hi:
        return (wd_deg >= lo) & (wd_deg < hi)
    else:                          # sector straddles 0 / 360
        return (wd_deg >= lo) | (wd_deg < hi)


def fit_weibull_sectors(
    ws: np.ndarray,
    wd: np.ndarray,
    n_sectors: int = N_SECTORS,
) -> dict:
    """
    Fit a 2-parameter Weibull distribution for each directional sector.

    Parameters
    ----------
    ws : 1-D array   hourly wind speeds [m/s] at reference height
    wd : 1-D array   hourly wind directions [deg], meteorological convention
    n_sectors : int  number of equal-width sectors (default 12)

    Returns
    -------
    dict with keys:
        sector_centers  [deg]
        weibull_A       Weibull scale parameter per sector  [m/s]
        weibull_k       Weibull shape parameter per sector  [-]
        frequencies     sector probability (sums to 1)
    """
    sw = 360.0 / n_sectors
    centers = np.arange(sw / 2, 360, sw)   # 15, 45, …, 345

    A_list, k_list, freq_list = [], [], []

    for center in centers:
        mask = _sector_mask(wd, center, sw)
        ws_sec = ws[mask]
        freq_list.append(len(ws_sec) / len(ws))

        if len(ws_sec) > 20:
            shape, _, scale = weibull_min.fit(ws_sec, floc=0)
        else:
            # Sector too sparse — fall back to omnidirectional fit
            shape, _, scale = weibull_min.fit(ws, floc=0)

        k_list.append(float(shape))
        A_list.append(float(scale))

    freq = np.array(freq_list)
    freq /= freq.sum()          # guarantee exact normalisation

    return {
        'sector_centers': centers,
        'weibull_A':      np.array(A_list),
        'weibull_k':      np.array(k_list),
        'frequencies':    freq,
    }


def scale_weibull_to_hub(
    weibull_A: np.ndarray,
    h_ref: float,
    h_hub: float,
    alpha: float,
) -> np.ndarray:
    """
    Scale Weibull A from reference height to hub height using power law.

    A_hub = A_ref × (h_hub / h_ref) ^ alpha
    k is height-independent and is not changed.
    """
    return weibull_A * (h_hub / h_ref) ** alpha

# =============================================================================
# LOADER: WIND RESOURCE  (from WP2_HPP.yaml → site.energy_resource)
# =============================================================================

def load_wind_resource(hpp: dict, verbose: bool = True) -> dict:
    """
    Read wind resource from the HPP dict and compute all Weibull parameters.

    Expects HPP structure:
        site.energy_resource.h_ref
        site.energy_resource.shear.alpha
        site.energy_resource.weibull_fit.n_sectors
        site.energy_resource.weibull_fit.turbulence_intensity
        site.energy_resource.time_series.wind_resource.wind_speed   (list)
        site.energy_resource.time_series.wind_resource.wind_direction (list)

    Returns
    -------
    dict
        name, latitude, longitude, altitude,
        h_ref, shear_alpha,
        sector_centers, weibull_A (at h_ref), weibull_k, frequencies,
        turbulence_intensity
    """
    er  = hpp['site']['energy_resource']
    ts  = er['time_series']['wind_resource']

    ws_raw = np.array(ts['wind_speed'],     dtype=float)
    wd_raw = np.array(ts['wind_direction'],  dtype=float)

    h_ref      = float(er['h_ref'])
    alpha      = float(er['shear']['alpha'])
    n_sectors  = int(er['weibull_fit'].get('n_sectors', N_SECTORS))
    ti         = float(er['weibull_fit'].get('turbulence_intensity', 0.14))

    wb = fit_weibull_sectors(ws_raw, wd_raw, n_sectors)

    if verbose:
        mean_A = np.average(wb['weibull_A'], weights=wb['frequencies'])
        mean_k = np.average(wb['weibull_k'], weights=wb['frequencies'])
        print(f"\n✓ Wind resource  ({len(ws_raw):,} hours, {n_sectors} sectors)")
        print(f"  ERA5 reference height : {h_ref:.0f} m")
        print(f"  Mean wind speed       : {ws_raw.mean():.2f} m/s")
        print(f"  Weibull A (mean)      : {mean_A:.3f} m/s  @ {h_ref:.0f} m")
        print(f"  Weibull k (mean)      : {mean_k:.3f}")
        print(f"  Shear α               : {alpha}")
        print(f"  TI                    : {ti*100:.1f}%")

    return {
        'name':       hpp['site'].get('name', 'WP2_site'),
        'latitude':   hpp['site'].get('latitude'),
        'longitude':  hpp['site'].get('longitude'),
        'altitude':   hpp['site'].get('altitude', 0.0),
        'h_ref':            h_ref,
        'shear_alpha':      alpha,
        'sector_centers':   wb['sector_centers'],
        'weibull_A':        wb['weibull_A'],     # at h_ref — scaling done in build_pywake
        'weibull_k':        wb['weibull_k'],
        'frequencies':      wb['frequencies'],
        'turbulence_intensity': ti,
    }

# =============================================================================
# LOADER: TURBINE  (Cp/Ct → power curve)
# =============================================================================

def load_turbine(hpp: dict, interp_n: int = DEFAULT_CONFIG['interp_n'],
                 verbose: bool = True) -> dict:
    """
    Load NREL 5MW turbine from WP2_Wind_Farm.yaml and compute power curve.

    The YAML stores Cp_curve and Ct_curve (Jenna's format).
    Power curve is derived here:
        P(v) = Cp(v) × ½ρ × (π/4 × D²) × v³
    then capped at rated_power and zeroed outside [cut_in, cut_out].

    Returns
    -------
    dict
        name, diameter, hub_height, rated_power,
        cut_in, cut_out,
        ws_interp, power_interp, ct_interp
    """
    wf = hpp['wind_farm']['turbines']
    perf = wf['performance']

    name        = wf['name']
    rated_power = float(perf['rated_power'])
    diameter    = float(wf['rotor_diameter'])
    hub_height  = float(wf['hub_height'])
    cut_in      = float(wf.get('cut_in_wind_speed', 3.0))
    cut_out     = float(wf.get('cut_out_wind_speed', 25.0))

    # --- Cp curve ---
    ws_cp  = np.array(perf['Cp_curve']['Cp_wind_speeds'], dtype=float)
    cp_val = np.array(perf['Cp_curve']['Cp_values'],      dtype=float)

    # --- Ct curve ---
    ws_ct  = np.array(perf['Ct_curve']['Ct_wind_speeds'], dtype=float)
    ct_val = np.array(perf['Ct_curve']['Ct_values'],      dtype=float)

    # --- Dense interpolation grid ---
    ws_grid = np.linspace(0.0, max(ws_cp.max(), ws_ct.max()) + 5.0, interp_n)

    # --- Power from Cp ---
    A_swept = np.pi / 4.0 * diameter ** 2
    cp_interp = np.interp(ws_grid, ws_cp, cp_val, left=0.0, right=0.0)
    power_interp = cp_interp * 0.5 * RHO * A_swept * ws_grid ** 3
    power_interp = np.minimum(power_interp, rated_power)

    # --- Ct ---
    ct_interp = np.interp(ws_grid, ws_ct, ct_val, left=0.0, right=0.0)

    # --- Zero outside operating range ---
    off = (ws_grid < cut_in) | (ws_grid > cut_out)
    power_interp[off] = 0.0
    ct_interp[off]    = 0.0

    if verbose:
        sp = rated_power / A_swept
        print(f"\n✓ Turbine: {name}")
        print(f"  Rotor diameter  : {diameter:.2f} m")
        print(f"  Hub height      : {hub_height:.1f} m")
        print(f"  Rated power     : {rated_power/1e6:.1f} MW")
        print(f"  Specific power  : {sp:.0f} W/m²")
        print(f"  Cp_peak         : {cp_val.max():.3f}")

    return {
        'name':          name,
        'diameter':      diameter,
        'hub_height':    hub_height,
        'rated_power':   rated_power,
        'cut_in':        cut_in,
        'cut_out':       cut_out,
        'ws_interp':     ws_grid,
        'power_interp':  power_interp,
        'ct_interp':     ct_interp,
    }

# =============================================================================
# LOADER: LAYOUT  (UTM coords → centred local metres)
# =============================================================================

def load_layout(hpp: dict, verbose: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Read turbine layout from WP2_Wind_Farm.yaml.

    UTM Zone 32N coordinates are centred to (0, 0) to avoid floating-point precision issues in PyWake with large absolute coordinate values.

    Returns
    -------
    x, y : np.ndarray   layout in metres, origin at (0, 0)
    n    : int          number of turbines
    """
    coords = hpp['wind_farm']['layouts']['coordinates']
    x = np.array(coords['x'], dtype=float)
    y = np.array(coords['y'], dtype=float)
    x -= x.min()
    y -= y.min()
    n = len(x)

    if verbose:
        cap = n * hpp['wind_farm']['turbines']['performance']['rated_power'] / 1e6
        print(f"\n✓ Layout: {n} turbines, {cap:.0f} MW installed")
        print(f"  Farm extent : {(x.max()-x.min())/1e3:.2f} × {(y.max()-y.min())/1e3:.2f} km")

    return x, y, n

# =============================================================================
# LOADER: BATTERY
# =============================================================================

def load_battery(hpp: dict, verbose: bool = True) -> dict:
    """
    Read battery specifications from battery.yaml.

    Returns flat dict of all battery parameters for use in optimisation.
    """
    bs = hpp['storage_system']
    bat = bs['battery_systems']
    pcu = bs['power_conditioning_unit']
    lim = bs.get('operating_limits', {})
    eco = bs.get('economics', {})
    deg = bs.get('degradation', {})

    result = {
        'name':             bs.get('name', 'Battery'),
        'technology':       bat.get('technology'),
        'energy_capacity_Wh':   float(bat['energy_capacity']),
        'power_capacity_W':     float(bat['power_capacity']),
        'rte_nominal':          float(bat['round_trip_efficiency_nominal']),
        'rte_025C':             float(bat.get('round_trip_efficiency_0.25C', bat['round_trip_efficiency_nominal'])),
        'n_full_load_cycles':   int(bat['n_full_load_cycles']),
        'pcu_efficiency':       float(pcu['efficiency']),
        'n_systems':            int(bs.get('n_systems', 1)),
        'soc_min':              float(lim.get('soc_min', 0.10)),
        'soc_max':              float(lim.get('soc_max', 0.90)),
        'soc_initial':          float(lim.get('soc_initial', 0.50)),
        'dod':                  float(lim.get('dod', 0.80)),
        'chemistry':            deg.get('chemistry', 'LMO'),
        'eol_capacity_fraction': float(deg.get('eol_capacity_fraction', 0.80)),
        'temperature_C':         float(deg.get('temperature_C', 25.0)),
        'capex_EUR_per_kWh':     float(eco.get('capex_EUR_per_kWh', 150.0)),
        'capex_EUR_per_kW':      float(eco.get('capex_EUR_per_kW', 100.0)),
        'repl_energy_EUR_per_kWh': float(eco.get('repl_energy_EUR_per_kWh', 72.0)),   # energy expansion (replacement + deg valuation)
        'repl_power_EUR_per_kW':   float(eco.get('repl_power_EUR_per_kW',   96.0)),    # power expansion (replacement only)
        'opex_EUR_per_kW_year': float(eco.get('opex_EUR_per_kW_year', 7.0)),
        'lifetime_years':        float(eco.get('lifetime_years', 15.0)),
    }

    if verbose:
        e_MWh = result['energy_capacity_Wh'] / 1e6
        p_MW  = result['power_capacity_W']   / 1e6
        print(f"\n✓ Battery: {e_MWh:.0f} MWh / {p_MW:.0f} MW  ({result['technology']})")
        print(f"  RTE             : {result['rte_nominal']*100:.1f}%")
        print(f"  Cycle life      : {result['n_full_load_cycles']:,} cycles")
        print(f"  SoC limits      : {result['soc_min']*100:.0f}% – {result['soc_max']*100:.0f}%  (DoD: {result['dod']*100:.0f}%)")

    return result

# =============================================================================
# LOADER: CABLES
# =============================================================================

def load_cables(hpp: dict, verbose: bool = True) -> dict:
    """
    Read electrical collection array from WP2_Cables.yaml.

    Returns
    -------
    dict with keys:
        substation_x, substation_y   [m]
        edges                        list of (from, to, cable_type) tuples
        cable_catalogue              list of dicts {cross_section, capacity, cost}
    """
    ec = hpp['electrical_collection_array']
    sub = ec['electrical_substations'][0]['electrical_substation']['coordinates']
    sub_x = float(sub['x'][0])
    sub_y = float(sub['y'][0])

    edges = [tuple(e) for e in ec['edges']]

    cat = ec['cables']
    cable_catalogue = [
        {
            'type':          cat['cable_type'][i],
            'cross_section': cat['cross_section'][i],
            'capacity_A':    cat['capacity'][i],
            'cost':          cat['cost'][i],
        }
        for i in range(len(cat['cable_type']))
    ]

    if verbose:
        print(f"\n✓ Cables: {len(edges)} connections, {len(cable_catalogue)} cable types")
        print(f"  Substation at ({sub_x:.1f}, {sub_y:.1f}) m")

    return {
        'substation_x':     sub_x,
        'substation_y':     sub_y,
        'edges':            edges,
        'cable_catalogue':  cable_catalogue,
    }

# =============================================================================
# PYWAKE OBJECT CONSTRUCTION
# =============================================================================

def build_pywake_objects(
    wind_resource: dict,
    turbine: dict,
    config: dict | None = None,
) -> tuple:
    """
    Build PyWake XRSite and WindTurbine objects.

    Weibull A is scaled from h_ref (where ERA5 is given) to hub height using the power-law shear stored in the wind resource dict.

    Parameters
    ----------
    wind_resource : dict   from load_wind_resource()
    turbine       : dict   from load_turbine()
    config        : dict   optional overrides to DEFAULT_CONFIG

    Returns
    -------
    site, windturbine, ws_bins, wd_bins
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    # Scale Weibull A from reference height to hub height
    A_hub = scale_weibull_to_hub(
        wind_resource['weibull_A'],
        h_ref  = wind_resource['h_ref'],
        h_hub  = turbine['hub_height'],
        alpha  = wind_resource['shear_alpha'],
    )

    wd   = wind_resource['sector_centers']
    freq = wind_resource['frequencies']
    k    = wind_resource['weibull_k']
    ti   = wind_resource['turbulence_intensity']

    # Triplicate for cyclic interpolation at 0/360 boundary (PyWake requirement)
    wd_c   = np.concatenate([wd - 360, wd, wd + 360])
    A_c    = np.tile(A_hub, 3)
    k_c    = np.tile(k, 3)
    freq_c = np.tile(freq, 3)

    ws_grid = np.arange(0, 30, 0.5)

    ds = xr.Dataset(
        data_vars={
            'Sector_frequency': ('wd', freq_c),
            'Weibull_A':        ('wd', A_c),
            'Weibull_k':        ('wd', k_c),
            'TI':               ('ws', np.full(len(ws_grid), ti)),
        },
        coords={'wd': wd_c, 'ws': ws_grid},
    )

    site = XRSite(
        ds=ds,
        initial_position=np.array([[0, 0]]),
        interp_method='linear',
        shear=None,
    )

    windturbine = WindTurbine(
        name=turbine['name'],
        diameter=turbine['diameter'],
        hub_height=turbine['hub_height'],
        powerCtFunction=PowerCtTabular(
            ws=turbine['ws_interp'],
            power=turbine['power_interp'],
            power_unit='W',
            ct=turbine['ct_interp'],
        ),
    )

    ws_bins = np.arange(cfg['ws_min'], cfg['ws_max'] + cfg['ws_step'], cfg['ws_step'])
    wd_bins = np.arange(0, 360, cfg['wd_step'])

    return site, windturbine, ws_bins, wd_bins

# =============================================================================
# WAKE MODEL FACTORY  (unchanged interface)
# =============================================================================

def get_wake_model(model_name: str, site, windturbine, **kwargs):
    """
    Factory — return a configured PyWake wake model by name.

    Both NOJ and Bastankhah use identical turbulence modeling and rotor averaging:
    - Crespo-Hernandez turbulence model
    - LinearSum wake superposition
    - EqGridRotorAvg(3) rotor averaging (3×3 grid)

    The ONLY difference is the wake deficit model: NOJ vs Bastankhah-Porte-Agel.

    Supported: 'NOJ', 'Bastankhah'
    """
    from py_wake.deficit_models.noj import NOJ
    from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
    from py_wake.turbulence_models.crespo import CrespoHernandez
    from py_wake.superposition_models import LinearSum
    from py_wake.rotor_avg_models import EqGridRotorAvg

    name = model_name.strip().upper()

    # Shared configuration for fair comparison
    shared_config = {
        'turbulenceModel': CrespoHernandez(),
        'superpositionModel': LinearSum(),
        'rotorAvgModel': EqGridRotorAvg(3),
    }

    if name == 'NOJ':
        return NOJ(
            site, windturbine,
            **shared_config,
            **kwargs
        )

    elif name == 'BASTANKHAH':
        return Bastankhah_PorteAgel_2014(
            site, windturbine,
            k=0.0572, # via the Niayifar–Porté-Agel relation k* = 0.38 I + 0.004 with I=0.14
            **shared_config,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown wake model '{model_name}'. Choose 'NOJ' or 'Bastankhah'.")

# =============================================================================
# QUICK SETUP  (all-in-one entry point for run scripts)
# =============================================================================

def quick_setup(hpp_yaml: str | Path, config: dict | None = None,
                verbose: bool = True) -> dict:
    """
    Load WP2_HPP.yaml and return everything needed to run PyWake analyses.

    Parameters
    ----------
    hpp_yaml : str or Path
        Path to WP2_HPP.yaml  (the master entry-point file)
    config   : dict, optional
        Override any DEFAULT_CONFIG key, e.g. {'wd_step': 1, 'interp_n': 10000}
    verbose  : bool
        Print loading summaries (default True)

    Returns
    -------
    dict with keys:
        hpp             raw HPP config dict (for ad-hoc access)
        site            PyWake XRSite object
        windturbine     PyWake WindTurbine object
        x, y            turbine coordinates [m], origin at (0, 0)
        n_turbines      int
        wind_resource   dict from load_wind_resource()
        turbine_dat     dict from load_turbine()
        battery         dict from load_battery()
        cables          dict from load_cables()
        ws_bins         np.ndarray  for AEP calculation
        wd_bins         np.ndarray  for AEP calculation
        config          the resolved config dict used
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    if verbose:
        print("=" * 72)
        print(" LOADING WP2 HPP CONFIGURATION")
        print("=" * 72)

    hpp = load_yaml(hpp_yaml)

    wind_resource = load_wind_resource(hpp, verbose=verbose)
    turbine_dat   = load_turbine(hpp, interp_n=cfg['interp_n'], verbose=verbose)
    x, y, n       = load_layout(hpp, verbose=verbose)
    battery       = load_battery(hpp, verbose=verbose)
    cables        = load_cables(hpp, verbose=verbose)

    site, windturbine, ws_bins, wd_bins = build_pywake_objects(
        wind_resource, turbine_dat, cfg
    )

    if verbose:
        A_hub = scale_weibull_to_hub(
            wind_resource['weibull_A'],
            wind_resource['h_ref'],
            turbine_dat['hub_height'],
            wind_resource['shear_alpha'],
        )
        mean_A_hub = np.average(A_hub, weights=wind_resource['frequencies'])
        print(f"\n  Weibull A scaled to hub ({turbine_dat['hub_height']:.0f} m): "
              f"{mean_A_hub:.3f} m/s")
        print(f"\n{'='*72}")
        print(" READY")
        print(f"{'='*72}")

    return {
        'hpp':           hpp,
        'site':          site,
        'windturbine':   windturbine,
        'x':             x,
        'y':             y,
        'n_turbines':    n,
        'wind_resource': wind_resource,
        'turbine_dat':   turbine_dat,
        'battery':       battery,
        'cables':        cables,
        'ws_bins':       ws_bins,
        'wd_bins':       wd_bins,
        'config':        cfg,
    }

# =============================================================================
# SHIPP INTEGRATION HELPERS
# =============================================================================

def run_pywake_timeseries(
    setup: dict,
    model_name: str = 'Bastankhah',
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run PyWake time series simulation using ERA5 hourly data.

    This is the bridge function to SHIPP: it produces the hourly wind power time series that SHIPP needs for battery optimization.

    Parameters
    ----------
    setup : dict
        From quick_setup()
    model_name : str
        'NOJ' or 'Bastankhah' (default: Bastankhah for higher accuracy)
    verbose : bool

    Returns
    -------
    power_total_W : np.ndarray  shape (n_hours,)
        Total wind farm power [W] for each hour
    timestamps : list of str
        ISO8601 timestamps for each hour

    Example
    -------
    >>> setup = quick_setup("WP2_HPP.yaml")
    >>> power_W, times = run_pywake_timeseries(setup)
    >>> # Save to CSV for SHIPP
    >>> pd.DataFrame({'time': times, 'power_W': power_W}).to_csv('power.csv')
    """
    if verbose:
        print("\n" + "=" * 60)
        print("RUNNING PYWAKE TIME SERIES SIMULATION")
        print("=" * 60)

    # Extract hourly time series from HPP
    ts = setup['hpp']['site']['energy_resource']['time_series']['wind_resource']
    timestamps = ts['time']
    ws_hourly = np.array(ts['wind_speed'], dtype=float)
    wd_hourly = np.array(ts['wind_direction'], dtype=float)
    
    # Check if hourly TI is available (some ERA5 datasets have this)
    ti_hourly = None
    if 'turbulence_intensity' in ts:
        ti_data = ts['turbulence_intensity']
        # Could be dict with 'data' key or direct array
        if isinstance(ti_data, dict) and 'data' in ti_data:
            ti_hourly = np.array(ti_data['data'], dtype=float)
        elif isinstance(ti_data, (list, np.ndarray)):
            ti_hourly = np.array(ti_data, dtype=float)

    n_hours = len(timestamps)

    if verbose:
        print(f"\nTime series: {n_hours:,} hours")
        print(f"Period: {timestamps[0]} → {timestamps[-1]}")
        print(f"Wind speed: {ws_hourly.mean():.2f} ± {ws_hourly.std():.2f} m/s")
        if ti_hourly is not None:
            print(f"TI: {ti_hourly.mean():.3f} ± {ti_hourly.std():.3f} (hourly values)")
        else:
            print(f"TI: Using constant from config ({setup['wind_resource']['turbulence_intensity']:.2f})")
        print(f"\nBuilding {model_name} wake model...")

    # Build wake model
    wf_model = get_wake_model(model_name, setup['site'], setup['windturbine'])

    if verbose:
        print(f"Running simulation ({n_hours:,} hours, {setup['n_turbines']} turbines)...")
        print("(This may take 1-2 minutes...)")

    # Run simulation with or without hourly TI
    sim_kwargs = {
        'x': setup['x'],
        'y': setup['y'],
        'wd': wd_hourly,
        'ws': ws_hourly,
        'time': True,
    }
    if ti_hourly is not None:
        sim_kwargs['TI'] = ti_hourly
    
    sim_res = wf_model(**sim_kwargs)

    # Extract total power
    power_per_turbine = sim_res.Power.values  # [n_turbines, n_hours]
    power_total = power_per_turbine.sum(axis=0)  # [n_hours]

    if verbose:
        aep_gwh = power_total.sum() / 1e9
        cf = power_total.mean() / (setup['n_turbines'] * setup['turbine_dat']['rated_power'])
        print(f"\n✓ Simulation complete!")
        print(f"  AEP: {aep_gwh:.1f} GWh")
        print(f"  Capacity factor: {cf*100:.2f}%")
        print(f"  Peak power: {power_total.max()/1e6:.1f} MW")
        print("=" * 60)

    return power_total, timestamps


def export_power_timeseries_for_shipp(
    power_W: np.ndarray,
    timestamps: list[str],
    output_path: str | Path,
) -> None:
    """
    Export PyWake power time series to CSV for SHIPP battery optimization.

    Parameters
    ----------
    power_W : np.ndarray
        Hourly wind farm power [W]
    timestamps : list of str
        ISO8601 timestamp strings
    output_path : str or Path
        Where to save CSV

    Creates CSV with columns: timestamp, power_W, power_MW
    """
    import pandas as pd

    df = pd.DataFrame({
        'timestamp': timestamps,
        'power_W': power_W,
        'power_MW': power_W / 1e6,
    })

    df.to_csv(output_path, index=False)
    print(f"✓ Exported {len(df):,} hours → {Path(output_path).name}")