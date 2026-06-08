#!/usr/bin/env python
"""Comprehensive data quality check for NW Pacific SST + Wind NetCDF files.

Usage:
    python scripts/datasets/inspect_nwp_data.py
"""
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

# ─── Paths ───
WIND_PATH = r"/home/lab/zhangxm/gxy/Earthformer/datasets/SST-PREDICT/Wind_cropped.nc"
SST_PATH = r"/home/lab/zhangxm/gxy/Earthformer/datasets/SST-PREDICT/SST_cropped.nc"


# WIND_PATH = r"F:\Data\ERA5 SST\SST.nc"
# SST_PATH = r"F:\Data\ERA5 SST\SST.nc"

# Known fill values
FILL_FLOAT32 = 3.4028234663852886e+38
MISSING_MARKER = 3.4e38


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def inspect_file(path, name):
    """Inspect a single NetCDF file and return key stats."""
    print_section(f"FILE: {name}")
    print(f"  Path: {path}")

    if not os.path.exists(path):
        print(f"  [ERROR] File not found!")
        return None

    ds = xr.open_dataset(path)

    print(f"  Dims:       {dict(ds.dims)}")
    print(f"  Coords:     {list(ds.coords.keys())}")
    print(f"  Variables:  {list(ds.data_vars.keys())}")

    info = {'name': name, 'dims': dict(ds.dims), 'vars': {}}

    # ── Time dimension ──
    time_dim = None
    for dim_name in ds.dims:
        if 'time' in dim_name.lower():
            time_dim = dim_name
            break

    if time_dim is None:
        print(f"  [WARN] No time dimension found in dims: {list(ds.dims)}")
    else:
        info['time_dim'] = time_dim
        info['time_len'] = ds.dims[time_dim]
        time_vals = ds[time_dim].values
        print(f"\n  --- Time ({time_dim}, {len(time_vals)} steps) ---")
        print(f"  First:     {time_vals[0]}")
        print(f"  Last:      {time_vals[-1]}")
        print(f"  Dtype:     {time_vals.dtype}")

        # Convert to datetime if possible
        try:
            if np.issubdtype(time_vals.dtype, np.integer):
                # Seconds since 1970-01-01 (ERA5 convention)
                try:
                    unit_str = ds[time_dim].units
                    print(f"  Units:     {unit_str}")
                except AttributeError:
                    pass
                # Try to interpret as seconds since epoch
                if time_vals[0] > 1e9:
                    dates = pd.to_datetime(time_vals, unit='s')
                elif time_vals[0] > 1e8:
                    dates = pd.to_datetime(time_vals, unit='s')
                else:
                    dates = pd.to_datetime(time_vals, unit='s')
            else:
                dates = pd.to_datetime(time_vals)

            print(f"  Date range: {dates[0]} → {dates[-1]}")

            # Check for gaps
            if len(dates) > 1:
                diffs = np.diff(dates)
                unique_diffs, counts = np.unique(diffs, return_counts=True)
                # Most common diff
                dominant_idx = np.argmax(counts)
                typical_step = unique_diffs[dominant_idx]
                print(f"  Typical step: {typical_step}  ({counts[dominant_idx]}/{len(diffs)} steps)")
                # Find gaps > 2x typical
                gap_mask = diffs > 2 * typical_step
                if gap_mask.any():
                    gap_indices = np.where(gap_mask)[0]
                    print(f"  [WARN] {len(gap_indices)} gaps > 2× typical step found:")
                    for gi in gap_indices[:5]:
                        print(f"    At step {gi}: {dates[gi]} → {dates[gi+1]} (gap={diffs[gi]})")
                else:
                    print(f"  Time is continuous (no gaps > 2× typical step)")

            # Detect data frequency
            freq_days = typical_step / np.timedelta64(1, 'D')
            info['typical_hours'] = freq_days * 24
            if freq_days < 0.05:
                print(f"  Frequency:  ~{freq_days*24:.0f}h (hourly/sub-daily)")
            elif freq_days < 1.5:
                print(f"  Frequency:  ~{freq_days:.1f}d (daily)")
            elif freq_days < 10:
                print(f"  Frequency:  ~{freq_days:.1f}d (weekly/sub-monthly)")
            elif freq_days < 40:
                print(f"  Frequency:  ~{freq_days:.0f}d (monthly)")
            else:
                print(f"  Frequency:  ~{freq_days:.0f}d (sub-annual+)")

        except Exception as e:
            print(f"  [WARN] Could not parse time: {e}")

    # ── Spatial dimensions ──
    lat_dim, lon_dim = None, None
    for dim_name in ds.dims:
        if 'lat' in dim_name.lower():
            lat_dim = dim_name
        if 'lon' in dim_name.lower():
            lon_dim = dim_name

    if lat_dim:
        lat_vals = ds[lat_dim].values
        print(f"\n  --- Latitude ({lat_dim}, {len(lat_vals)} pts) ---")
        print(f"  Range:      {lat_vals[0]:.4f} → {lat_vals[-1]:.4f}")
        print(f"  Resolution: {abs(lat_vals[1] - lat_vals[0]):.4f}°")
        print(f"  Direction:  {'increasing' if lat_vals[1] > lat_vals[0] else 'decreasing'}")
        info['lat_range'] = (float(lat_vals[0]), float(lat_vals[-1]))
        info['lat_res'] = float(abs(lat_vals[1] - lat_vals[0]))
        info['lat_dir'] = 'inc' if lat_vals[1] > lat_vals[0] else 'dec'
        info['n_lat'] = len(lat_vals)

    if lon_dim:
        lon_vals = ds[lon_dim].values
        print(f"\n  --- Longitude ({lon_dim}, {len(lon_vals)} pts) ---")
        print(f"  Range:      {lon_vals[0]:.2f} → {lon_vals[-1]:.2f}")
        print(f"  Resolution: {abs(lon_vals[1] - lon_vals[0]):.4f}°")
        print(f"  Direction:  {'increasing' if lon_vals[1] > lon_vals[0] else 'decreasing'}")
        info['lon_range'] = (float(lon_vals[0]), float(lon_vals[-1]))
        info['lon_res'] = float(abs(lon_vals[1] - lon_vals[0]))
        info['n_lon'] = len(lon_vals)

    # ── Data variables ──
    for var_name in ds.data_vars:
        var = ds[var_name]
        vals = var.values
        n_total = vals.size

        print(f"\n  --- Variable: {var_name} ---")
        print(f"  Shape:      {var.shape}")
        print(f"  Dtype:      {var.dtype}")
        print(f"  Total cells:{n_total:,}")

        # Basic stats (skip fill values)
        finite_mask = np.isfinite(vals)
        # Also exclude the known fill value
        fill_mask = np.abs(vals) > 1e10
        valid_mask = finite_mask & (~fill_mask)
        n_valid = valid_mask.sum()
        n_nan = (~finite_mask).sum()
        n_fill = fill_mask.sum() - n_nan  # explicit fill values (3.4e38)
        n_missing = n_total - n_valid

        print(f"  Finite:     {n_valid:,} ({100*n_valid/n_total:.2f}%)")
        print(f"  NaN:        {n_nan:,} ({100*n_nan/n_total:.4f}%)")
        print(f"  Fill (3.4e38): {n_fill:,} ({100*n_fill/n_total:.4f}%)" if n_fill > 0 else f"  Fill (3.4e38): 0")
        print(f"  Total missing: {n_missing:,} ({100*n_missing/n_total:.2f}%)")

        if n_valid > 0:
            valid_vals = vals[valid_mask]
            print(f"  Min:        {valid_vals.min():.4f}")
            print(f"  Max:        {valid_vals.max():.4f}")
            print(f"  Mean:       {valid_vals.mean():.4f}")
            print(f"  Std:        {valid_vals.std():.4f}")
            print(f"  1st %ile:   {np.percentile(valid_vals, 1):.4f}")
            print(f"  99th %ile:  {np.percentile(valid_vals, 99):.4f}")

            # Check for physically unreasonable values
            if 'sst' in var_name.lower():
                # SST should be roughly 270–310K or -2–35°C
                if valid_vals.max() > 300:  # Probably Kelvin
                    too_hot = (valid_vals > 310).sum()
                    too_cold = (valid_vals < 270).sum()
                    print(f"  [Range check K] >310K: {too_hot}, <270K: {too_cold}")
                else:  # Probably Celsius
                    too_hot = (valid_vals > 35).sum()
                    too_cold = (valid_vals < -5).sum()
                    print(f"  [Range check °C] >35°C: {too_hot}, <-5°C: {too_cold}")

            if var_name in ['u10', 'v10', 'u', 'v']:
                too_strong = (np.abs(valid_vals) > 50).sum()
                print(f"  [Wind check] |wind|>50m/s: {too_strong} cells ({100*too_strong/n_valid:.4f}%)")

        # Missing rate per time step
        if time_dim and valid_mask.ndim >= 1:
            time_axis = var.dims.index(time_dim)
            axes_for_spatial = tuple(i for i in range(valid_mask.ndim) if i != time_axis)
            missing_per_time = (~valid_mask).sum(axis=axes_for_spatial) / max(1, np.prod([var.shape[a] for a in axes_for_spatial]))
            high_miss = (missing_per_time > 0.1).sum()
            all_miss = (missing_per_time > 0.99).sum()
            print(f"  Time steps with >10% missing: {high_miss}/{len(missing_per_time)}")
            if all_miss > 0:
                print(f"  [WARN] {all_miss} time steps have >99% missing!")

        # Per-lat missing rate
        if lat_dim and valid_mask.ndim >= 1 and lon_dim:
            lat_axis = var.dims.index(lat_dim)
            lon_axis = var.dims.index(lon_dim)
            axes_for_lat = tuple(i for i in range(valid_mask.ndim) if i != lat_axis)
            missing_per_lat = (~valid_mask).sum(axis=axes_for_lat) / max(1, valid_mask.size // valid_mask.shape[lat_axis])

        info['vars'][var_name] = {
            'shape': var.shape,
            'n_valid': int(n_valid),
            'n_missing': int(n_missing),
            'pct_missing': float(100 * n_missing / n_total),
            'min': float(valid_vals.min()) if n_valid > 0 else None,
            'max': float(valid_vals.max()) if n_valid > 0 else None,
            'mean': float(valid_vals.mean()) if n_valid > 0 else None,
        }

    ds.close()
    return info


def check_cross_file_consistency(info1, info2):
    """Check that two nc files are compatible for merging."""
    print_section("CROSS-FILE CONSISTENCY CHECK")

    if info1 is None or info2 is None:
        print("  Cannot check — one or both files missing.")
        return

    # Time
    t1_len = info1.get('time_len')
    t2_len = info2.get('time_len')
    print(f"\n  Time steps: file1={t1_len}, file2={t2_len}")
    if t1_len != t2_len:
        print(f"  [ERROR] Time dimensions differ! Cannot merge.")
    else:
        print(f"  [OK] Time dimensions match: {t1_len} steps")

    # Lat
    nlat1, nlat2 = info1.get('n_lat'), info2.get('n_lat')
    print(f"  Latitude pts: file1={nlat1}, file2={nlat2}")
    if nlat1 != nlat2:
        print(f"  [ERROR] Latitude dimensions differ!")
    elif info1.get('lat_range') and info2.get('lat_range'):
        lat1_0, lat1_1 = info1['lat_range']
        lat2_0, lat2_1 = info2['lat_range']
        if abs(lat1_0 - lat2_0) < 0.01 and abs(lat1_1 - lat2_1) < 0.01:
            print(f"  [OK] Latitude ranges match: {lat1_0:.2f} → {lat1_1:.2f}")
        else:
            print(f"  [WARN] Latitude ranges differ: ({lat1_0:.2f},{lat1_1:.2f}) vs ({lat2_0:.2f},{lat2_1:.2f})")

    # Lon
    nlon1, nlon2 = info1.get('n_lon'), info2.get('n_lon')
    print(f"  Longitude pts: file1={nlon1}, file2={nlon2}")
    if nlon1 != nlon2:
        print(f"  [ERROR] Longitude dimensions differ!")
    elif info1.get('lon_range') and info2.get('lon_range'):
        lon1_0, lon1_1 = info1['lon_range']
        lon2_0, lon2_1 = info2['lon_range']
        if abs(lon1_0 - lon2_0) < 0.01 and abs(lon1_1 - lon2_1) < 0.01:
            print(f"  [OK] Longitude ranges match: {lon1_0:.2f} → {lon1_1:.2f}")
        else:
            print(f"  [WARN] Longitude ranges differ: ({lon1_0:.2f},{lon1_1:.2f}) vs ({lon2_0:.2f},{lon2_1:.2f})")

    # Frequency
    h1 = info1.get('typical_hours')
    h2 = info2.get('typical_hours')
    if h1 and h2 and abs(h1 - h2) < 1:
        print(f"  [OK] Time frequencies match: ~{h1:.1f}h")
    elif h1 and h2:
        print(f"  [WARN] Time frequencies differ: ~{h1:.1f}h vs ~{h2:.1f}h")


def print_summary(info1, info2):
    """Print summary table."""
    print_section("SUMMARY")

    print(f"\n  {'':-<50}")
    print(f"  {'Metric':<30s} {'SST.nc':>10s} {'Wind.nc':>10s}")
    print(f"  {'':-<50}")

    rows = [
        ('Grid (lat × lon)', f"{info1['n_lat']}×{info1['n_lon']}", f"{info2['n_lat']}×{info2['n_lon']}"),
        ('Time steps', f"{info1['time_len']}", f"{info2['time_len']}"),
        ('Lat resolution', f"{info1['lat_res']:.4f}°", f"{info2['lat_res']:.4f}°"),
        ('Lon resolution', f"{info1['lon_res']:.4f}°", f"{info2['lon_res']:.4f}°"),
        ('Lat range', f"{info1['lat_range'][0]:.1f}→{info1['lat_range'][1]:.1f}",
         f"{info2['lat_range'][0]:.1f}→{info2['lat_range'][1]:.1f}"),
        ('Lon range', f"{info1['lon_range'][0]:.1f}→{info1['lon_range'][1]:.1f}",
         f"{info2['lon_range'][0]:.1f}→{info2['lon_range'][1]:.1f}"),
    ]
    for name, v1, v2 in rows:
        print(f"  {name:<30s} {v1:>10s} {v2:>10s}")

    # Variable-specific
    for var_name in info1.get('vars', {}):
        v1 = info1['vars'][var_name]
        v2 = info2['vars'].get(var_name)
        if v2 is None:
            print(f"\n  --- {var_name} (SST.nc only) ---")
            print(f"  Min: {v1['min']:.4f}  Max: {v1['max']:.4f}  Miss: {v1['pct_missing']:.2f}%")
    for var_name in info2.get('vars', {}):
        v2 = info2['vars'][var_name]
        v1 = info1['vars'].get(var_name)
        if v1 is None:
            print(f"\n  --- {var_name} (Wind.nc only) ---")
            print(f"  Min: {v2['min']:.4f}  Max: {v2['max']:.4f}  Miss: {v2['pct_missing']:.2f}%")
        else:
            print(f"\n  --- {var_name} ---")
            print(f"  SST.nc:  min={v1['min']:.4f}  max={v1['max']:.4f}  miss={v1['pct_missing']:.2f}%")
            print(f"  Wind.nc: min={v2['min']:.4f}  max={v2['max']:.4f}  miss={v2['pct_missing']:.2f}%")

    print(f"\n  {'':-<50}")


def main():
    print("=" * 70)
    print("  NW Pacific Data Quality Check")
    print(f"  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    info_wind = inspect_file(WIND_PATH, "Wind (u10, v10)")
    info_sst = inspect_file(SST_PATH, "SST")

    if info_wind and info_sst:
        check_cross_file_consistency(info_wind, info_sst)
        print_summary(info_sst, info_wind)
    else:
        print("\n[ERROR] One or both files missing. Check paths.")
        for p in [WIND_PATH, SST_PATH]:
            print(f"  {'[EXISTS]' if os.path.exists(p) else '[MISSING]'} {p}")


    print("\nDone.")


if __name__ == "__main__":
    main()
