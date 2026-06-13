#!/usr/bin/env python
"""Inspect SLA data — dimensions, variables, coordinates, coverage.

Usage:
    python scripts/datasets/inspect_sla_data.py
"""

import os
import sys
import numpy as np
import xarray as xr
from datetime import datetime

SLA_ROOT = "/home/lab/zhangxm/tzjzllllll/sladata"

# Sub-directories to inspect
SUBDIRS = ["slatest", "slatrain", "slaval", "sla 21-25"]


def print_header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def inspect_subdir(subdir_name):
    """Inspect a single SLA sub-directory."""
    subdir_path = os.path.join(SLA_ROOT, subdir_name)

    if not os.path.isdir(subdir_path):
        print(f"\n  [WARN] Directory not found: {subdir_path}")
        return None

    # ── Find available years ──
    years = sorted([
        d for d in os.listdir(subdir_path)
        if os.path.isdir(os.path.join(subdir_path, d)) and d.isdigit()
    ])
    if not years:
        print(f"\n  [WARN] No year directories in {subdir_path}")
        return None

    yr_min, yr_max = years[0], years[-1]
    print(f"\n  Years: {yr_min} – {yr_max} ({len(years)} years)")

    # ── Count total files ──
    total_files = 0
    for yr in years:
        yr_path = os.path.join(subdir_path, yr)
        months = os.listdir(yr_path)
        for mo in months:
            mo_path = os.path.join(yr_path, mo)
            if os.path.isdir(mo_path):
                total_files += len([f for f in os.listdir(mo_path) if f.endswith('.nc')])
    print(f"  Total .nc files: {total_files}")

    # ── Read one sample file ──
    # pick middle year, first month
    mid_yr = years[len(years) // 2]
    yr_path = os.path.join(subdir_path, mid_yr)
    months = sorted(os.listdir(yr_path))
    first_mo = months[0]
    mo_path = os.path.join(yr_path, first_mo)
    sample_file = sorted([f for f in os.listdir(mo_path) if f.endswith('.nc')])[0]
    sample_path = os.path.join(mo_path, sample_file)

    print(f"  Sample file: {sample_path}")
    print()

    ds = xr.open_dataset(sample_path)

    # ── Full dataset info ──
    print(f"  --- Dataset structure ---")
    print(f"  {ds}")

    # ── Dimensions ──
    print(f"\n  --- Dimensions ---")
    for dim_name, dim_size in ds.dims.items():
        print(f"  {dim_name}: {dim_size}")

    # ── Data variables ──
    print(f"\n  --- Data variables ---")
    for var_name in ds.data_vars:
        var = ds[var_name]
        print(f"  '{var_name}': shape={var.shape}, dtype={var.dtype}, dims={list(var.dims)}")
        attrs = var.attrs
        if attrs:
            for k, v in attrs.items():
                val_str = str(v)[:80]
                print(f"      {k}: {val_str}")

    # ── Coordinate ranges ──
    print(f"\n  --- Coordinate ranges ---")
    lat_vals = None
    lon_vals = None
    for coord_name in ds.coords:
        coord = ds[coord_name]
        if coord_name == 'time' or coord_name == 'valid_time':
            print(f"  time: {coord.values[0]} → {coord.values[-1]}")
            print(f"        dtype={coord.dtype}")
        elif 'lat' in coord_name.lower():
            lat_vals = coord.values
            print(f"  {coord_name}: [{float(lat_vals.min()):.4f}, {float(lat_vals.max()):.4f}]  "
                  f"shape={lat_vals.shape}  "
                  f"ascending={bool(lat_vals[1] > lat_vals[0])}")
            if len(lat_vals) > 1:
                diffs = np.diff(lat_vals)
                print(f"        step: mean={float(diffs.mean()):.5f}, "
                      f"min={float(diffs.min()):.5f}, max={float(diffs.max()):.5f}")
        elif 'lon' in coord_name.lower():
            lon_vals = coord.values
            print(f"  {coord_name}: [{float(lon_vals.min()):.4f}, {float(lon_vals.max()):.4f}]  "
                  f"shape={lon_vals.shape}  "
                  f"ascending={bool(lon_vals[1] > lon_vals[0])}")
            if len(lon_vals) > 1:
                diffs = np.diff(lon_vals)
                print(f"        step: mean={float(diffs.mean()):.5f}, "
                      f"min={float(diffs.min()):.5f}, max={float(diffs.max()):.5f}")

    # ── Sample values ──
    var_names = list(ds.data_vars)
    main_var = var_names[0]
    data = ds[main_var].values
    print(f"\n  --- '{main_var}' value stats (sample file) ---")
    print(f"  shape:  {data.shape}")
    print(f"  min:    {float(np.nanmin(data)):.4f}")
    print(f"  max:    {float(np.nanmax(data)):.4f}")
    print(f"  mean:   {float(np.nanmean(data)):.4f}")
    print(f"  std:    {float(np.nanstd(data)):.4f}")
    nan_count = int(np.isnan(data).sum())
    total_count = int(np.prod(data.shape))
    print(f"  NaN:    {nan_count} / {total_count} ({nan_count/total_count*100:.1f}%)")

    # Snapshot coords before closing
    lat0 = float(lat_vals[0]) if lat_vals is not None else None
    lon0 = float(lon_vals[0]) if lon_vals is not None else None

    ds.close()
    return {
        'name': subdir_name,
        'years': (int(yr_min), int(yr_max)),
        'total_files': total_files,
        'var_names': var_names,
        'lat_start': lat0,
        'lon_start': lon0,
        'nan_pct': nan_count / total_count * 100,
        'sample_shape': data.shape,
    }


def compare_grids(results):
    """Compare coordinate grids across sub-directories."""
    print_header("Cross-Directory Grid Comparison")

    valid = [r for r in results if r is not None]
    if len(valid) < 2:
        print("  Not enough subdirs to compare.")
        return

    # Check if all variable names match
    print(f"\n  Variable names:")
    for r in valid:
        print(f"    {r['name']}: {r['var_names']}")

    # Check year ranges
    print(f"\n  Year coverage:")
    for r in valid:
        y0, y1 = r['years']
        print(f"    {r['name']:15s}: {y0} – {y1}  ({r['total_files']} files)")


def check_overlap_with_ssta(data_dir="/home/lab/zhangxm/gxy/Earthformer/datasets/SST-PREDICT/"):
    """Check time alignment with existing SSTA data."""
    print_header("Time Alignment with SSTA")

    ssta_path = os.path.join(data_dir, "ssta.nc")
    if not os.path.exists(ssta_path):
        print(f"  SSTA file not found: {ssta_path}")
        print(f"  (skip — run on server to see this section)")
        return

    ds_ssta = xr.open_dataset(ssta_path)
    ssta_times = ds_ssta.valid_time.values
    print(f"  SSTA time range: {ssta_times[0]} → {ssta_times[-1]}")
    print(f"  SSTA len: {len(ssta_times)}")
    ds_ssta.close()


def main():
    print("=" * 70)
    print("  SLA Data Inspector")
    print(f"  Root: {SLA_ROOT}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if not os.path.isdir(SLA_ROOT):
        print(f"\n[FATAL] SLA root directory not found: {SLA_ROOT}")
        print("Run this script on the server that has access to SLA data.")
        sys.exit(1)

    results = []
    for subdir in SUBDIRS:
        print_header(f"Sub-directory: '{subdir}'")
        r = inspect_subdir(subdir)
        results.append(r)

    compare_grids(results)

    check_overlap_with_ssta()

    print_header("Summary & Recommendations")

    valid = [r for r in results if r is not None]
    if not valid:
        print("  No valid subdirectories found.")
        return

    # Determine which subdirs to use for train/val/test
    print(f"\n  Found {len(valid)} sub-directories with data.")

    yr_ranges = [(r['years'][0], r['years'][1], r['name']) for r in valid]
    yr_ranges.sort()

    print("\n  Timeline (sorted):")
    for y0, y1, name in yr_ranges:
        print(f"    {name:15s}: {y0} – {y1}")

    print("\n  Recommended mapping to existing SSTA split:")
    print("    SSTA train (2001-2022) → slatrain (align by year)")
    print("    SSTA val   (2023-2024) → slaval   (align by year)")
    print("    SSTA test  (2025)      → slatest  (align by year)")
    if any('21-25' in r['name'] for r in valid):
        print("    Note: 'sla 21-25' may overlap with other dirs — verify to avoid duplicates")

    print(f"\n  Done. ({datetime.now().strftime('%H:%M:%S')})")


if __name__ == "__main__":
    main()
