#!/usr/bin/env python
"""Preprocess daily SLA data for NW Pacific SSTA prediction.

Scans four SLA subdirectories, reads daily .nc files for 2001-2025,
crops to 10°N–50°N / 120°E–180°E, interpolates to ERA5 161×241 grid,
saves single SLA_cropped.nc.

Usage:
    python scripts/datasets/preprocess_sla.py
"""

import os
import re
import sys
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime, timedelta
from tqdm import tqdm

# ─── Paths ───
SLA_ROOT = "/home/lab/zhangxm/tzjzllllll/sladata"
OUTPUT_DIR = "/home/lab/zhangxm/gxy/Earthformer/datasets/SST-PREDICT/"
OUTPUT_FILE = "SLA_cropped.nc"

SUBDIRS = ["slatrain", "slaval", "slatest", "sla 21-25"]

# ─── Target grid (matching ERA5 SST_cropped.nc) ───
LAT_MIN, LAT_MAX = 10.0, 50.0
LON_MIN, LON_MAX = 120.0, 180.0
N_LAT, N_LON = 161, 241
TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT, dtype=np.float64)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON, dtype=np.float64)
# Pre-built mesh for interpolation (reused every day)
TARGET_PTS = np.stack(np.meshgrid(TARGET_LAT, TARGET_LON, indexing="ij"), axis=-1)
# Shape: (161, 241, 2)

START_DATE = "2001-01-01"
END_DATE = "2025-12-31"


# ──────────────────────────────────────────────────────────────────
# File collection
# ──────────────────────────────────────────────────────────────────

def collect_files():
    """Scan all four subdirs, return {date_str: file_path} mapping."""
    date_to_path = {}
    total = 0
    for subdir in SUBDIRS:
        subdir_path = os.path.join(SLA_ROOT, subdir)
        if not os.path.isdir(subdir_path):
            print(f"  [WARN] Not found: {subdir_path}")
            continue
        n = 0
        for yr_dir in sorted(os.listdir(subdir_path)):
            yr_path = os.path.join(subdir_path, yr_dir)
            if not os.path.isdir(yr_path) or not yr_dir.isdigit():
                continue
            for mo_dir in sorted(os.listdir(yr_path)):
                mo_path = os.path.join(yr_path, mo_dir)
                if not os.path.isdir(mo_path):
                    continue
                for fname in sorted(os.listdir(mo_path)):
                    # Pattern 1: YYYY_MM_DD.nc  (1993-2005, 2016-2025)
                    m = re.match(r"(\d{4})_(\d{2})_(\d{2})\.nc$", fname)
                    if m:
                        date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                        date_to_path[date_str] = os.path.join(mo_path, fname)
                        n += 1
                        continue
                    # Pattern 2: dt_global_allsat_phy_l4_YYYYMMDD_YYYYMMDD.nc  (2006-2015)
                    m = re.match(r"dt_global_allsat_phy_l4_(\d{4})(\d{2})(\d{2})_\d{8}\.nc$", fname)
                    if m:
                        date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                        date_to_path[date_str] = os.path.join(mo_path, fname)
                        n += 1
        print(f"  {subdir}: {n} files")
        total += n
    print(f"  Total: {len(date_to_path)} unique dates (raw count: {total})")
    return date_to_path


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    t0 = datetime.now()
    print("=" * 62)
    print("  SLA Preprocessing")
    print(f"  Source: {SLA_ROOT}")
    print(f"  Output: {os.path.join(OUTPUT_DIR, OUTPUT_FILE)}")
    print(f"  Grid:   {N_LAT}×{N_LON}  ({LAT_MIN}–{LAT_MAX}N, {LON_MIN}–{LON_MAX}E)")
    print(f"  Period:  {START_DATE} → {END_DATE}")
    print("=" * 62)

    # ── 1. Collect files ──
    print("\n[1/3] Scanning SLA directories ...")
    date_to_path = collect_files()

    # ── 2. Build date list ──
    dates = []
    d = datetime.strptime(START_DATE, "%Y-%m-%d")
    d_end = datetime.strptime(END_DATE, "%Y-%m-%d")
    while d <= d_end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    n_dates = len(dates)
    print(f"  Target date range: {dates[0]} → {dates[-1]}  ({n_dates} days)")

    # ── 3. Interpolate day-by-day ──
    print(f"\n[2/3] Interpolating {n_dates} days to {N_LAT}×{N_LON} grid ...")

    # Pre-load source grid once (all SLA files share the same 720×1440 grid)
    any_file = next(iter(date_to_path.values()))
    with xr.open_dataset(any_file) as ds_tpl:
        SRC_LAT = ds_tpl.latitude.values.astype(np.float64).copy()
        SRC_LON = ds_tpl.longitude.values.astype(np.float64).copy()
    print(f"  Source grid: {len(SRC_LAT)}×{len(SRC_LON)}  "
          f"lat=[{float(SRC_LAT[0]):.3f}, {float(SRC_LAT[-1]):.3f}], "
          f"lon=[{float(SRC_LON[0]):.3f}, {float(SRC_LON[-1]):.3f}]")

    sla_daily = []
    n_missing = 0
    n_error = 0
    errors = []
    missing_dates = []

    # Pre-build interpolator target points
    pbar = tqdm(dates, desc="  Interpolating", unit="day",
                bar_format="{l_bar}{bar:40}{r_bar}")

    for date_str in pbar:
        if date_str in date_to_path:
            try:
                with xr.open_dataset(date_to_path[date_str]) as ds:
                    sla = ds["sla"].values.astype(np.float64)
                if sla.ndim == 3:
                    sla = sla[0]

                interp = RegularGridInterpolator(
                    (SRC_LAT, SRC_LON), sla,
                    bounds_error=False, fill_value=np.nan)
                sla_daily.append(interp(TARGET_PTS).astype(np.float32))
            except Exception as exc:
                sla_daily.append(np.full((N_LAT, N_LON), np.nan, dtype=np.float32))
                n_error += 1
                errors.append(f"{date_str}: {exc}")
        else:
            sla_daily.append(np.full((N_LAT, N_LON), np.nan, dtype=np.float32))
            n_missing += 1
            missing_dates.append(date_str)

        # Update postfix every 100 days
        if len(sla_daily) % 100 == 0:
            found = len(sla_daily) - n_missing - n_error
            pbar.set_postfix(found=found, missing=n_missing, error=n_error)

    sla_data = np.stack(sla_daily, axis=0)  # (T, 161, 241)
    n_success = n_dates - n_missing - n_error
    print(f"\n  Done: {n_success}/{n_dates} loaded, "
          f"{n_missing} missing, {n_error} errors")

    if missing_dates:
        print(f"  Missing dates (first 10): {missing_dates[:10]}")
    if errors:
        print(f"  Errors (first 5): {errors[:5]}")

    # Quick stats
    ocean_mask = ~np.isnan(sla_data)
    n_ocean_per_day = ocean_mask.sum(axis=(1, 2))  # (T,)
    print(f"  Ocean pixels per day: {int(n_ocean_per_day.mean())} ± {int(n_ocean_per_day.std())}")
    print(f"  SLA range (ocean): [{float(np.nanmin(sla_data)):.4f}, "
          f"{float(np.nanmax(sla_data)):.4f}] m")
    print(f"  SLA mean (ocean): {float(np.nanmean(sla_data)):.4f} m")
    print(f"  SLA std  (ocean): {float(np.nanstd(sla_data)):.4f} m")

    # ── 4. Save ──
    print(f"\n[3/3] Saving to {OUTPUT_FILE} ...")
    time_coord = np.datetime64(START_DATE) + np.arange(n_dates).astype("timedelta64[D]")

    ds_out = xr.Dataset(
        {"sla": (["valid_time", "latitude", "longitude"], sla_data)},
        coords={
            "valid_time": time_coord,
            "latitude": TARGET_LAT.astype(np.float32),
            "longitude": TARGET_LON.astype(np.float32),
        },
    )
    ds_out["sla"].attrs["units"] = "m"
    ds_out["sla"].attrs["long_name"] = "Sea Level Anomaly"
    ds_out["latitude"].attrs["units"] = "degrees_north"
    ds_out["longitude"].attrs["units"] = "degrees_east"

    encoding = {"sla": {"dtype": "float32", "zlib": True, "complevel": 4}}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    ds_out.to_netcdf(out_path, encoding=encoding)
    ds_out.close()

    elapsed = (datetime.now() - t0).total_seconds()
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Saved:  {out_path}")
    print(f"  Size:   {size_mb:.1f} MB")
    print(f"  Time:   {elapsed:.0f}s  ({elapsed / 60:.1f} min)")
    print(f"\n{'=' * 62}")
    print("  Done!")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
