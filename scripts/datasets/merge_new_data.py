#!/usr/bin/env python
"""Merge 2023-2025 ERA5 data into the existing NW Pacific SSTA dataset.

Steps:
  1. Crop + flip lat + K→°C for new SST and Wind
  2. Compute SSTA using existing climatology (2001-2022)
  3. Concatenate old + new along time axis
  4. Clean up temp files

Usage:
    python scripts/datasets/merge_new_data.py
"""
import os
import sys
import numpy as np
import xarray as xr
from datetime import datetime

# ─── Paths ───
DATA_DIR = "/home/lab/zhangxm/gxy/Earthformer/datasets/SST-PREDICT"
SST_NEW = os.path.join(DATA_DIR, "SST23-25.nc")
WIND_NEW = os.path.join(DATA_DIR, "wind23-25.nc")
CLIM_PATH = os.path.join(DATA_DIR, "climatology.nc")
SSTA_OLD = os.path.join(DATA_DIR, "ssta.nc")
SST_OLD = os.path.join(DATA_DIR, "SST_cropped.nc")
WIND_OLD = os.path.join(DATA_DIR, "Wind_cropped.nc")
MASK_PATH = os.path.join(DATA_DIR, "mask.npy")

# Crop range (same as preprocess_nwp.py)
LAT_MIN, LAT_MAX = 10.0, 50.0
LON_MIN, LON_MAX = 120.0, 180.0


def print_step(msg):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")


def crop_and_flip(ds):
    ds = ds.sel(longitude=slice(LON_MIN, LON_MAX))
    lat_vals = ds.latitude.values
    if lat_vals[0] > lat_vals[-1]:
        ds = ds.isel(latitude=slice(None, None, -1))
    ds = ds.sel(latitude=slice(LAT_MIN, LAT_MAX))
    return ds


def backup_file(path):
    if os.path.exists(path):
        backup = path.replace('.nc', '_2001-2022_backup.nc')
        if not os.path.exists(backup):
            print_step(f"Backing up: {path} → {backup}")
            os.rename(path, backup)
        else:
            print_step(f"Backup already exists: {backup}")


def main():
    print("=" * 60)
    print("  Merge 2023-2025 Data")
    print("=" * 60)

    # ── Step 1: Crop + flip new SST ──
    print_step("Step 1/5: Cropping new SST (2023-2025) ...")
    ds_sst_new = xr.open_dataset(SST_NEW)
    ds_sst_new = crop_and_flip(ds_sst_new)
    sst_k = ds_sst_new['sst'].values
    sst_c = sst_k - 273.15
    ds_sst_new['sst'] = (('valid_time', 'latitude', 'longitude'), sst_c.astype(np.float32))
    ds_sst_new['sst'].attrs['units'] = 'degC'
    n_new = ds_sst_new.dims['valid_time']
    print(f"  New SST: {dict(ds_sst_new.dims)}, range: {np.nanmin(sst_c):.1f} ~ {np.nanmax(sst_c):.1f} °C")

    # ── Step 2: Crop + flip new Wind ──
    print_step("Step 2/5: Cropping new Wind (2023-2025) ...")
    ds_wind_new = xr.open_dataset(WIND_NEW)
    ds_wind_new = crop_and_flip(ds_wind_new)
    print(f"  New Wind: {dict(ds_wind_new.dims)}")

    # ── Step 3: SSTA = SST - Climatology ──
    print_step("Step 3/5: Computing SSTA for new data ...")
    ds_clim = xr.open_dataset(CLIM_PATH)
    clim = ds_clim['climatology']

    ssta_new = ds_sst_new['sst'].groupby('valid_time.dayofyear') - clim
    ssta_new = ssta_new.drop_vars('dayofyear')
    ssta_new.name = 'ssta'
    ssta_ds_new = ssta_new.to_dataset()
    ssta_ds_new['ssta'].attrs['units'] = 'degC'
    ssta_ds_new['ssta'].attrs['long_name'] = 'Sea Surface Temperature Anomaly'

    vals = ssta_new.values
    print(f"  New SSTA: {ssta_new.shape}, range: {np.nanmin(vals):.2f} ~ {np.nanmax(vals):.2f} °C")

    # ── Step 4: Concatenate ──
    print_step("Step 4/5: Concatenating old + new along time ...")

    # SST
    print("  Concatenating SST_cropped ...")
    ds_sst_old = xr.open_dataset(SST_OLD)
    ds_sst_merged = xr.concat([ds_sst_old, ds_sst_new], dim='valid_time')
    print(f"    Old: {dict(ds_sst_old.dims)}, New: {dict(ds_sst_new.dims)} → Merged: {dict(ds_sst_merged.dims)}")

    # Wind
    print("  Concatenating Wind_cropped ...")
    ds_wind_old = xr.open_dataset(WIND_OLD)
    ds_wind_merged = xr.concat([ds_wind_old, ds_wind_new], dim='valid_time')
    print(f"    Old: {dict(ds_wind_old.dims)}, New: {dict(ds_wind_new.dims)} → Merged: {dict(ds_wind_merged.dims)}")

    # SSTA
    print("  Concatenating ssta ...")
    ds_ssta_old = xr.open_dataset(SSTA_OLD)
    ds_ssta_merged = xr.concat([ds_ssta_old, ssta_ds_new], dim='valid_time')
    print(f"    Old: {dict(ds_ssta_old.dims)}, New: {dict(ssta_ds_new.dims)} → Merged: {dict(ds_ssta_merged.dims)}")

    # Verify time continuity
    t_old = ds_sst_old.valid_time.values[-1]
    t_new = ds_sst_new.valid_time.values[0]
    print(f"  Time gap: {t_old} → {t_new}")
    expected_next = t_old + np.timedelta64(1, 'D')
    if t_new == expected_next:
        print("  [OK] Time is continuous (1 day gap, expected for daily data)")
    else:
        diff = (t_new - t_old) / np.timedelta64(1, 'D')
        print(f"  [WARN] Time gap is {diff:.0f} days")

    ds_clim.close()
    ds_sst_old.close()
    ds_wind_old.close()
    ds_ssta_old.close()
    ds_sst_new.close()
    ds_wind_new.close()

    # ── Step 5: Backup + Save ──
    print_step("Step 5/5: Backing up old files + saving merged files ...")

    backup_file(SST_OLD)
    backup_file(WIND_OLD)
    backup_file(SSTA_OLD)

    encoding_sst = {'sst': {'dtype': 'float32', 'zlib': True, 'complevel': 4}}
    encoding_ssta = {'ssta': {'dtype': 'float32', 'zlib': True, 'complevel': 4}}
    encoding_wind = {'u10': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
                     'v10': {'dtype': 'float32', 'zlib': True, 'complevel': 4}}

    print(f"  Saving: {SST_OLD}")
    ds_sst_merged.to_netcdf(SST_OLD, encoding=encoding_sst)
    print(f"  Saving: {WIND_OLD}")
    ds_wind_merged.to_netcdf(WIND_OLD, encoding=encoding_wind)
    print(f"  Saving: {SSTA_OLD}")
    ds_ssta_merged.to_netcdf(SSTA_OLD, encoding=encoding_ssta)

    ds_sst_merged.close()
    ds_wind_merged.close()
    ds_ssta_merged.close()

    # ── Summary ──
    n_total = ds_sst_merged.dims['valid_time']
    print(f"\n{'=' * 60}")
    print(f"  Merge Complete!")
    print(f"  Old: 8035 days (2001-2022)")
    print(f"  New: {n_new} days (2023-2025)")
    print(f"  Total: {n_total} days ({n_total//365} years)")
    print(f"\n  Next step:")
    print(f"    Delete normalization_stats.npz so training recomputes stats")
    print(f"    Or just re-run training — it auto-detects new data range")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
