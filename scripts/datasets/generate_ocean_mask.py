#!/usr/bin/env python
"""
Generate ocean-land mask from SST_cropped.nc for NW Pacific SST prediction.

Principle:
  SST is only defined over ocean. Land grid points contain NaN.
  By checking NaN at the first timestep, we get a static land-sea mask.
  This mask is valid for the entire time series because:
    - Coastlines are invariant over 22 years at 0.25° resolution.
    - Sea level change (<1m/century) is far below 0.25° (~28km) detectability.
    - ERA5 uses a fixed land-sea grid for all timesteps.

Usage in Earthformer training:
  - Mask out land pixels from the loss function so the model only learns over ocean.
  - Apply consistent NaN/zero-filling to all variables (SST, u10, v10).
  - During inference, multiply output by mask to clean land artifacts.

Usage:
    python scripts/datasets/generate_ocean_mask.py
"""
import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime

# ─── Paths ───
SST_PATH = "/home/lab/zhangxm/gxy/Earthformer/datasets/SST-PREDICT/SST_cropped.nc"
OUTPUT_DIR = "/home/lab/zhangxm/gxy/Earthformer/datasets/SST-PREDICT/"


def print_step(msg):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")


def main():
    print("=" * 60)
    print("  Ocean-Land Mask Generation")
    print("=" * 60)

    if not os.path.exists(SST_PATH):
        print(f"  [ERROR] SST_cropped.nc not found at: {SST_PATH}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load data ──
    print_step("Loading SST_cropped.nc ...")
    ds = xr.open_dataset(SST_PATH)
    sst_var = ds['sst']
    print(f"  Shape:  {sst_var.shape}")
    print(f"  Dims:   {list(sst_var.dims)}")

    # ── Step 2: Generate mask from first timestep ──
    print_step("Generating mask from valid_time=0 ...")
    sst_t0 = sst_var.isel(valid_time=0).values  # (lat, lon)

    # ocean=1 where SST is not NaN, land=0 where SST is NaN
    ocean_mask = (~np.isnan(sst_t0)).astype(np.uint8)
    n_ocean = int(ocean_mask.sum())
    n_land = int(ocean_mask.size - n_ocean)
    pct_ocean = 100 * n_ocean / ocean_mask.size
    pct_land = 100 * n_land / ocean_mask.size

    print(f"\n  {'─' * 40}")
    print(f"  Mask Statistics:")
    print(f"    Shape:        {ocean_mask.shape}")
    print(f"    Ocean cells:  {n_ocean:,}  ({pct_ocean:.1f}%)")
    print(f"    Land cells:   {n_land:,}  ({pct_land:.1f}%)")
    print(f"    Total:        {ocean_mask.size:,}")
    print(f"  {'─' * 40}")

    # ── Step 3: Verify mask consistency across timesteps ──
    print_step("Verifying mask consistency across time ...")
    n_time = sst_var.shape[0]
    # Random sample of timesteps (first, middle, last + random)
    check_indices = [0,
                     n_time // 4,
                     n_time // 2,
                     3 * n_time // 4,
                     n_time - 1]
    # Add 5 random indices
    rng = np.random.default_rng(42)
    check_indices += list(rng.integers(1, n_time - 1, size=5))
    check_indices = sorted(set(check_indices))

    all_match = True
    for idx in check_indices:
        nan_map = np.isnan(sst_var.isel(valid_time=idx).values)
        ref_nan_map = (ocean_mask == 0)
        match = np.array_equal(nan_map, ref_nan_map)
        if not match:
            n_diff = int((nan_map != ref_nan_map).sum())
            all_match = False
            print(f"    [MISMATCH] timestep {idx}: {n_diff} cells differ")
        else:
            print(f"    [OK] timestep {idx} matches")

    if all_match:
        print(f"\n  All {len(check_indices)} sampled timesteps consistent — mask is valid for entire series.")
    else:
        print(f"\n  [WARN] Some timesteps differ from reference mask!")

    # ── Step 4: Save mask.npy ──
    mask_npy_path = os.path.join(OUTPUT_DIR, "mask.npy")
    print_step(f"Saving mask.npy → {mask_npy_path}")
    np.save(mask_npy_path, ocean_mask)

    # ── Step 5: Save mask.nc (with coords) ──
    mask_nc_path = os.path.join(OUTPUT_DIR, "mask.nc")
    print_step(f"Saving mask.nc → {mask_nc_path}")
    mask_ds = xr.Dataset(
        {'mask': (['latitude', 'longitude'], ocean_mask)},
        coords={
            'latitude': ds['latitude'].values,
            'longitude': ds['longitude'].values,
        }
    )
    mask_ds['mask'].attrs['units'] = '1'
    mask_ds['mask'].attrs['long_name'] = 'Ocean-Land Mask (1=ocean, 0=land)'
    mask_ds['mask'].attrs['description'] = (
        f"Generated from {SST_PATH} at valid_time=0. "
        f"Ocean={n_ocean}({pct_ocean:.1f}%), Land={n_land}({pct_land:.1f}%)"
    )
    mask_ds.to_netcdf(mask_nc_path)

    # ── Step 6: Plot ──
    mask_png_path = os.path.join(OUTPUT_DIR, "mask.png")
    print_step(f"Plotting mask → {mask_png_path}")

    lat_vals = ds.latitude.values
    lon_vals = ds.longitude.values

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.pcolormesh(lon_vals, lat_vals, ocean_mask,
                       cmap='Blues', shading='auto',
                       vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1], shrink=0.7)
    cbar.ax.set_yticklabels(['Land', 'Ocean'])
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title(f'NW Pacific Ocean-Land Mask\n'
                 f'{LAT_MIN}°N–{LAT_MAX}°N, {LON_MIN}°E–{LON_MAX}°E, '
                 f'{pct_ocean:.1f}% ocean')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(mask_png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {mask_png_path}")

    ds.close()

    print(f"\n{'=' * 60}")
    print(f"  Done! Generated files:")
    print(f"    {mask_npy_path}  ({os.path.getsize(mask_npy_path)/1024:.0f} KB)")
    print(f"    {mask_nc_path}  ({os.path.getsize(mask_nc_path)/1e6:.1f} MB)")
    print(f"    {mask_png_path}  ({os.path.getsize(mask_png_path)/1e6:.1f} MB)")
    print(f"{'=' * 60}")

    # ── Explanation ──
    print(f"""
  Why mask from only the first timestep?
  ─────────────────────────────────────
  1. Coastlines are invariant over 22 years at 0.25° (~28km) resolution.
  2. ERA5 uses a fixed land-sea grid — the NaN pattern is identical at all
     timesteps, as verified above ({'all' if all_match else 'some'} sampled indices matched).
  3. Using t=0 avoids unnecessary computation over 8035 timesteps.

  Typical usage in Earthformer training:
  ─────────────────────────────────────
  Loss masking:
      loss = (mask * (pred - target) ** 2).sum() / mask.sum()
  This ensures the model only learns over ocean pixels, ignoring land.

  Applying mask to all variables:
  ─────────────────────────────────────
  # 1. Before normalization: set land to 0
  sst  = np.where(mask, sst,  0.0)
  u10  = np.where(mask, u10,  0.0)
  v10  = np.where(mask, v10,  0.0)

  # 2. During training, apply mask to loss
  loss = ((pred - target) * mask[None, :, :, None]) ** 2
  loss = loss.sum() / mask.sum()

  # 3. During inference: clean land artifacts
  pred = pred * mask[None, :, :, None]
""")


if __name__ == "__main__":
    LAT_MIN, LAT_MAX = 10.0, 50.0
    LON_MIN, LON_MAX = 120.0, 180.0
    main()
