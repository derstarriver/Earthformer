#!/usr/bin/env python
"""
Compute Daily Climatology from ERA5 SST for NW Pacific.

Daily Climatology = average SST for each calendar day (Jan 1 .. Dec 31)
over a multi-year reference period (2001–2022).

Usage:
    python scripts/datasets/compute_climatology.py
"""
import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# ─── Paths ───
SST_PATH = "datasets/SST-PREDICT/SST_cropped.nc"
MASK_PATH = "datasets/SST-PREDICT/mask.npy"
OUTPUT_DIR = "datasets/SST-PREDICT/"
CLIM_PATH = os.path.join(OUTPUT_DIR, "climatology.nc")


def print_step(msg):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")


def plot_map(data, lon, lat, title, save_path, cmap='RdYlBu_r', vmin=None, vmax=None):
    """Plot a 2D map with cartopy coastline overlay."""
    try:
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(
            central_longitude=(lon[0] + lon[-1]) / 2))
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())

        im = ax.pcolormesh(lon, lat, data, cmap=cmap, shading='auto',
                           vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('SST (°C)')
        ax.set_title(title, fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print_step(f"  Plot saved: {save_path}")
    except Exception as e:
        # Fallback without cartopy
        print(f"  [WARN] cartopy plot failed ({e}), using simple plot ...")
        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, shading='auto',
                           vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label('SST (°C)')
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print_step(f"  Plot saved (no cartopy): {save_path}")


def main():
    print("=" * 60)
    print("  Daily Climatology Computation")
    print("=" * 60)

    if not os.path.exists(SST_PATH):
        print(f"  [ERROR] SST_cropped.nc not found: {SST_PATH}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # ── Step 1: Load ──
        print_step("Step 1/5: Loading SST_cropped.nc ...")
        ds = xr.open_dataset(SST_PATH)
        sst = ds['sst']
        print(f"  Shape: {sst.shape}  ({list(sst.dims)})")
        print(f"  Range: {np.nanmin(sst.values):.2f} ~ {np.nanmax(sst.values):.2f} °C")

        lat = ds.latitude.values
        lon = ds.longitude.values

        # ── Step 2: Compute daily climatology ──
        print_step("Step 2/5: Computing daily climatology (groupby dayofyear + mean) ...")

        # Extract day-of-year (1..366) from valid_time coordinate
        doy = sst.valid_time.dt.dayofyear

        # Group by dayofyear and compute mean, skipping NaN
        climatology = sst.groupby(doy).mean(dim='valid_time', skipna=True)
        # Shape: (dayofyear, latitude, longitude)

        print(f"  Climatology shape: {climatology.shape}")
        print(f"  Day range: {int(climatology.dayofyear.min())} ~ {int(climatology.dayofyear.max())}")
        expected_days = 366  # leap year full coverage
        actual_days = len(climatology.dayofyear)
        if actual_days == expected_days:
            print(f"  All {expected_days} calendar days present (including Feb 29)")
        else:
            print(f"  [WARN] {actual_days} days found (expected {expected_days})")

        # ── Step 3: Statistics ──
        print_step("Step 3/5: Computing statistics ...")
        clim_vals = climatology.values
        valid_mask = np.isfinite(clim_vals)
        n_valid = valid_mask.sum()
        n_nan = (~valid_mask).sum()
        n_total = clim_vals.size

        print(f"\n  {'─' * 45}")
        print(f"  Climatology Statistics:")
        print(f"    Shape:    {climatology.shape}")
        print(f"    Min:      {np.nanmin(clim_vals):.2f} °C")
        print(f"    Max:      {np.nanmax(clim_vals):.2f} °C")
        print(f"    Mean:     {np.nanmean(clim_vals):.2f} °C")
        print(f"    Std:      {np.nanstd(clim_vals):.2f} °C")
        print(f"    Valid:    {n_valid:,} ({100*n_valid/n_total:.1f}%)")
        print(f"    NaN:      {n_nan:,} ({100*n_nan/n_total:.1f}%)  [land]")
        print(f"  {'─' * 45}")

        # Annual mean
        annual_mean = climatology.mean(dim='dayofyear')
        print(f"  Annual mean SST range: {float(annual_mean.min()):.2f} ~ {float(annual_mean.max()):.2f} °C")

        # ── Step 4: Save ──
        print_step(f"Step 4/5: Saving → {CLIM_PATH}")
        # Rename to more descriptive variable name
        clim_ds = climatology.to_dataset(name='climatology')
        clim_ds['climatology'].attrs['units'] = 'degC'
        clim_ds['climatology'].attrs['long_name'] = 'Daily SST Climatology (2001-2022)'
        clim_ds['climatology'].attrs['description'] = (
            'Daily climatology computed as mean SST for each calendar day '
            'over 2001-2022 from ERA5. 366 days (includes Feb 29 for leap years). '
            'NaN values denote land grid points.'
        )
        clim_ds.to_netcdf(CLIM_PATH)
        print(f"  Saved: {CLIM_PATH}")

        # ── Step 5: Plot ──
        print_step("Step 5/5: Generating plots ...")

        plot_map(annual_mean.values, lon, lat,
                 f'Annual Mean SST Climatology ({float(annual_mean.min()):.1f} ~ {float(annual_mean.max()):.1f} °C)',
                 os.path.join(OUTPUT_DIR, "annual_mean_climatology.png"),
                 cmap='RdYlBu_r', vmin=5, vmax=32)

        # Summer: July 15 = dayofyear 196 (non-leap) or 197 (leap)
        summer_day = 196
        if summer_day in climatology.dayofyear.values:
            summer = climatology.sel(dayofyear=summer_day)
            plot_map(summer.values, lon, lat,
                     f'Summer SST Climatology — July 15 ({float(summer.min()):.1f} ~ {float(summer.max()):.1f} °C)',
                     os.path.join(OUTPUT_DIR, "summer_climatology.png"),
                     cmap='RdYlBu_r', vmin=5, vmax=32)
        else:
            print(f"  [WARN] dayofyear {summer_day} not found, skipping summer plot")

        # Winter: Jan 15 = dayofyear 15
        winter_day = 15
        if winter_day in climatology.dayofyear.values:
            winter = climatology.sel(dayofyear=winter_day)
            plot_map(winter.values, lon, lat,
                     f'Winter SST Climatology — Jan 15 ({float(winter.min()):.1f} ~ {float(winter.max()):.1f} °C)',
                     os.path.join(OUTPUT_DIR, "winter_climatology.png"),
                     cmap='RdYlBu_r', vmin=5, vmax=32)
        else:
            print(f"  [WARN] dayofyear {winter_day} not found, skipping winter plot")

        ds.close()

        print(f"\n{'=' * 60}")
        print(f"  Done! Output:")
        print(f"    {CLIM_PATH}")
        print(f"    {OUTPUT_DIR}/annual_mean_climatology.png")
        print(f"    {OUTPUT_DIR}/summer_climatology.png")
        print(f"    {OUTPUT_DIR}/winter_climatology.png")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\n  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Explanation ──
    print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║                    Background & Rationale                     ║
  ╚══════════════════════════════════════════════════════════════╝

  1. What is Daily Climatology?
  ─────────────────────────────
  SST(t, lat, lon) = Climatology(doy, lat, lon) + Anomaly(t, lat, lon)
                       └─── deterministic ───┘   └── stochastic ──┘

  The daily climatology C(doy) is the expected SST for calendar day
  "doy" (1..366), computed by averaging all Jan 1 values across
  22 years, all Jan 2 values, etc.

  It captures the deterministic seasonal cycle — solar heating,
  monsoon-driven upwelling, boundary currents — that repeats every year.

  2. Why predict SSTA (anomaly) instead of absolute SST?
  ─────────────────────────────────────────────────────
  Two reasons:

  (a) Signal-to-noise ratio.
      SST  = [seasonal cycle ~5-15°C] + [interannual anomaly ~0.5-2°C]
      SSTA = [interannual anomaly ~0.5-2°C] alone

      The seasonal cycle dominates the variance (~90%), so a model
      trained on absolute SST spends most of its capacity memorizing
      "summer is warm, winter is cold" instead of learning predictable
      dynamics.

  (b) Stationarity.
      SSTA has a near-zero mean, bounded variance, and no long-term
      trend over 22 years. This makes the learning problem well-posed
      for gradient-based optimization. Absolute SST has a trend (global
      warming ~0.15°C/decade) that the model must extrapolate.

  3. Why Earthformer is well-suited for SSTA prediction?
  ─────────────────────────────────────────────────────
  - **Spatiotemporal attention**: Cuboid attention decomposes (T,H,W)
    into sub-cubes, capturing both local eddy dynamics and basin-scale
    teleconnections simultaneously.
  - **Non-autoregressive decoding**: All lead times predicted in one
    shot, avoiding error accumulation from recursive rollout.
  - **Flexible receptive field**: Axial/dilated strategies can be
    tuned per-layer — shallow layers capture sub-monthly eddies,
    deep layers capture seasonal-to-interannual modes.

  Typical workflow:
    raw_sst - climatology = ssta
    ssta, u10, v10 → Earthformer → predicted_ssta
    predicted_sst = predicted_ssta + climatology

  Land mask is applied to SSTA (set land SSTA = 0, since it should
  always be approximately zero over land after removing climatology).
""")


if __name__ == "__main__":
    main()
