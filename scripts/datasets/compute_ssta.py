#!/usr/bin/env python
"""
Generate SSTA (Sea Surface Temperature Anomaly) from SST and Daily Climatology.

    SSTA(t, lat, lon) = SST(t, lat, lon) - Climatology(dayofyear(t), lat, lon)

Usage:
    python scripts/datasets/compute_ssta.py
"""
import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime

# ─── Paths ───
SST_PATH = "datasets/SST-PREDICT/SST_cropped.nc"
CLIM_PATH = "datasets/SST-PREDICT/climatology.nc"
MASK_PATH = "datasets/SST-PREDICT/mask.npy"
OUTPUT_DIR = "datasets/SST-PREDICT/"
SSTA_PATH = os.path.join(OUTPUT_DIR, "ssta.nc")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

LAT_MIN, LAT_MAX = 10.0, 50.0
LON_MIN, LON_MAX = 120.0, 180.0


def print_step(msg):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")


def plot_map(data, lon, lat, title, save_path, cmap='RdBu_r', vmin=None, vmax=None, symmetric=True):
    """Plot 2D map with auto-vmin/vmax if not specified."""
    if vmin is None and symmetric:
        vabs = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
        vmin, vmax = -vabs, vabs

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.pcolormesh(lon, lat, data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('SSTA (°C)')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print_step(f"  Plot saved: {save_path}")


def main():
    print("=" * 60)
    print("  SSTA Generation")
    print("=" * 60)

    os.makedirs(FIG_DIR, exist_ok=True)

    try:
        # ═══════════════════════════════════════════════════════
        # Step 1: Load data
        # ═══════════════════════════════════════════════════════
        print_step("Step 1/8: Loading SST and Climatology ...")

        ds_sst = xr.open_dataset(SST_PATH)
        ds_clim = xr.open_dataset(CLIM_PATH)

        sst = ds_sst['sst']           # (valid_time, lat, lon)
        clim = ds_clim['climatology'] # (dayofyear, lat, lon)

        print(f"\n  SST:")
        print(f"    Shape:      {sst.shape}")
        print(f"    Time range: {str(sst.valid_time.values[0])[:10]} → "
              f"{str(sst.valid_time.values[-1])[:10]}")
        print(f"    Min/Max:    {float(sst.min()):.2f} / {float(sst.max()):.2f} °C")
        print(f"    Mean ± Std: {float(sst.mean()):.2f} ± {float(sst.std()):.2f} °C")

        print(f"\n  Climatology:")
        print(f"    Shape:      {clim.shape}")
        print(f"    Day range:  {int(clim.dayofyear.min())} – {int(clim.dayofyear.max())}")
        print(f"    Min/Max:    {float(clim.min()):.2f} / {float(clim.max()):.2f} °C")
        print(f"    Mean ± Std: {float(clim.mean()):.2f} ± {float(clim.std()):.2f} °C")

        # ═══════════════════════════════════════════════════════
        # Step 2: Vectorized SSTA computation
        # ═══════════════════════════════════════════════════════
        print_step("Step 2/8: Computing SSTA = SST - Climatology(dayofyear) ...")

        # xarray's groupby + arithmetic handles the alignment automatically:
        #   sst.groupby('valid_time.dayofyear') gives groups indexed by doy
        #   subtracting clim aligns on the shared doy dimension
        ssta = sst.groupby('valid_time.dayofyear') - clim
        # Rename the grouped dim back to valid_time
        ssta = ssta.drop_vars('dayofyear')  # remove coord added by groupby

        # Give it a proper name
        ssta.name = 'ssta'

        # ═══════════════════════════════════════════════════════
        # Step 3: SSTA Statistics
        # ═══════════════════════════════════════════════════════
        print_step("Step 3/8: Computing SSTA statistics ...")

        vals = ssta.values
        valid_mask = np.isfinite(vals)
        n_valid = valid_mask.sum()
        n_nan = (~valid_mask).sum()
        n_total = vals.size

        print(f"\n  {'─' * 45}")
        print(f"  SSTA Statistics:")
        print(f"    Shape:    {ssta.shape}")
        print(f"    Min:      {np.nanmin(vals):.3f} °C")
        print(f"    Max:      {np.nanmax(vals):.3f} °C")
        print(f"    Mean:     {np.nanmean(vals):.4f} °C  (should be ~0)")
        print(f"    Std:      {np.nanstd(vals):.3f} °C")
        print(f"    Valid:    {n_valid:,} ({100*n_valid/n_total:.1f}%)")
        print(f"    NaN:      {n_nan:,} ({100*n_nan/n_total:.1f}%)  [land]")

        # Extreme anomaly check
        valid_vals = vals[valid_mask]
        n_gt5 = int((np.abs(valid_vals) > 5).sum())
        n_gt8 = int((np.abs(valid_vals) > 8).sum())
        print(f"    |SSTA| > 5°C: {n_gt5:,} ({100*n_gt5/len(valid_vals):.4f}%)")
        print(f"    |SSTA| > 8°C: {n_gt8:,} ({100*n_gt8/len(valid_vals):.4f}%)")
        if n_gt8 > 0:
            print(f"    [NOTE] >8°C anomalies are rare (<0.01% expected) — may indicate data issues")
        print(f"  {'─' * 45}")

        # ═══════════════════════════════════════════════════════
        # Step 4: Verification
        # ═══════════════════════════════════════════════════════
        print_step("Step 4/8: Verifying SSTA = SST - Climatology ...")

        rng = np.random.default_rng(42)
        n_time = sst.shape[0]
        check_indices = rng.integers(0, n_time, size=5)

        print(f"\n  {'─' * 65}")
        print(f"  {'Date':>12s}  {'SST_mean':>8s}  {'Clim_mean':>8s}  "
              f"{'SSTA_mean':>8s}  {'Residual':>8s}")
        print(f"  {'─' * 65}")
        all_ok = True
        for idx in check_indices:
            t = ssta.valid_time.values[idx]
            sst_day = sst.isel(valid_time=idx).values
            doy = int(ssta.valid_time.dt.dayofyear.values[idx])
            clim_day = clim.sel(dayofyear=doy).values
            ssta_day = ssta.isel(valid_time=idx).values

            sst_m = np.nanmean(sst_day)
            clim_m = np.nanmean(clim_day)
            ssta_m = np.nanmean(ssta_day)
            residual = sst_m - (clim_m + ssta_m)

            mark = " [OK]" if abs(residual) < 0.01 else " [WARN]"
            if abs(residual) >= 0.01:
                all_ok = False
            print(f"  {str(t)[:10]:>12s}  {sst_m:8.3f}  {clim_m:8.3f}  "
                  f"{ssta_m:8.3f}  {residual:+8.4f}{mark}")
        print(f"  {'─' * 65}")
        if all_ok:
            print(f"  All checks passed: SST = Climatology + SSTA (residual < 0.01)")
        else:
            print(f"  [WARN] Some residuals > 0.01")

        # ═══════════════════════════════════════════════════════
        # Step 5: Save ssta.nc
        # ═══════════════════════════════════════════════════════
        print_step(f"Step 5/8: Saving → {SSTA_PATH}")

        ssta_ds = ssta.to_dataset(name='ssta')
        ssta_ds['ssta'].attrs['units'] = 'degC'
        ssta_ds['ssta'].attrs['long_name'] = 'Sea Surface Temperature Anomaly'
        ssta_ds['ssta'].attrs['description'] = (
            'SSTA = SST - Daily Climatology (2001-2022). '
            'Climatology computed as 22-year mean per calendar day. '
            'NaN values denote land grid points.'
        )
        encoding = {'ssta': {'dtype': 'float32', 'zlib': True, 'complevel': 4}}
        ssta_ds.to_netcdf(SSTA_PATH, encoding=encoding)
        print(f"  Saved: {SSTA_PATH}")

        # ═══════════════════════════════════════════════════════
        # Step 6: Visualization
        # ═══════════════════════════════════════════════════════
        print_step("Step 6/8: Generating plots ...")

        lon = ds_sst.longitude.values
        lat = ds_sst.latitude.values

        # 6a: Time-mean SSTA (should be ~0 everywhere)
        ssta_mean = ssta.mean(dim='valid_time', skipna=True).values
        plot_map(ssta_mean, lon, lat,
                 f'Time-Mean SSTA ({float(np.nanmean(ssta_mean)):.4f} °C, should be ~0)',
                 os.path.join(FIG_DIR, "ssta_mean.png"),
                 cmap='RdBu_r', vmin=-0.5, vmax=0.5)

        # 6b: Time-std SSTA (variability hot-spots)
        ssta_std = ssta.std(dim='valid_time', skipna=True).values
        plot_map(ssta_std, lon, lat,
                 f'SSTA Standard Deviation ({float(np.nanmean(ssta_std)):.2f} °C mean)',
                 os.path.join(FIG_DIR, "ssta_std.png"),
                 cmap='OrRd', vmin=0, vmax=3, symmetric=False)

        # 6c: Random day snapshot
        rand_idx = rng.integers(0, n_time)
        ssta_day = ssta.isel(valid_time=rand_idx).values
        day_str = str(ssta.valid_time.values[rand_idx])[:10]
        plot_map(ssta_day, lon, lat,
                 f'SSTA Snapshot — {day_str}',
                 os.path.join(FIG_DIR, "sample_ssta_day.png"),
                 cmap='RdBu_r', vmin=-3, vmax=3)

        # ═══════════════════════════════════════════════════════
        # Step 7: Mask statistics
        # ═══════════════════════════════════════════════════════
        print_step("Step 7/8: Checking mask overlap ...")
        if os.path.exists(MASK_PATH):
            mask = np.load(MASK_PATH)
            ssta_t0 = ssta.isel(valid_time=0).values
            ocean_in_ssta = np.isfinite(ssta_t0).sum()
            land_in_ssta = (~np.isfinite(ssta_t0)).sum()
            print(f"  SSTA at t=0: ocean={ocean_in_ssta}, land( NaN)={land_in_ssta}")
            print(f"  Mask:         ocean={mask.sum()}, land={mask.size - mask.sum()}")
            print(f"  Consistency:  {'OK' if ocean_in_ssta == mask.sum() else 'CHECK'}")

        # ═══════════════════════════════════════════════════════
        # Step 8: Cleanup & report
        # ═══════════════════════════════════════════════════════
        ds_sst.close()
        ds_clim.close()

        print(f"\n{'=' * 60}")
        print(f"  Done!")
        print(f"    {SSTA_PATH}  ({(os.path.getsize(SSTA_PATH)/1e6):.1f} MB)")
        print(f"    {FIG_DIR}/ssta_mean.png")
        print(f"    {FIG_DIR}/ssta_std.png")
        print(f"    {FIG_DIR}/sample_ssta_day.png")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\n  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Explanation ──
    print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║               What is SSTA & Why Use It?                     ║
  ╚══════════════════════════════════════════════════════════════╝

  1. Definition
  ────────────
      SST(t, lat, lon) = Climatology(doy, lat, lon) + SSTA(t, lat, lon)
       └─ total signal ─┘  └── deterministic ──┘   └─ anomaly ─┘

  Climatology captures the predictable seasonal cycle — warm summers,
  cold winters, monsoonal cooling. SSTA captures everything else:
  eddies, fronts, ENSO teleconnections, interannual variability.

  2. Why SSTA is more stationary than absolute SST
  ─────────────────────────────────────────────────
      SSTA ~ N(0, σ²)          zero mean, bounded variance
      SST  ~ N(C(doy), σ²)     time-varying mean, trend

  An Earthformer trained on SSTA learns to map:

      [SSTA(t-11), ..., SSTA(t)] → [SSTA(t+1), ..., SSTA(t+N)]

  The target distribution has a time-invariant mean (~0) and
  variance, making gradient-based optimization well-posed.
  Absolute SST has a strong seasonal signal that dominates the
  loss — the model wastes capacity memorizing "summer ≈ 28°C".

  3. Why Earthformer is well-suited
  ────────────────────────────────
  - Cuboid attention naturally separates spatial scales:
    shallow layers capture local eddies (~100km),
    deep layers capture basin-scale modes (~1000km).
  - Multi-channel input (SSTA, u10, v10) allows the model to
    learn air-sea coupling physics from data.
  - Non-autoregressive decoding prevents error accumulation
    across lead times.

  4. Post-processing
  ─────────────────
  After model predicts SSTA, recover SST:
      SST_pred(t+k) = SSTA_pred(t+k) + Climatology(doy(t+k), lat, lon)
""")


if __name__ == "__main__":
    main()
