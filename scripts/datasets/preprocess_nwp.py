#!/usr/bin/env python
"""Preliminary NW Pacific data preprocessing — crop + flip lat + SST unit conversion.

Input:
    datasets/SST-PREDICT/SST.nc       (8035, 321, 561) daily SST [K]
    datasets/SST-PREDICT/wind01-22.nc (8035, 321, 561) daily u10, v10 [m/s]

Processing:
    1. Crop to 10N–50N, 120E–180E
    2. Flip latitude to ascending
    3. SST: Kelvin → Celsius

Output:
    datasets/SST-PREDICT/SST_cropped.nc
    datasets/SST-PREDICT/Wind_cropped.nc

Usage:
    python scripts/datasets/preprocess_nwp.py
"""
import os
import argparse
import numpy as np
import xarray as xr
from datetime import datetime

# ─── Paths ───
SST_PATH = "/home/lab/zhangxm/gxy/Earthformer/datasets/SST-PREDICT/SST.nc"
WIND_PATH = "/home/lab/zhangxm/gxy/Earthformer/datasets/SST-PREDICT/wind01-22.nc"
OUTPUT_DIR = "/home/lab/zhangxm/gxy/Earthformer/datasets/SST-PREDICT/"

# ─── Crop range ───
LAT_MIN, LAT_MAX = 10.0, 50.0    # 10N–50N
LON_MIN, LON_MAX = 120.0, 180.0  # 120E–180E


def crop_and_flip(ds):
    """Crop lon → flip lat to ascending → crop lat."""
    # Crop longitude
    ds = ds.sel(longitude=slice(LON_MIN, LON_MAX))

    # Flip latitude to ascending if currently descending
    lat_vals = ds.latitude.values
    if lat_vals[0] > lat_vals[-1]:
        ds = ds.isel(latitude=slice(None, None, -1))

    # Crop latitude
    ds = ds.sel(latitude=slice(LAT_MIN, LAT_MAX))

    return ds


def process_sst(input_path, output_path):
    """Load, crop, flip, K→°C, save SST."""
    print(f"\n{'='*60}")
    print(f"  Processing SST")
    print(f"{'='*60}")
    print(f"  Input:  {input_path}")

    ds = xr.open_dataset(input_path)
    print(f"  Original: {dict(ds.dims)}")

    t0 = datetime.now()

    # Crop + flip
    ds = crop_and_flip(ds)
    print(f"  After crop+flip: {dict(ds.dims)}")

    # SST K → °C
    sst_raw = ds['sst'].values
    print(f"  SST before: min={np.nanmin(sst_raw):.1f}K, max={np.nanmax(sst_raw):.1f}K")

    sst_c = sst_raw - 273.15
    ds['sst'] = (('valid_time', 'latitude', 'longitude'), sst_c.astype(np.float32))
    ds['sst'].attrs['units'] = 'degC'
    ds['sst'].attrs['long_name'] = 'Sea Surface Temperature'
    print(f"  SST after:  min={np.nanmin(sst_c):.1f}°C, max={np.nanmax(sst_c):.1f}°C")

    # Verify
    lat_vals = ds.latitude.values
    lon_vals = ds.longitude.values
    print(f"  Lat range:  {lat_vals[0]:.2f} → {lat_vals[-1]:.2f} ({len(lat_vals)} pts, {'ascending' if lat_vals[1] > lat_vals[0] else 'descending'})")
    print(f"  Lon range:  {lon_vals[0]:.2f} → {lon_vals[-1]:.2f} ({len(lon_vals)} pts)")

    # Save
    encoding = {'sst': {'dtype': 'float32', 'zlib': True, 'complevel': 4}}
    ds.to_netcdf(output_path, encoding=encoding)
    ds.close()

    elapsed = (datetime.now() - t0).total_seconds()
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Saved: {output_path} ({size_mb:.1f} MB, {elapsed:.0f}s)")


def process_wind(input_path, output_path):
    """Load, crop, flip, save Wind."""
    print(f"\n{'='*60}")
    print(f"  Processing Wind")
    print(f"{'='*60}")
    print(f"  Input:  {input_path}")

    ds = xr.open_dataset(input_path)
    print(f"  Original: {dict(ds.dims)}")

    t0 = datetime.now()

    # Crop + flip
    ds = crop_and_flip(ds)
    print(f"  After crop+flip: {dict(ds.dims)}")

    # Verify wind range
    for vn in ['u10', 'v10']:
        vals = ds[vn].values
        print(f"  {vn}: min={np.nanmin(vals):.2f}, max={np.nanmax(vals):.2f} m/s")

    lat_vals = ds.latitude.values
    lon_vals = ds.longitude.values
    print(f"  Lat range:  {lat_vals[0]:.2f} → {lat_vals[-1]:.2f} ({len(lat_vals)} pts)")
    print(f"  Lon range:  {lon_vals[0]:.2f} → {lon_vals[-1]:.2f} ({len(lon_vals)} pts)")

    # Save
    encoding = {'u10': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
                'v10': {'dtype': 'float32', 'zlib': True, 'complevel': 4}}
    ds.to_netcdf(output_path, encoding=encoding)
    ds.close()

    elapsed = (datetime.now() - t0).total_seconds()
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Saved: {output_path} ({size_mb:.1f} MB, {elapsed:.0f}s)")


def main():
    parser = argparse.ArgumentParser(description="Crop + flip NW Pacific SST/Wind data")
    parser.add_argument('--sst_in', type=str, default=SST_PATH)
    parser.add_argument('--wind_in', type=str, default=WIND_PATH)
    parser.add_argument('--out_dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sst_out = os.path.join(args.out_dir, "SST_cropped.nc")
    wind_out = os.path.join(args.out_dir, "Wind_cropped.nc")

    print("=" * 60)
    print("  NW Pacific Data — Preliminary Processing")
    print(f"  Crop:  {LAT_MIN}N–{LAT_MAX}N, {LON_MIN}E–{LON_MAX}E")
    print(f"  SST:   Kelvin → Celsius")
    print(f"  Wind:  unchanged (m/s)")
    print(f"  Time:  daily (no aggregation)")
    print("=" * 60)

    process_sst(args.sst_in, sst_out)
    process_wind(args.wind_in, wind_out)

    print(f"\n{'='*60}")
    print(f"  Done! Output files:")
    print(f"    {sst_out}")
    print(f"    {wind_out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
