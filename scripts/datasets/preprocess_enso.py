#!/usr/bin/env python
"""Standalone ENSO data preprocessing — run once before training.

Usage:
    python scripts/datasets/preprocess_enso.py --data_dir ./datasets/enso_multivar/

Reads CMIP_train.nc + CMIP_label.nc + SODA_train.nc + SODA_label.nc from data_dir,
transforms/normalizes/splits, and saves .npz cache files for instant training startup.

Output files in data_dir:
    .cache_cmip_all.npz   — processed CMIP6+CMIP5 data & labels
    .cache_cmip_meta.pkl  — variable stats & coordinate arrays
    .cache_soda.npz       — processed SODA data & labels
    .cache_soda_meta.pkl  — coordinate arrays
"""
import os
import sys
import pickle
import argparse
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm


# ─── Core processing functions (same as dataloader) ───

NINO_WINDOW_T = 3
DEFAULT_VARS = ['sst', 't300', 'ua', 'va']


def fold(data, size=36, stride=12):
    assert size % stride == 0
    times = size // stride
    remain = (data.shape[0] - 1) % times
    if remain > 0:
        ls = list(data[::times]) + [data[-1, -(remain * stride):]]
        outdata = np.concatenate(ls, axis=0)
    else:
        outdata = np.concatenate(data[::times], axis=0)
    assert outdata.shape[0] == size * ((data.shape[0] - 1) // times + 1) + remain * stride
    return outdata


def data_transform(data, num_years_per_model):
    length = data.shape[0]
    assert length % num_years_per_model == 0
    num_models = length // num_years_per_model
    outdata = np.stack(np.split(data, length / num_years_per_model, axis=0), axis=-1)
    outdata = fold(outdata, size=36, stride=12)
    assert outdata.shape[-1] == num_models
    assert not np.any(np.isnan(outdata))
    return outdata


def cat_over_last_dim(data):
    return np.concatenate(np.moveaxis(data, -1, 0), axis=0)


def find_nino_indices(lat_vals, lon_vals):
    lat_min, lat_max = -5.0, 5.0
    lon_min, lon_max = 190.0, 240.0
    lat_idx = np.where((lat_vals >= lat_min) & (lat_vals <= lat_max))[0]
    lon_idx = np.where((lon_vals >= lon_min) & (lon_vals <= lon_max))[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(f"Nino 3.4 region not found in data!")
    lat_slice = slice(lat_idx[0], lat_idx[-1] + 1)
    lon_slice = slice(lon_idx[0], lon_idx[-1] + 1)
    return lat_slice, lon_slice


def normalize(data, var_stats, var_names):
    for i, vn in enumerate(var_names):
        vmin, vmax = var_stats[vn]
        vrange = max(vmax - vmin, 1e-8)
        data[..., i] = (data[..., i] - vmin) / vrange


# ─── CMIP processing ───

def process_cmip(data_dir, cmip6_cutoff, cmip6_ypm, cmip5_ypm):
    """Process CMIP raw nc → normalized numpy arrays."""
    data_dir = Path(data_dir)
    print("=" * 60)
    print("Step 1/2: Processing CMIP data")
    print("=" * 60)

    ds_train = xr.open_dataset(data_dir / 'CMIP_train.nc')
    ds_label = xr.open_dataset(data_dir / 'CMIP_label.nc')

    if 'year' in ds_train.dims and 'month' in ds_train.dims:
        ds_train = ds_train.transpose('year', 'month', 'lat', 'lon')
    if 'year' in ds_label.dims and 'month' in ds_label.dims:
        ds_label = ds_label.transpose('year', 'month')

    print(f"  Raw shape: {dict(ds_train.dims)}")
    print(f"  Variables: {list(ds_train.data_vars.keys())}")

    # Select Pacific longitudes
    lon_mask = np.logical_and(ds_train.lon.values >= 95, ds_train.lon.values <= 330)
    ds_train = ds_train.sel(lon=lon_mask)
    print(f"  After lon filter (95E–330E): {dict(ds_train.dims)}")

    available_vars = [v for v in DEFAULT_VARS if v in ds_train.data_vars]
    if len(available_vars) < 4:
        print(f"  WARNING: Only {len(available_vars)}/4 vars found: {available_vars}")

    # Compute per-variable statistics
    var_stats = {}
    for vn in available_vars:
        vals = ds_train[vn].values
        var_stats[vn] = (float(np.nanmin(vals)), float(np.nanmax(vals)))
    print(f"  Variable stats (min, max):")
    for vn in available_vars:
        print(f"    {vn}: ({var_stats[vn][0]:.4f}, {var_stats[vn][1]:.4f})")

    # Transform each variable
    print("  Transforming CMIP6...")
    cmip6_list = []
    for vn in tqdm(available_vars, desc="  CMIP6 vars", unit="var"):
        vals = np.nan_to_num(ds_train[vn].values, nan=0.0)
        cmip6_list.append(cat_over_last_dim(
            data_transform(vals[:cmip6_cutoff], cmip6_ypm)))
    cmip6_data = np.stack(cmip6_list, axis=-1).astype(np.float32)

    print("  Transforming CMIP5...")
    cmip5_list = []
    for vn in tqdm(available_vars, desc="  CMIP5 vars", unit="var"):
        vals = np.nan_to_num(ds_train[vn].values, nan=0.0)
        cmip5_list.append(cat_over_last_dim(
            data_transform(vals[cmip6_cutoff:], cmip5_ypm)))
    cmip5_data = np.stack(cmip5_list, axis=-1).astype(np.float32)

    # Normalize
    print("  Normalizing...")
    normalize(cmip6_data, var_stats, available_vars)
    normalize(cmip5_data, var_stats, available_vars)

    # Process labels
    print("  Processing labels...")
    label_vals = np.nan_to_num(ds_label.nino.values, nan=0.0)
    cmip6_nino = cat_over_last_dim(data_transform(label_vals[:cmip6_cutoff], cmip6_ypm))
    cmip5_nino = cat_over_last_dim(data_transform(label_vals[cmip6_cutoff:], cmip5_ypm))

    lat_vals = ds_train.lat.values.copy()
    lon_vals = ds_train.lon.values.copy()
    ds_train.close()
    ds_label.close()

    print(f"  CMIP6 data: {cmip6_data.shape}  (months, lat, lon, ch)")
    print(f"  CMIP5 data: {cmip5_data.shape}")
    print(f"  CMIP6 nino: {cmip6_nino.shape}, CMIP5 nino: {cmip5_nino.shape}")

    return cmip6_data, cmip5_data, cmip6_nino, cmip5_nino, var_stats, lat_vals, lon_vals


# ─── SODA processing ───

def process_soda(data_dir, var_stats):
    """Process SODA raw nc → normalized numpy arrays."""
    print()
    print("=" * 60)
    print("Step 2/2: Processing SODA data")
    print("=" * 60)

    data_dir = Path(data_dir)
    ds_train = xr.open_dataset(data_dir / 'SODA_train.nc')
    ds_label = xr.open_dataset(data_dir / 'SODA_label.nc')

    if 'year' in ds_train.dims and 'month' in ds_train.dims:
        ds_train = ds_train.transpose('year', 'month', 'lat', 'lon')
    if 'year' in ds_label.dims and 'month' in ds_label.dims:
        ds_label = ds_label.transpose('year', 'month')

    print(f"  Raw shape: {dict(ds_train.dims)}")

    lon_mask = np.logical_and(ds_train.lon.values >= 95, ds_train.lon.values <= 330)
    ds_train = ds_train.sel(lon=lon_mask)
    print(f"  After lon filter: {dict(ds_train.dims)}")

    available_vars = [v for v in DEFAULT_VARS if v in ds_train.data_vars]
    n_years = ds_train.dims['year']

    soda_list = []
    for vn in tqdm(available_vars, desc="  SODA vars", unit="var"):
        vals = np.nan_to_num(ds_train[vn].values, nan=0.0).astype(np.float32)
        soda_list.append(cat_over_last_dim(
            data_transform(vals, num_years_per_model=n_years)))

    soda_data = np.stack(soda_list, axis=-1).astype(np.float32)

    print("  Normalizing (using CMIP stats)...")
    if var_stats is not None:
        normalize(soda_data, var_stats, available_vars)

    print("  Processing labels...")
    label_vals = np.nan_to_num(ds_label.nino.values, nan=0.0).astype(np.float32)
    soda_nino = cat_over_last_dim(data_transform(label_vals, num_years_per_model=n_years))

    lat_vals = ds_train.lat.values.copy()
    lon_vals = ds_train.lon.values.copy()
    ds_train.close()
    ds_label.close()

    print(f"  SODA data: {soda_data.shape}  (months, lat, lon, ch)")
    print(f"  SODA nino: {soda_nino.shape}")

    return soda_data, soda_nino, lat_vals, lon_vals


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ENSO data — run once to generate cache files for fast training")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing CMIP_train.nc, CMIP_label.nc, SODA_train.nc, SODA_label.nc')
    parser.add_argument('--cmip6_cutoff', type=int, default=2265,
                        help='First N year-rows that belong to CMIP6')
    parser.add_argument('--cmip6_years_per_model', type=int, default=151)
    parser.add_argument('--cmip5_years_per_model', type=int, default=140)
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if cache exists')
    args = parser.parse_args()

    data_dir = str(args.data_dir)

    # Check input files
    for fname in ['CMIP_train.nc', 'CMIP_label.nc', 'SODA_train.nc', 'SODA_label.nc']:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"ERROR: {fpath} not found!")
            sys.exit(1)

    cmip_cache = os.path.join(data_dir, ".cache_cmip_all.npz")
    soda_cache = os.path.join(data_dir, ".cache_soda.npz")

    if not args.force and os.path.exists(cmip_cache) and os.path.exists(soda_cache):
        print("Cache files already exist. Use --force to reprocess.")
        print(f"  {cmip_cache}")
        print(f"  {soda_cache}")
        return

    # Remove old cache if forcing
    if args.force:
        for f in [cmip_cache, os.path.join(data_dir, ".cache_cmip_meta.pkl"),
                  soda_cache, os.path.join(data_dir, ".cache_soda_meta.pkl")]:
            if os.path.exists(f):
                os.remove(f)
                print(f"Removed old cache: {f}")

    import time
    t0 = time.time()

    # Step 1: CMIP
    cmip6_data, cmip5_data, cmip6_nino, cmip5_nino, var_stats, lat_vals, lon_vals = \
        process_cmip(data_dir, args.cmip6_cutoff,
                     args.cmip6_years_per_model, args.cmip5_years_per_model)

    # Step 2: SODA
    soda_data, soda_nino, soda_lat_vals, soda_lon_vals = \
        process_soda(data_dir, var_stats)

    # Save CMIP cache
    print()
    print("=" * 60)
    print("Saving cache files...")
    print("=" * 60)

    print(f"  Saving {cmip_cache} ...")
    with tqdm(total=1, desc="  Saving CMIP cache", unit="file") as pbar:
        np.savez_compressed(cmip_cache,
                            cmip6_data=cmip6_data, cmip5_data=cmip5_data,
                            cmip6_nino=cmip6_nino, cmip5_nino=cmip5_nino)
        pbar.update(1)
    with open(os.path.join(data_dir, ".cache_cmip_meta.pkl"), 'wb') as f:
        pickle.dump({'var_stats': var_stats, 'lat_vals': lat_vals, 'lon_vals': lon_vals}, f)

    with tqdm(total=1, desc="  Saving SODA cache", unit="file") as pbar:
        np.savez_compressed(soda_cache,
                            soda_data=soda_data,
                            soda_nino=soda_nino)
        pbar.update(1)
    with open(os.path.join(data_dir, ".cache_soda_meta.pkl"), 'wb') as f:
        pickle.dump({'lat_vals': soda_lat_vals, 'lon_vals': soda_lon_vals}, f)

    # Summary
    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"Preprocessing complete! ({elapsed:.0f}s)")
    print("=" * 60)
    print(f"  CMIP6:    {cmip6_data.shape}")
    print(f"  CMIP5:    {cmip5_data.shape}")
    print(f"  SODA:     {soda_data.shape}")
    print(f"  Channels: {cmip6_data.shape[3]} ({', '.join(DEFAULT_VARS[:cmip6_data.shape[3]])})")

    nino_lat, nino_lon = find_nino_indices(lat_vals, lon_vals)
    print(f"  Niño 3.4:  lat[{nino_lat}], lon[{nino_lon}]")
    print(f"  Grid:      {cmip6_data.shape[1]} lat × {cmip6_data.shape[2]} lon")
    print()
    print("Cache files ready. Training will load them instantly.")


if __name__ == "__main__":
    main()
