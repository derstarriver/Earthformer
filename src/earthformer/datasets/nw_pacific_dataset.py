#!/usr/bin/env python
"""
NW Pacific SST Prediction Dataset for Earthformer.

Builds a lazy-loading PyTorch Dataset from preprocessed ERA5 + SLA data:
  - SSTA (ssta.nc): daily sea surface temperature anomaly
  - Wind (Wind_cropped.nc): daily 10m u10, v10
  - SLA  (SLA_cropped.nc): daily sea level anomaly
  - Ocean mask (mask.npy): 1=ocean, 0=land

    Input:  14 days × 161×241 × 4 channels [ssta, u10, v10, sla]
    Output:  3 days × 161×241 × 1 channel  [ssta]

Usage:
    from earthformer.datasets.nw_pacific_dataset import build_dataloaders
    train_loader, val_loader, test_loader, stats = build_dataloaders(
        data_dir="datasets/SST-PREDICT/", batch_size=2)
"""
import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime


# ─── Paths (relative to project root) ───
DEFAULT_DATA_DIR = "datasets/SST-PREDICT"

# ─── Task config ───
INPUT_LEN = 14           # 14-day context
PRED_LEN = 3             # 3-day forecast
TOTAL_LEN = INPUT_LEN + PRED_LEN  # 17

# ─── Train/val/test split by year ───
TRAIN_YEARS = (2001, 2022)
VAL_YEARS = (2023, 2024)
TEST_YEARS = (2025, 2025)


class NWPacificDataset(Dataset):
    """Lazy dataset for NW Pacific daily SST prediction.

    Stores a single contiguous float32 array of shape (N_time, lat, lon, C)
    plus an ocean mask. __getitem__ slices out (TOTAL_LEN, lat, lon, C) on
    demand without copying the full dataset.
    """

    def __init__(self, data, mask, input_len=14, pred_len=3, stride=1):
        """
        Parameters
        ----------
        data: np.ndarray, shape (N_time, lat, lon, channels)
            Pre-masked & normalized data.
        mask: np.ndarray, shape (lat, lon)
            Ocean mask (1=ocean, 0=land).
        stride: int
            Sliding window step size (1 = every day, 3 = every 3 days, etc.)
        """
        super().__init__()
        self.data = data
        self.mask = mask
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_len = input_len + pred_len
        self.stride = stride
        self.n_samples = (data.shape[0] - self.total_len) // stride + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"Data length {data.shape[0]} too short for "
                f"input_len={input_len} + pred_len={pred_len}")

        size_gb = data.nbytes / 1e9
        print(f"  Dataset: {self.n_samples:,} samples (stride={stride}), "
              f"data={data.shape}, {size_gb:.2f} GB (in-memory)")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        seq = self.data[start:start + self.total_len]            # (17, 161, 241, C)
        x = np.ascontiguousarray(seq[:self.input_len])           # (14, 161, 241, C)
        y = np.ascontiguousarray(seq[self.input_len:, ..., 0:1]) # (3, 161, 241, 1) — ssta only

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(self.mask[..., None]),  # (161, 241, 1)
        )


def compute_normalization_stats(data_dir, train_mask=None):
    """Compute per-channel mean/std from the training portion only.

    Returns dict of {channel_name: (mean, std)} and saves to .npz.
    """
    print_step("Computing normalization stats (train set only) ...")

    ssta_path = os.path.join(data_dir, "ssta.nc")
    wind_path = os.path.join(data_dir, "Wind_cropped.nc")

    ds_ssta = xr.open_dataset(ssta_path)
    ds_wind = xr.open_dataset(wind_path)

    ssta = ds_ssta['ssta'].values
    u10 = ds_wind['u10'].values
    v10 = ds_wind['v10'].values

    # Build year mask if not provided
    if train_mask is None:
        years = ds_ssta.valid_time.dt.year.values
        train_mask = (years >= TRAIN_YEARS[0]) & (years <= TRAIN_YEARS[1])

    # Ocean-only stats (exclude land=0 for mean computation)
    mask_path = os.path.join(data_dir, "mask.npy")
    ocean_mask = np.load(mask_path).astype(bool)

    def ocean_mean_std(arr, time_mask):
        train_data = arr[time_mask]  # select train years
        # Only compute over ocean pixels (ignore land=0 which was filled)
        ocean_vals = train_data[:, ocean_mask]
        mean = float(np.mean(ocean_vals))
        std = float(np.std(ocean_vals))
        return mean, std

    stats = {}
    stats['ssta'] = ocean_mean_std(ssta, train_mask)
    u10_m = float(np.mean(u10[train_mask][:, ocean_mask]))
    u10_s = float(np.std(u10[train_mask][:, ocean_mask]))
    v10_m = float(np.mean(v10[train_mask][:, ocean_mask]))
    v10_s = float(np.std(v10[train_mask][:, ocean_mask]))
    stats['u10'] = (u10_m, u10_s)
    stats['v10'] = (v10_m, v10_s)

    # SLA: ocean-only stats (NaN-safe, exclude land + missing data)
    sla_path = os.path.join(data_dir, "SLA_cropped.nc")
    ds_sla = xr.open_dataset(sla_path)
    sla = ds_sla['sla'].values
    sla_train = sla[train_mask]                          # (T_train, 161, 241)
    sla_ocean = sla_train[:, ocean_mask]                 # ocean pixels only
    sla_mean = float(np.nanmean(sla_ocean))
    sla_std  = float(np.nanstd(sla_ocean))
    stats['sla'] = (sla_mean, sla_std)
    ds_sla.close()

    print(f"  ssta: mean={stats['ssta'][0]:.4f}, std={stats['ssta'][1]:.4f}")
    print(f"  u10:  mean={stats['u10'][0]:.4f}, std={stats['u10'][1]:.4f}")
    print(f"  v10:  mean={stats['v10'][0]:.4f}, std={stats['v10'][1]:.4f}")
    print(f"  sla:  mean={stats['sla'][0]:.4f}, std={stats['sla'][1]:.4f}")

    # Save
    npz_path = os.path.join(data_dir, "normalization_stats.npz")
    np.savez(npz_path,
             ssta_mean=stats['ssta'][0], ssta_std=stats['ssta'][1],
             u10_mean=stats['u10'][0], u10_std=stats['u10'][1],
             v10_mean=stats['v10'][0], v10_std=stats['v10'][1],
             sla_mean=stats['sla'][0], sla_std=stats['sla'][1])
    print(f"  Saved: {npz_path}")

    ds_ssta.close()
    ds_wind.close()
    return stats


def build_data_array(data_dir, stats=None):
    """Load SSTA + Wind + SLA, mask land→0, normalize, stack into (T, lat, lon, 4).

    Returns the consolidated array + ocean mask.
    """
    print_step("Loading and assembling data array ...")

    ssta_path = os.path.join(data_dir, "ssta.nc")
    wind_path = os.path.join(data_dir, "Wind_cropped.nc")
    sla_path  = os.path.join(data_dir, "SLA_cropped.nc")
    mask_path = os.path.join(data_dir, "mask.npy")

    ds_ssta = xr.open_dataset(ssta_path)
    ds_wind = xr.open_dataset(wind_path)
    ds_sla  = xr.open_dataset(sla_path)
    mask = np.load(mask_path).astype(np.float32)

    ssta = ds_ssta['ssta'].values.astype(np.float32)   # (T, 161, 241)
    u10  = ds_wind['u10'].values.astype(np.float32)
    v10  = ds_wind['v10'].values.astype(np.float32)
    sla  = ds_sla['sla'].values.astype(np.float32)      # (T, 161, 241)
    years = ds_ssta.valid_time.dt.year.values

    ds_ssta.close()
    ds_wind.close()
    ds_sla.close()

    # Verify time alignment
    assert sla.shape[0] == ssta.shape[0], \
        f"SLA time dim {sla.shape[0]} != SSTA {ssta.shape[0]}"

    print(f"  Raw shapes: ssta={ssta.shape}, u10={u10.shape}, v10={v10.shape}, sla={sla.shape}")

    # ── Normalize FIRST (ocean-only stats, land gets arbitrary values) ──
    if stats is None:
        stats = compute_normalization_stats(data_dir)

    ssta = (ssta - stats['ssta'][0]) / stats['ssta'][1]
    u10  = (u10  - stats['u10'][0]) / stats['u10'][1]
    v10  = (v10  - stats['v10'][0]) / stats['v10'][1]
    sla  = (sla  - stats['sla'][0]) / stats['sla'][1]

    # ── Mask land → 0 AFTER normalize (guarantees land ≡ 0) ──
    ssta = np.where(mask, ssta, 0.0)
    u10  = np.where(mask, u10,  0.0)
    v10  = np.where(mask, v10,  0.0)
    # SLA: mask land AND NaN (coastal altimetry gaps) to 0
    sla  = np.where((mask > 0) & ~np.isnan(sla), sla, 0.0)
    n_land = int((mask == 0).sum())
    n_sla_nan = int(((mask > 0) & np.isnan(sla)).sum())
    print(f"  Land cells set to 0: {n_land}")
    print(f"  SLA ocean NaN cells: {n_sla_nan}")
    print(f"  Normalized: ssta∈[{ssta.min():.2f},{ssta.max():.2f}], "
          f"u10∈[{u10.min():.2f},{u10.max():.2f}], v10∈[{v10.min():.2f},{v10.max():.2f}], "
          f"sla∈[{sla[sla!=0].min():.2f},{sla.max():.2f}]")

    # ── Stack channels ──
    data = np.stack([ssta, u10, v10, sla], axis=-1)  # (T, 161, 241, 4)
    print(f"  Final data shape: {data.shape} ({data.nbytes/1e9:.2f} GB)")

    return data, mask, years


def build_dataloaders(data_dir=None, batch_size=2, num_workers=4, stats=None):
    """Main entry point: build train/val/test DataLoaders.

    Parameters
    ----------
    data_dir: str
        Path to directory containing ssta.nc, Wind_cropped.nc, mask.npy
    batch_size: int
    num_workers: int
    stats: dict or None
        If None, compute from training set and save to normalization_stats.npz

    Returns
    -------
    train_loader, val_loader, test_loader: DataLoader
    stats: dict
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    print("=" * 60)
    print("  NW Pacific Dataset Builder")
    print(f"  Input: {INPUT_LEN}d × 4 channels [ssta,u10,v10,sla]  →  Output: {PRED_LEN}d × 1 channel [ssta]")
    print("=" * 60)

    # ── Load stats if already computed ──
    npz_path = os.path.join(data_dir, "normalization_stats.npz")
    if stats is None and os.path.exists(npz_path):
        s = np.load(npz_path)
        if 'sla_mean' not in s:
            # Old stats without SLA → recompute
            print("  Old normalization stats (no SLA), recomputing ...")
        else:
            stats = {
                'ssta': (float(s['ssta_mean']), float(s['ssta_std'])),
                'u10':  (float(s['u10_mean']),  float(s['u10_std'])),
                'v10':  (float(s['v10_mean']),  float(s['v10_std'])),
                'sla':  (float(s['sla_mean']),  float(s['sla_std'])),
            }
            print(f"  Loaded normalization stats from {npz_path}")

    # ── Build unified data array ──
    data, mask, years = build_data_array(data_dir, stats=stats)

    # ── Time split ──
    print_step("Splitting by year ...")
    train_idx = np.where((years >= TRAIN_YEARS[0]) & (years <= TRAIN_YEARS[1]))[0]
    val_idx = np.where((years >= VAL_YEARS[0]) & (years <= VAL_YEARS[1]))[0]
    test_idx = np.where((years >= TEST_YEARS[0]) & (years <= TEST_YEARS[1]))[0]

    print(f"  Train: {train_idx[0]}-{train_idx[-1]} ({len(train_idx)} days, "
          f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[1]})")
    print(f"  Val:   {val_idx[0]}-{val_idx[-1]} ({len(val_idx)} days, "
          f"{VAL_YEARS[0]}-{VAL_YEARS[1]})")
    print(f"  Test:  {test_idx[0]}-{test_idx[-1]} ({len(test_idx)} days, "
          f"{TEST_YEARS[0]}-{TEST_YEARS[1]})")

    # ── Build Dataset (in-memory, lazy window) ──
    print_step("Building Datasets ...")
    train_ds = NWPacificDataset(data[train_idx], mask, INPUT_LEN, PRED_LEN, stride=3)
    val_ds   = NWPacificDataset(data[val_idx], mask, INPUT_LEN, PRED_LEN, stride=1)
    test_ds  = NWPacificDataset(data[test_idx], mask, INPUT_LEN, PRED_LEN, stride=1)

    # ── DataLoaders ──
    print_step("Building DataLoaders ...")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"\n  Train: {len(train_ds):,} samples → {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds):,} samples → {len(val_loader)} batches")
    print(f"  Test:  {len(test_ds):,} samples → {len(test_loader)} batches")

    # ── Quick sanity check ──
    print_step("Sanity check (first batch) ...")
    X, Y, m = next(iter(train_loader))
    print(f"  X:     {list(X.shape)}  (B,T,H,W,C)   — {X.dtype}")
    print(f"  Y:     {list(Y.shape)}  (B,T,H,W,C)   — {Y.dtype}")
    print(f"  mask:  {list(m.shape)}  (H,W,C)")
    print(f"  X NaN:     {torch.isnan(X).sum().item()}")
    print(f"  X mean±std: {X.mean().item():.4f} ± {X.std().item():.4f}")
    print(f"  Y mean±std: {Y.mean().item():.4f} ± {Y.std().item():.4f}")

    print(f"\n{'=' * 60}")
    print(f"  Dataset ready for Earthformer training.")
    print(f"{'=' * 60}")

    return train_loader, val_loader, test_loader, stats


def print_step(msg):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ─── Quick test ───
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    train_loader, val_loader, test_loader, stats = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    # Iterate a few batches to verify throughput
    print_step("Iterating 5 train batches ...")
    for i, (X, Y, mask) in enumerate(train_loader):
        if i >= 5:
            break
        print(f"  batch {i}: X={X.shape}, Y={Y.shape}")
    print("  OK.")
