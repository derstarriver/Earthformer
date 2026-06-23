#!/usr/bin/env python
"""Persistence baseline for NW Pacific SSTA prediction.

Uses the last input-day SSTA as the prediction for all future days:
    pred[t] = X[last_day, :, :, ssta_channel]   for t = 1, 2, 3

Computes EXACTLY the same test metrics as Earthformer's test_epoch_end
(identical accumulation, aggregation, and degC conversion).

Usage:
    python scripts/cuboid_transformer/nwp_sst/persistence_baseline.py \
        --data_dir datasets/SST-PREDICT/
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from earthformer.datasets.nw_pacific_dataset import build_dataloaders


def main():
    p = argparse.ArgumentParser(description="Persistence baseline for NWP SSTA prediction")
    p.add_argument("--data_dir", type=str, default="datasets/SST-PREDICT/")
    args = p.parse_args()

    # ── Load normalization stats ──
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    stats = np.load(stats_path)
    ssta_std = float(stats["ssta_std"])
    print(f"SSTA std = {ssta_std:.4f} °C  (from {stats_path})\n")

    # ── Build test DataLoader ──
    _, _, test_loader, _ = build_dataloaders(
        data_dir=args.data_dir, batch_size=2, num_workers=2)

    n_test_samples = len(test_loader.dataset)
    print(f"Test samples: {n_test_samples}\n")

    # ── Accumulate metrics — identical to Earthformer test_step/test_epoch_end ──
    # Get device from first batch
    X_sample, _, _ = next(iter(test_loader))
    device = X_sample.device

    sq_err_sum = torch.zeros(3, device=device)
    abs_err_sum = torch.zeros(3, device=device)
    n_ocean_total = 0.0

    print("Computing persistence baseline ...")
    for X, Y, mask in test_loader:
        X, Y, mask = X.to(device), Y.to(device), mask.to(device)
        B, T_out = Y.shape[0], Y.shape[1]

        # ── Persistence: broadcast last SSTA to all output days ──
        last_ssta = X[:, -1:, :, :, 0:1]               # (B, 1, 161, 241, 1)
        pred = last_ssta.expand(-1, T_out, -1, -1, -1)  # (B, 3, 161, 241, 1) — no copy

        # ── Ocean mask (identical to Earthformer test_step line 447) ──
        mask_t = mask.reshape(B, 1, mask.shape[1], mask.shape[2], 1)

        # Per-day accumulation (identical to Earthformer test_step lines 448-451)
        sq_err = ((pred - Y) ** 2 * mask_t).sum(dim=(0, 2, 3, 4))   # (T,) sum over B,H,W,C
        abs_err = ((pred - Y).abs() * mask_t).sum(dim=(0, 2, 3, 4))  # (T,)
        n_ocean = mask_t.sum().item()  # sum all dims → B × ocean_pixels_per_sample

        sq_err_sum += sq_err
        abs_err_sum += abs_err
        n_ocean_total += n_ocean

    # ── Compute metrics (identical to Earthformer test_epoch_end lines 456-472) ──
    mse_per_day = sq_err_sum / n_ocean_total   # (3,) normalized per-pixel MSE
    mae_per_day = abs_err_sum / n_ocean_total  # (3,) normalized per-pixel MAE

    mse_degC = mse_per_day * (ssta_std ** 2)
    rmse_degC = torch.sqrt(mse_degC)
    mae_degC = mae_per_day * ssta_std

    # ── Print & Save ──
    print(f"\n  {'Day':>6s}  {'MSE( norm )':>12s}  {'MAE( norm )':>12s}"
          f"  {'RMSE(°C)':>10s}  {'MAE(°C)':>10s}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}")
    for d in range(len(mse_per_day)):
        print(f"  {d+1:>6d}  {float(mse_per_day[d]):12.6f}  {float(mae_per_day[d]):12.6f}  "
              f"{float(rmse_degC[d]):10.4f}  {float(mae_degC[d]):10.4f}")
    print(f"  {'avg':>6s}  {float(mse_per_day.mean()):12.6f}  {float(mae_per_day.mean()):12.6f}  "
          f"{float(rmse_degC.mean()):10.4f}  {float(mae_degC.mean()):10.4f}")
    print()

    # Save CSV (same format as Earthformer test_metrics.csv)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    out_path = os.path.join(script_dir, "persistence_baseline.csv")
    with open(out_path, "w") as f:
        f.write("lead_day,mse_norm,mae_norm,rmse_celsius,mae_celsius\n")
        for d in range(len(mse_per_day)):
            f.write(f"{d+1},{float(mse_per_day[d]):.6f},{float(mae_per_day[d]):.6f},"
                    f"{float(rmse_degC[d]):.4f},{float(mae_degC[d]):.4f}\n")
        f.write(f"avg,{float(mse_per_day.mean()):.6f},{float(mae_per_day.mean()):.6f},"
                f"{float(rmse_degC.mean()):.4f},{float(mae_degC.mean()):.4f}\n")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
