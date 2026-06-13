#!/usr/bin/env python
"""Visualize NWP SSTA training metrics + test results + prediction sample.

CSV format (metrics.csv):
    epoch, train_loss, valid_loss, valid_mse, valid_mae, learning_rate

Usage:
    # All plots (training curves + test bar + prediction sample)
    python scripts/cuboid_transformer/nwp_sst/visualize_logs.py \
        --exp_dir /home/lab/zhangxm/gxy/Earthformer/scripts/cuboid_transformer/nwp_sst/experiments/nwp_exp1/ \
        --ckpt_path /home/lab/zhangxm/gxy/Earthformer/scripts/cuboid_transformer/nwp_sst/experiments/nwp_exp1/checkpoints/last.ckpt \
        --data_dir datasets/SST-PREDICT/ \
        --save experiment_summary

    # Training curves only
    python scripts/cuboid_transformer/nwp_sst/visualize_logs.py \
        --exp_dir experiments/nwp_exp1/ --save curves
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safer on servers
import matplotlib.pyplot as plt
import pandas as pd


def plot_training(df, save_path):
    """4-panel training curves. Saves to file, returns path."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Train & Valid loss
    ax = axes[0, 0]
    if 'train_loss' in df.columns and len(df) > 0:
        ax.plot(df['epoch'], df['train_loss'], 'b-o', markersize=4, label='Train')
    if 'valid_loss' in df.columns and len(df) > 0:
        ax.plot(df['epoch'], df['valid_loss'], 'r-s', markersize=4, label='Valid')
        best = df['valid_loss'].idxmin()
        ax.axvline(x=df.loc[best, 'epoch'], color='gray', linestyle='--', alpha=0.5)
        ax.annotate(f'best epoch {int(df.loc[best,"epoch"])}',
                    (df.loc[best, 'epoch'], df.loc[best, 'valid_loss']),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (masked, per-day)')
    ax.set_title('Train & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Valid MSE & MAE (dual axis)
    ax = axes[0, 1]
    if 'valid_mse' in df.columns and len(df) > 0:
        ax.plot(df['epoch'], df['valid_mse'], 'r-o', markersize=4, label='MSE (norm)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE', color='red')
    ax.tick_params(axis='y', labelcolor='red')
    ax.grid(True, alpha=0.3)
    if 'valid_mae' in df.columns and len(df) > 0:
        ax2 = ax.twinx()
        ax2.plot(df['epoch'], df['valid_mae'], 'b-s', markersize=4, label='MAE (norm)')
        ax2.set_ylabel('MAE', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
    ax.set_title('Validation MSE & MAE')

    # 3. Learning rate
    ax = axes[1, 0]
    if 'learning_rate' in df.columns and len(df) > 0:
        ax.plot(df['epoch'], df['learning_rate'], 'purple', marker='.', markersize=3, linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('LR Schedule (warmup + cosine)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 4. Valid loss zoom (last 80%)
    ax = axes[1, 1]
    start = max(0, int(len(df) * 0.2))
    if 'valid_loss' in df.columns and len(df) > start:
        subset = df.iloc[start:]
        ax.plot(subset['epoch'], subset['valid_loss'], 'r-s', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Valid Loss')
    ax.set_title('Valid Loss (later epochs)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


def plot_test(test_csv, save_path):
    """RMSE/MAE bar chart per lead day. Saves to file, returns path."""
    if not os.path.exists(test_csv):
        print(f"  Test metrics not found, skipping: {test_csv}")
        return None

    df = pd.read_csv(test_csv)
    days = df[df['lead_day'] != 'avg']
    if len(days) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, color, title in [
        (axes[0], 'rmse_celsius', 'steelblue', 'SSTA RMSE (°C) by Lead Day'),
        (axes[1], 'mae_celsius', 'coral', 'SSTA MAE (°C) by Lead Day')]:
        ax.bar(range(len(days)), days[col].values, color=color, alpha=0.8)
        ax.set_xlabel('Lead Day')
        ax.set_ylabel('°C')
        ax.set_title(title)
        ax.set_xticks(range(len(days)))
        ax.set_xticklabels(days['lead_day'].values)
        ax.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(days[col].values):
            ax.text(i, v + max(0.003, v * 0.01), f'{v:.3f}', ha='center', fontsize=9)

    avg_row = df[df['lead_day'] == 'avg']
    if len(avg_row) > 0:
        for ax, col in [(axes[0], 'rmse_celsius'), (axes[1], 'mae_celsius')]:
            avg_val = avg_row[col].values[0]
            ax.axhline(y=avg_val, color='red', linestyle='--', alpha=0.5, label=f'Avg={avg_val:.3f}°C')
            ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


def plot_prediction(ckpt_path, data_dir, save_path):
    """Load checkpoint, run inference on one random test sample, plot pred vs true."""
    import torch
    from earthformer.datasets.nw_pacific_dataset import build_dataloaders

    stats = np.load(os.path.join(data_dir, "normalization_stats.npz"))
    ssta_std = float(stats['ssta_std'])

    _, _, test_loader, _ = build_dataloaders(data_dir=data_dir, batch_size=1, num_workers=0)

    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    sd = {k.replace('torch_nn_module.', ''): v for k, v in sd.items()
          if k.startswith('torch_nn_module.')} or sd

    # Detect old 3‑channel checkpoints and pad initial Conv weights
    old_ckpt = False
    for k, v in sd.items():
        if 'initial_encoder' in k and 'conv_block' in k and 'conv' in k \
           and v.ndim == 4 and v.shape[1] == 3:
            old_ckpt = True
            print(f"  Detected 3‑channel checkpoint — padding Conv to 4 channels")
    if old_ckpt:
        for k, v in list(sd.items()):
            if 'initial_encoder' in k and 'conv_block' in k and 'conv' in k \
               and v.ndim == 4 and v.shape[1] == 3:
                pad = torch.zeros(v.shape[0], 1, v.shape[2], v.shape[3])
                sd[k] = torch.cat([v, pad], dim=1)

    from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
    model = CuboidTransformerModel(
        input_shape=(14, 161, 241, 4), target_shape=(3, 161, 241, 1),
        base_units=64, scale_alpha=1.0,
        enc_depth=[2, 2, 2], dec_depth=[2, 2, 2],
        enc_use_inter_ffn=True, dec_use_inter_ffn=True, dec_hierarchical_pos_embed=True,
        downsample=2, downsample_type="patch_merge", upsample_type="upsample",
        enc_attn_patterns=["axial", "spatial_lg_8", "divided_st"],
        dec_self_attn_patterns=["axial", "spatial_lg_8", "divided_st"],
        dec_cross_attn_patterns=["cross_1x1"] * 3,
        dec_use_first_self_attn=False,
        num_heads=4, attn_drop=0.1, proj_drop=0.1, ffn_drop=0.1,
        ffn_activation="gelu", norm_layer="layer_norm", padding_type="zeros",
        pos_embed_type="t+h+w", use_relative_pos=True,
        self_attn_use_final_proj=True, z_init_method="zeros",
        num_global_vectors=8,
        use_dec_self_global=True, dec_self_update_global=True,
        use_dec_cross_global=True, use_global_vector_ffn=True,
        initial_downsample_type="conv", initial_downsample_activation="leaky",
        initial_downsample_scale=[1, 4, 4],
        initial_downsample_conv_layers=3, final_upsample_conv_layers=2, checkpoint_level=0,
    )
    model.load_state_dict(sd, strict=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    # Get one random batch
    data_iter = iter(test_loader)
    rng = np.random.default_rng()
    for _ in range(rng.integers(0, max(1, len(test_loader)))):
        try:
            X, Y, mask = next(data_iter)
        except StopIteration:
            break
    X, Y, mask = X.to(device), Y.to(device), mask.to(device)

    with torch.no_grad():
        pred = model(X)

    X_c = X[0, -1, ..., 0].cpu().numpy() * ssta_std
    Y_c = Y[0, :, ..., 0].cpu().numpy() * ssta_std
    P_c = pred[0, :, ..., 0].cpu().numpy() * ssta_std
    m = mask[0, ..., 0].cpu().numpy().astype(bool)
    for arr in [Y_c, P_c]:
        arr[:, ~m] = np.nan
    X_c[~m] = np.nan

    # Clamp colorbar to 1st–99th percentile of valid data (outlier-proof)
    all_valid = np.concatenate([
        X_c[~np.isnan(X_c)].ravel(),
        Y_c[~np.isnan(Y_c)].ravel(),
        P_c[~np.isnan(P_c)].ravel(),
    ])
    if len(all_valid) > 0:
        lo, hi = np.nanpercentile(all_valid, [1, 99])
        vmax = max(abs(lo), abs(hi), 0.5)
    else:
        vmax = 1.0
    vmin = -vmax

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    ax = axes[0, 0]
    im = ax.imshow(X_c, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower',
                   extent=[120, 180, 10, 50], aspect='auto')
    ax.set_title('Input (last day) °C')
    plt.colorbar(im, ax=ax, shrink=0.8)

    for d in range(3):
        for col, (data, label) in enumerate([(P_c[d], 'Pred'), (Y_c[d], 'True')], 1):
            ax_d = axes[d, col]
            im = ax_d.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower',
                             extent=[120, 180, 10, 50], aspect='auto')
            ax_d.set_title(f'Day +{d+1} ({label})')
            plt.colorbar(im, ax=ax_d, shrink=0.8)

    rmse_d = [np.sqrt(np.nanmean((P_c[d] - Y_c[d]) ** 2)) for d in range(3)]
    mae_d = [np.nanmean(np.abs(P_c[d] - Y_c[d])) for d in range(3)]
    fig.suptitle(
        f'SSTA Prediction — RMSE: {[f"{v:.3f}" for v in rmse_d]} °C  |  '
        f'MAE: {[f"{v:.3f}" for v in mae_d]} °C', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")
    print(f"    RMSE(°C): {[f'{v:.3f}' for v in rmse_d]}")
    print(f"    MAE( °C): {[f'{v:.3f}' for v in mae_d]}")
    return save_path


def main():
    p = argparse.ArgumentParser(description="Visualize NWP training metrics")
    p.add_argument('--exp_dir', type=str, required=True,
                   help='Experiment directory (e.g. experiments/nwp_exp1/)')
    p.add_argument('--save', type=str, default=None,
                   help='Base name for saved plots')
    p.add_argument('--ckpt_path', type=str, default=None,
                   help='Checkpoint path for prediction sample')
    p.add_argument('--data_dir', type=str, default='datasets/SST-PREDICT/')
    args = p.parse_args()

    generated = []

    # ── 1. Training curves ──
    csv_path = os.path.join(args.exp_dir, 'metrics.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Detect epoch resets (multiple runs appended) — keep only last run
        resets = df.index[df['epoch'] < df['epoch'].shift()].tolist()
        if resets:
            last_start = resets[-1]
            print(f"metrics.csv: {len(df)} rows — detected {len(resets)} restart(s), "
                  f"using last {len(df) - last_start} rows")
            df = df.iloc[last_start:].reset_index(drop=True)

        print(f"metrics.csv: {len(df)} epochs ({int(df['epoch'].min())}→{int(df['epoch'].max())})")
        if 'valid_mse' in df.columns and len(df) > 0:
            best = df['valid_mse'].idxmin()
            print(f"  Best: epoch {int(df.loc[best,'epoch'])} (valid_mse={df.loc[best,'valid_mse']:.6f})")

        train_save = args.save.rsplit('.', 1)[0] + '_train.png' if args.save else \
            os.path.join(args.exp_dir, 'training_curves.png')
        generated.append(plot_training(df, train_save))
    else:
        print(f"metrics.csv not found, skipping training curves.")

    # ── 2. Test metrics bar charts ──
    test_csv = os.path.join(args.exp_dir, 'test_metrics.csv')
    test_save = args.save.rsplit('.', 1)[0] + '_test.png' if args.save else \
        os.path.join(args.exp_dir, 'test_curves.png')
    plot_test(test_csv, test_save)

    # ── 3. Prediction sample ──
    if args.ckpt_path:
        pred_save = args.save.rsplit('.', 1)[0] + '_pred.png' if args.save else \
            os.path.join(args.exp_dir, 'prediction_sample.png')
        plot_prediction(args.ckpt_path, args.data_dir, pred_save)

    print(f"\nDone. Generated files:")
    for f in generated:
        if f and os.path.exists(f):
            print(f"  {f}")


if __name__ == "__main__":
    main()
