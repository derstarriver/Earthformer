#!/usr/bin/env python
"""Visualize NWP SSTA training metrics + test results + prediction sample.

CSV format (metrics.csv):
    epoch, train_loss, valid_loss, valid_mse, valid_mae, learning_rate

Usage:
    # All plots (training curves + test bar + prediction sample)
    python scripts/cuboid_transformer/nwp_sst/visualize_logs.py \
        --exp_dir /home/lab/zhangxm/gxy/Earthformer/scripts/cuboid_transformer/nwp_sst/experiments/nwp_exp1/ \
        --ckpt_path /home/lab/zhangxm/gxy/Earthformer/scripts/cuboid_transformer/nwp_sst/experiments/nwp_exp1/checkpoints/model-epoch=063.ckpt \
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

    # Read model arch from experiment's cfg.yaml (no hardcoding)
    from omegaconf import OmegaConf
    ckpt_dir = os.path.dirname(os.path.realpath(ckpt_path))
    exp_dir = os.path.dirname(ckpt_dir)
    cfg_path = os.path.join(exp_dir, "cfg.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"cfg.yaml not found: {cfg_path}")
    mc = OmegaConf.load(cfg_path).model
    n_blocks = len(mc.enc_depth)

    def _resolve(key):
        val = mc.get(key)
        if val is None:
            return None
        if isinstance(val, str):
            return [val] * n_blocks
        return list(val)

    model_kw = dict(
        input_shape=tuple(mc.input_shape),
        target_shape=tuple(mc.target_shape),
        base_units=mc.base_units,
        scale_alpha=mc.scale_alpha,
        enc_depth=list(mc.enc_depth),
        dec_depth=list(mc.dec_depth),
        enc_use_inter_ffn=mc.enc_use_inter_ffn,
        dec_use_inter_ffn=mc.dec_use_inter_ffn,
        dec_hierarchical_pos_embed=mc.dec_hierarchical_pos_embed,
        downsample=mc.downsample,
        downsample_type=mc.downsample_type,
        enc_attn_patterns=_resolve("self_pattern"),
        dec_self_attn_patterns=_resolve("cross_self_pattern"),
        dec_cross_attn_patterns=_resolve("cross_pattern"),
        dec_cross_last_n_frames=mc.get("dec_cross_last_n_frames"),
        dec_use_first_self_attn=mc.dec_use_first_self_attn,
        num_heads=mc.num_heads,
        attn_drop=mc.attn_drop,
        proj_drop=mc.proj_drop,
        ffn_drop=mc.ffn_drop,
        upsample_type=mc.upsample_type,
        ffn_activation=mc.ffn_activation,
        gated_ffn=mc.get("gated_ffn", False),
        norm_layer=mc.norm_layer,
        num_global_vectors=mc.num_global_vectors,
        use_dec_self_global=mc.use_dec_self_global,
        dec_self_update_global=mc.dec_self_update_global,
        use_dec_cross_global=mc.use_dec_cross_global,
        use_global_vector_ffn=mc.use_global_vector_ffn,
        use_global_self_attn=mc.get("use_global_self_attn", False),
        separate_global_qkv=mc.get("separate_global_qkv", False),
        global_dim_ratio=mc.get("global_dim_ratio", 1),
        initial_downsample_type=mc.initial_downsample_type,
        initial_downsample_activation=mc.initial_downsample_activation,
        initial_downsample_scale=list(mc.initial_downsample_scale),
        initial_downsample_conv_layers=mc.initial_downsample_conv_layers,
        final_upsample_conv_layers=mc.final_upsample_conv_layers,
        padding_type=mc.padding_type,
        z_init_method=mc.z_init_method,
        checkpoint_level=mc.get("checkpoint_level", 0),
        pos_embed_type=mc.pos_embed_type,
        use_relative_pos=mc.use_relative_pos,
        self_attn_use_final_proj=mc.self_attn_use_final_proj,
        attn_linear_init_mode=mc.get("attn_linear_init_mode", "0"),
        ffn_linear_init_mode=mc.get("ffn_linear_init_mode", "0"),
        conv_init_mode=mc.get("conv_init_mode", "0"),
        down_up_linear_init_mode=mc.get("down_up_linear_init_mode", "0"),
        norm_init_mode=mc.get("norm_init_mode", "0"),
    )
    print(f"  Model from cfg.yaml: scale_alpha={model_kw['scale_alpha']}, "
          f"downsample_scale={model_kw['initial_downsample_scale']}, "
          f"enc_depth={model_kw['enc_depth']}")

    from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
    model = CuboidTransformerModel(**model_kw)
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

    n_days = P_c.shape[0]

    # Dynamic layout: row 0 = input, rows 1..n_days = pred/true pairs
    n_rows = 1 + n_days
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.5 * n_rows))

    # Input (last day) spanning both columns
    ax_in = axes[0, 0]
    im = ax_in.imshow(X_c, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower',
                      extent=[120, 180, 10, 50], aspect='auto')
    ax_in.set_title('Input (last day SSTA) °C')
    plt.colorbar(im, ax=ax_in, shrink=0.8)
    # Hide unused second axis in input row
    axes[0, 1].set_visible(False)

    for d in range(n_days):
        for col, (data, label) in enumerate([(P_c[d], 'Pred'), (Y_c[d], 'True')]):
            ax_d = axes[d + 1, col]
            im = ax_d.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower',
                             extent=[120, 180, 10, 50], aspect='auto')
            ax_d.set_title(f'Day +{d+1} ({label})')
            plt.colorbar(im, ax=ax_d, shrink=0.8)

    rmse_d = [np.sqrt(np.nanmean((P_c[d] - Y_c[d]) ** 2)) for d in range(n_days)]
    mae_d = [np.nanmean(np.abs(P_c[d] - Y_c[d])) for d in range(n_days)]
    fig.suptitle(
        f'SSTA Prediction — RMSE(°C) avg={np.mean(rmse_d):.3f}  |  '
        f'MAE(°C) avg={np.mean(mae_d):.3f}', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")
    print(f"    RMSE(°C) per day: {[f'{v:.3f}' for v in rmse_d]}")
    print(f"    MAE( °C) per day: {[f'{v:.3f}' for v in mae_d]}")
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
