#!/usr/bin/env python
"""SST model deficiency diagnosis suite.

Runs a battery of diagnostic experiments on a trained Earthformer checkpoint
to answer: "What capability is the model missing — spatial, temporal, or physical?"

Usage:
    python scripts/cuboid_transformer/nwp_sst/diagnose_model.py \
        --exp_dir experiments/nwp_7day/ \
        --ckpt_name /home/gmm/zjj/gxy/Earthformer1/scripts/cuboid_transformer/nwp_sst/experiments/nwp_7day/checkpoints/model-epoch=033.ckpt \
        --data_dir datasets/SST-PREDICT/ \
        --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml
"""
import os, sys, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import ndimage, signal
from collections import defaultdict
import torch
import torch.nn.functional as F

_curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(_curr_dir, "..", "..", ".."))

from earthformer.datasets.nw_pacific_dataset import build_dataloaders
from earthformer.utils.checkpoint import pl_ckpt_to_pytorch_state_dict
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resolve_ckpt(ckpt_path, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # strip lightning prefix
    prefix = "torch_nn_module."
    sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
    # strip freq_branch keys if model doesn't have one (backward compat)
    return sd


def _build_model_from_cfg(cfg_path, device="cpu"):
    """Build CuboidTransformerModel from a cfg YAML.

    Uses the same config->model path as train_nwp_sst.py to guarantee consistency.
    """
    from omegaconf import OmegaConf
    oc = OmegaConf.load(open(cfg_path, "r"))
    mc = oc.model

    # Resolve attention patterns (consistent with train_nwp_sst.py)
    num_blocks = len(mc.enc_depth)

    def _resolve(key):
        val = mc[key]
        if isinstance(val, str):
            return [val] * num_blocks
        if isinstance(val, (list, tuple)):
            return list(val)
        return OmegaConf.to_container(val)

    enc_attn_patterns = _resolve("self_pattern")
    dec_self_attn_patterns = _resolve("cross_self_pattern")
    dec_cross_attn_patterns = _resolve("cross_pattern")

    model = CuboidTransformerModel(
        input_shape=list(mc.input_shape),
        target_shape=list(mc.target_shape),
        base_units=mc.base_units,
        scale_alpha=mc.get("scale_alpha", 1.0),
        enc_depth=list(mc.enc_depth),
        dec_depth=list(mc.dec_depth),
        enc_use_inter_ffn=mc.get("enc_use_inter_ffn", True),
        dec_use_inter_ffn=mc.get("dec_use_inter_ffn", True),
        dec_hierarchical_pos_embed=mc.get("dec_hierarchical_pos_embed", False),
        downsample=mc.get("downsample", 2),
        downsample_type=mc.get("downsample_type", "patch_merge"),
        upsample_type=mc.get("upsample_type", "upsample"),
        num_global_vectors=mc.get("num_global_vectors", 0),
        use_dec_self_global=mc.get("use_dec_self_global", True),
        dec_self_update_global=mc.get("dec_self_update_global", True),
        use_dec_cross_global=mc.get("use_dec_cross_global", True),
        use_global_vector_ffn=mc.get("use_global_vector_ffn", True),
        use_global_self_attn=mc.get("use_global_self_attn", False),
        separate_global_qkv=mc.get("separate_global_qkv", False),
        global_dim_ratio=mc.get("global_dim_ratio", 1),
        enc_attn_patterns=enc_attn_patterns,
        dec_self_attn_patterns=dec_self_attn_patterns,
        dec_cross_attn_patterns=dec_cross_attn_patterns,
        dec_cross_last_n_frames=mc.get("dec_cross_last_n_frames"),
        attn_drop=mc.get("attn_drop", 0.0),
        proj_drop=mc.get("proj_drop", 0.0),
        ffn_drop=mc.get("ffn_drop", 0.0),
        num_heads=mc.get("num_heads", 4),
        ffn_activation=mc.get("ffn_activation", "gelu"),
        gated_ffn=mc.get("gated_ffn", False),
        norm_layer=mc.get("norm_layer", "layer_norm"),
        padding_type=mc.get("padding_type", "zeros"),
        pos_embed_type=mc.get("pos_embed_type", "t+h+w"),
        use_relative_pos=mc.get("use_relative_pos", True),
        self_attn_use_final_proj=mc.get("self_attn_use_final_proj", True),
        dec_use_first_self_attn=mc.get("dec_use_first_self_attn", False),
        z_init_method=mc.get("z_init_method", "zeros"),
        initial_downsample_type=mc.get("initial_downsample_type", "conv"),
        initial_downsample_activation=mc.get("initial_downsample_activation", "leaky"),
        initial_downsample_scale=list(mc.get("initial_downsample_scale", [1, 4, 4])),
        initial_downsample_conv_layers=mc.get("initial_downsample_conv_layers", 3),
        final_upsample_conv_layers=mc.get("final_upsample_conv_layers", 2),
        checkpoint_level=mc.get("checkpoint_level", 0),
        attn_linear_init_mode=mc.get("attn_linear_init_mode", "0"),
        ffn_linear_init_mode=mc.get("ffn_linear_init_mode", "0"),
        conv_init_mode=mc.get("conv_init_mode", "0"),
        down_up_linear_init_mode=mc.get("down_up_linear_init_mode", "0"),
        norm_init_mode=mc.get("norm_init_mode", "0"),
    )
    return model


def _load_stats(data_dir):
    stats = np.load(os.path.join(data_dir, "normalization_stats.npz"))
    return {k: float(stats[k]) for k in stats.files}


def collect_predictions(model, dataloader, device):
    """Run inference over entire test set. Returns numpy dict.

    Model outputs delta-SSTA; pred is converted to absolute SSTA here so all
    downstream diagnostics compare apples-to-apples with truth (absolute SSTA).
    """
    model.eval()
    all_preds, all_truths, all_inputs, all_masks = [], [], [], []
    with torch.no_grad():
        for X, Y, mask in dataloader:
            X = X.to(device)
            delta_pred = model(X).cpu()                    # (B, Tout, H, W, 1) — delta
            X_last = X[:, -1:, :, :, 0:1].cpu()           # (B, 1,    H, W, 1) — last input SSTA
            pred = X_last + delta_pred                     # absolute SSTA, matches truth
            all_preds.append(pred)
            all_truths.append(Y)
            all_inputs.append(X.cpu())
            all_masks.append(mask)
    return {
        "pred": torch.cat(all_preds, dim=0).numpy(),       # (N, Tout, H, W, 1)
        "truth": torch.cat(all_truths, dim=0).numpy(),
        "input": torch.cat(all_inputs, dim=0).numpy(),     # (N, Tin, H, W, C)
        "mask": torch.cat(all_masks, dim=0).numpy(),       # (N, H, W, 1)
    }


def _ocean_mask(data):
    """Return (H, W) boolean ocean mask from data dict."""
    return data["mask"][0, :, :, 0].astype(bool)


def _to_celsius(x_norm, var="ssta", stats=None):
    if stats is None:
        return x_norm
    return x_norm * stats.get(f"{var}_std", 1.0)


# ===========================================================================
# Experiment 1.1 — Spatial Error Heatmap
# ===========================================================================

def exp_spatial_error(data, stats, out_dir):
    """Per-pixel RMSE(degC) map."""
    pred = data["pred"]                         # (N, T, H, W, 1)
    truth = data["truth"]
    omask = _ocean_mask(data)
    ssta_std = stats.get("ssta_std", 1.0)

    # RMSE over all samples & lead days, per pixel
    se = (pred - truth) ** 2
    mse_map = se.mean(axis=(0, 1, 4))           # (H, W)
    rmse_map = np.sqrt(mse_map) * ssta_std
    rmse_map[~omask] = np.nan

    # ---- figure 1: RMSE heatmap ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    im = axes[0].imshow(rmse_map, origin='lower', cmap='hot',
                        extent=[120, 180, 10, 50], aspect='auto')
    plt.colorbar(im, ax=axes[0], label='RMSE (degC)')
    axes[0].set_title('Per-pixel RMSE')

    # ---- figure 2: RMSE vs latitude ----
    lat_bins = np.linspace(10, 50, 17)  # 16 bands @ 2.5 deg
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    lat_vals = np.linspace(10, 50, rmse_map.shape[0])
    rmse_by_lat = []
    for lo, hi in zip(lat_bins[:-1], lat_bins[1:]):
        band = (lat_vals >= lo) & (lat_vals < hi)
        vals = rmse_map[band, :]
        rmse_by_lat.append(np.nanmean(vals))
    axes[1].plot(rmse_by_lat, lat_centers, 'o-')
    axes[1].set_xlabel('RMSE (degC)')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title('RMSE vs Latitude')
    axes[1].grid(True, alpha=0.3)

    fig.savefig(os.path.join(out_dir, "1p1_spatial_error.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- top-5% error locations ----
    rmse_valid = rmse_map[omask]
    thresh = np.percentile(rmse_valid, 95)
    top_pct = (rmse_map > thresh).sum() / omask.sum() * 100
    print(f"  [1.1] Mean RMSE={np.nanmean(rmse_map):.3f} degC, "
          f"top-5% (> {thresh:.3f}) span {top_pct:.1f}% of ocean pixels")

    return {"rmse_map": rmse_map, "top5_thresh": thresh}


# ===========================================================================
# Experiment 1.2 — Regional RMSE
# ===========================================================================

REGIONS = {
    "Equatorial_WPacific":   (120, 150, 10, 20),
    "Equatorial_CPacific":   (150, 180, 10, 20),
    "Kuroshio":              (130, 150, 30, 40),
    "Oyashio":               (145, 165, 40, 50),
    "Subtropical":           (150, 180, 20, 30),
    "Open_Ocean":            (160, 180, 30, 50),
}

def exp_regional_rmse(data, stats, out_dir):
    """RMSE broken down by 6 oceanographic regions."""
    pred = data["pred"]
    truth = data["truth"]
    omask = _ocean_mask(data)
    sstd = stats.get("ssta_std", 1.0)
    lon = np.linspace(120, 180, pred.shape[3])
    lat = np.linspace(10, 50, pred.shape[2])

    results = {}
    print("  [1.2] Regional RMSE (degC):")
    print(f"    {'Region':<24s} {'Day1':>7s} {'Day3':>7s} {'Day5':>7s} {'Day7':>7s}")

    for rname, (lon_lo, lon_hi, lat_lo, lat_hi) in REGIONS.items():
        lon_mask = (lon >= lon_lo) & (lon < lon_hi)
        lat_mask = (lat >= lat_lo) & (lat < lat_hi)
        rmask = lat_mask[:, None] & lon_mask[None, :] & omask
        if rmask.sum() == 0:
            results[rname] = [np.nan] * 7
            continue

        day_rmses = []
        for d in range(pred.shape[1]):
            se = (pred[:, d, :, :, 0] - truth[:, d, :, :, 0]) ** 2
            day_rmse = np.sqrt(se[:, rmask].mean()) * sstd
            day_rmses.append(day_rmse)

        results[rname] = day_rmses
        print(f"    {rname:<24s} {day_rmses[0]:7.3f} {day_rmses[2]:7.3f} "
              f"{day_rmses[4]:7.3f} {day_rmses[6]:7.3f}")

    # bar chart
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(REGIONS))
    days_to_plot = [0, 2, 4, 6]
    colors = ['#2166ac', '#92c5de', '#f4a582', '#ca0020']
    width = 0.2
    for i, (d, c) in enumerate(zip(days_to_plot, colors)):
        vals = [results[r][d] for r in REGIONS]
        ax.bar(x + i * width, vals, width, color=c, label=f'Day {d+1}')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([r.replace('_', '\n') for r in REGIONS], fontsize=8)
    ax.set_ylabel('RMSE (degC)')
    ax.legend()
    ax.set_title('Regional RMSE by Lead Day')
    fig.savefig(os.path.join(out_dir, "1p2_regional_rmse.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return results


# ===========================================================================
# Experiment 1.3 — Eddy Detection / Gradient Preservation
# ===========================================================================

def exp_gradient_error(data, stats, out_dir):
    """Check whether model preserves spatial gradients (fronts, eddy edges)."""
    pred = data["pred"]       # (N, T, H, W, 1)
    truth = data["truth"]
    omask = _ocean_mask(data)

    grad_bias_all = []
    grad_angle_all = []

    for d in range(pred.shape[1]):
        for i in range(min(pred.shape[0], 100)):  # sample cap for speed
            p = pred[i, d, :, :, 0]
            t = truth[i, d, :, :, 0]

            gy_p, gx_p = np.gradient(p)
            gy_t, gx_t = np.gradient(t)

            mag_p = np.sqrt(gx_p**2 + gy_p**2)
            mag_t = np.sqrt(gx_t**2 + gy_t**2)

            m = omask
            grad_bias_all.append(np.mean(mag_p[m] - mag_t[m]))

            dot = gx_p * gx_t + gy_p * gy_t
            denom = mag_p * mag_t + 1e-8
            cos_sim = np.clip(dot / denom, -1, 1)
            grad_angle_all.append(np.nanmean(np.arccos(cos_sim)[m]))

    mean_grad_bias = np.mean(grad_bias_all)
    mean_grad_angle = np.degrees(np.mean(grad_angle_all))

    print(f"  [1.3] Mean grad bias: {mean_grad_bias:.4f} "
          f"(<0 = oversmoothing)")
    print(f"  [1.3] Mean grad angle error: {mean_grad_angle:.1f} deg "
          f"(>45 = wrong gradient direction)")

    # plot: grad magnitude comparison (one sample)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sample_idx = 0
    p0 = pred[sample_idx, 3, :, :, 0]    # day 4
    t0 = truth[sample_idx, 3, :, :, 0]
    gy_p, gx_p = np.gradient(p0)
    gy_t, gx_t = np.gradient(t0)
    mag_p0 = np.sqrt(gx_p**2 + gy_p**2)
    mag_t0 = np.sqrt(gx_t**2 + gy_t**2)

    vmax = np.nanpercentile(np.concatenate([mag_p0[omask], mag_t0[omask]]), 99)
    axes[0].imshow(mag_t0, origin='lower', cmap='viridis', vmin=0, vmax=vmax,
                   extent=[120, 180, 10, 50], aspect='auto')
    axes[0].set_title('Truth |grad SST|')
    axes[1].imshow(mag_p0, origin='lower', cmap='viridis', vmin=0, vmax=vmax,
                   extent=[120, 180, 10, 50], aspect='auto')
    axes[1].set_title('Pred |grad SST|')
    im = axes[2].imshow(mag_p0 - mag_t0, origin='lower', cmap='RdBu_r',
                        vmin=-vmax/2, vmax=vmax/2,
                        extent=[120, 180, 10, 50], aspect='auto')
    axes[2].set_title('Diff (Pred - Truth)')
    plt.colorbar(im, ax=axes[2], shrink=0.8)
    fig.savefig(os.path.join(out_dir, "1p3_gradient_error.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {"grad_bias": mean_grad_bias, "grad_angle_error": mean_grad_angle}


# ===========================================================================
# Experiment 2.1 — Lead-day RMSE curve
# ===========================================================================

def exp_lead_day_rmse(data, stats, out_dir):
    """RMSE vs. lead day, with persistence baseline."""
    pred = data["pred"]     # (N, T, H, W, 1)
    truth = data["truth"]
    inp = data["input"]     # (N, Tin, H, W, C)
    omask = _ocean_mask(data)
    sstd = stats.get("ssta_std", 1.0)
    T_out = pred.shape[1]

    model_rmse = []
    pers_rmse = []

    for d in range(T_out):
        se = (pred[:, d, :, :, 0] - truth[:, d, :, :, 0]) ** 2
        model_rmse.append(np.sqrt(se[:, omask].mean()) * sstd)

        # persistence: last input day repeated
        pers = inp[:, -1, :, :, 0]  # last input SSTA
        se_p = (pers - truth[:, d, :, :, 0]) ** 2
        pers_rmse.append(np.sqrt(se_p[:, omask].mean()) * sstd)

    model_rmse = np.array(model_rmse)
    pers_rmse = np.array(pers_rmse)

    slope_model = (model_rmse[-1] - model_rmse[0]) / (T_out - 1)
    slope_pers = (pers_rmse[-1] - pers_rmse[0]) / (T_out - 1)
    slope_ratio = slope_model / (slope_pers + 1e-8)

    print(f"  [2.1] RMSE growth: model={slope_model:.4f} degC/day, "
          f"pers={slope_pers:.4f}, ratio={slope_ratio:.2f}")
    print(f"  [2.1] Day1={model_rmse[0]:.3f} -> Day{T_out}={model_rmse[-1]:.3f} degC")

    fig, ax = plt.subplots(figsize=(8, 5))
    days = np.arange(1, T_out + 1)
    ax.plot(days, model_rmse, 'o-', color='#2166ac', lw=2, label='Earthformer')
    ax.plot(days, pers_rmse, 's--', color='#ca0020', lw=1.5, label='Persistence')
    ax.set_xlabel('Lead Day')
    ax.set_ylabel('RMSE (degC)')
    ax.set_title(f'Error Growth (slope ratio={slope_ratio:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "2p1_lead_day_rmse.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {"model_rmse": model_rmse, "pers_rmse": pers_rmse, "slope_ratio": slope_ratio}


# ===========================================================================
# Experiment 2.2 — Autocorrelation decay
# ===========================================================================

def exp_autocorr(data, out_dir):
    """Compare ACF of pred vs truth vs persistence."""
    pred = data["pred"][:, :, :, :, 0]    # (N, T, H, W)
    truth = data["truth"][:, :, :, :, 0]
    omask = _ocean_mask(data)
    T_out = pred.shape[1]

    def _acf(seq):
        """seq: (N, T, H, W) -> per-pixel ACF averaged."""
        N, T = seq.shape[:2]
        acf_all = np.zeros((T - 1, seq.shape[2], seq.shape[3]))
        for lag in range(1, T):
            a = seq[:, lag:, :, :].reshape(-1, seq.shape[2], seq.shape[3])
            b = seq[:, :-lag, :, :].reshape(-1, seq.shape[2], seq.shape[3])
            # per-pixel correlation over time
            for hi in range(seq.shape[2]):
                for wi in range(seq.shape[3]):
                    if omask[hi, wi]:
                        aa, bb = a[:, hi, wi], b[:, hi, wi]
                        if aa.std() > 0 and bb.std() > 0:
                            acf_all[lag-1, hi, wi] = np.corrcoef(aa, bb)[0, 1]
        return np.array([np.nanmean(acf_all[lag, :, :][omask]) for lag in range(T_out - 1)])

    acf_truth = _acf(truth)
    acf_pred = _acf(pred)

    print(f"  [2.2] ACF lag-{T_out-1}: truth={acf_truth[-1]:.3f}, "
          f"pred={acf_pred[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    lags = np.arange(1, T_out)
    ax.plot(lags, acf_truth, 'o-', color='black', lw=2, label='Truth')
    ax.plot(lags, acf_pred, 's-', color='#2166ac', lw=2, label='Earthformer')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Lag (days)')
    ax.set_ylabel('Mean Autocorrelation')
    ax.set_title('Temporal Autocorrelation Decay')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "2p2_autocorr_decay.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {"acf_truth": acf_truth, "acf_pred": acf_pred}


# ===========================================================================
# Experiment 2.3 — Persistence improvement heatmap
# ===========================================================================

def exp_persistence_improve(data, stats, out_dir):
    """Per-pixel: (pers_rmse - model_rmse) / pers_rmse."""
    pred = data["pred"]
    truth = data["truth"]
    inp = data["input"]
    omask = _ocean_mask(data)
    sstd = stats.get("ssta_std", 1.0)

    # full output average RMSE
    mse_model = np.mean((pred - truth) ** 2, axis=(0, 1, 4))
    rmse_model = np.sqrt(mse_model) * sstd

    pers = inp[:, -1, :, :, 0:1]   # (N, H, W, 1)
    mse_pers = np.mean((pers[:, None, :, :, :] - truth) ** 2, axis=(0, 1, 4))
    rmse_pers = np.sqrt(mse_pers) * sstd

    improve = (rmse_pers - rmse_model) / (rmse_pers + 1e-8)
    improve[~omask] = np.nan

    neg_frac = (improve[omask] < 0).sum() / omask.sum()
    print(f"  [2.3] Model worse than persistence on {neg_frac*100:.1f}% of ocean pixels")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(improve, origin='lower', cmap='RdBu_r',
                   vmin=-1, vmax=1, extent=[120, 180, 10, 50], aspect='auto')
    plt.colorbar(im, ax=ax, label='Improvement over Persistence')
    ax.set_title(f'Model vs Persistence (negative=worse, {neg_frac*100:.1f}% worse)')
    fig.savefig(os.path.join(out_dir, "2p3_persist_improve.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {"improve_map": improve, "neg_frac": neg_frac}


# ===========================================================================
# Experiment 2.4 — Per-pixel temporal correlation
# ===========================================================================

def exp_temporal_corr(data, out_dir):
    """Per-pixel Pearson corr between pred and truth time series."""
    pred = data["pred"][:, :, :, :, 0]     # (N, T, H, W)
    truth = data["truth"][:, :, :, :, 0]
    omask = _ocean_mask(data)
    T_out = pred.shape[1]

    H, W = pred.shape[2], pred.shape[3]
    corr_map = np.zeros((T_out, H, W))

    for d in range(T_out):
        for hi in range(H):
            for wi in range(W):
                if omask[hi, wi]:
                    p = pred[:, d, hi, wi]
                    t = truth[:, d, hi, wi]
                    if p.std() > 0 and t.std() > 0:
                        corr_map[d, hi, wi] = np.corrcoef(p, t)[0, 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for idx, d in enumerate([0, 2, 4, 6]):
        ax = axes[idx // 2, idx % 2]
        cm = corr_map[d].copy()
        cm[~omask] = np.nan
        im = ax.imshow(cm, origin='lower', cmap='RdYlGn', vmin=0, vmax=1,
                       extent=[120, 180, 10, 50], aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f'Day {d+1} Temporal Correlation')
    fig.savefig(os.path.join(out_dir, "2p4_temporal_corr.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    low_corr_frac = (corr_map[6, :, :][omask] < 0.5).sum() / omask.sum()
    print(f"  [2.4] Day 7: {low_corr_frac*100:.1f}% pixels have corr < 0.5")

    return {"corr_map": corr_map, "low_corr_frac_day7": low_corr_frac}


# ===========================================================================
# Experiment 3.1 — SST tendency diagnosis
# ===========================================================================

def exp_tendency(data, stats, out_dir):
    """Check if model dSST/dt matches truth dSST/dt."""
    pred = data["pred"][:, :, :, :, 0]    # (N, T, H, W)
    truth = data["truth"][:, :, :, :, 0]
    inp = data["input"]                    # (N, Tin, H, W, C)
    omask = _ocean_mask(data)
    sstd = stats.get("ssta_std", 1.0)
    T_out = pred.shape[1]

    # day-to-day tendency (truth)
    dT_truth = truth[:, 1:, :, :] - truth[:, :-1, :, :]     # (N, T-1, H, W)
    dT_pred = pred[:, 1:, :, :] - pred[:, :-1, :, :]

    # overall correlation
    m = omask
    dTt = dT_truth[:, :, m].ravel()
    dTp = dT_pred[:, :, m].ravel()
    overall_corr = np.corrcoef(dTt, dTp)[0, 1] if dTt.std() > 0 else 0

    print(f"  [3.1] dSST/dt overall corr: {overall_corr:.3f}")

    # bin by magnitude
    bins = {"strong_cooling": (-10, -0.5), "weak": (-0.5, 0.5), "strong_warming": (0.5, 10)}
    bin_rmses = {}
    for bname, (lo, hi) in bins.items():
        bmask = (dTt > lo) & (dTt < hi)
        if bmask.sum() > 0:
            se = (dTp[bmask] - dTt[bmask]) ** 2
            bin_rmses[bname] = np.sqrt(se.mean()) * sstd
        else:
            bin_rmses[bname] = np.nan

    print(f"  [3.1] RMSE by tendency bin: "
          f"cooling={bin_rmses.get('strong_cooling', np.nan):.3f}, "
          f"weak={bin_rmses.get('weak', np.nan):.3f}, "
          f"warming={bin_rmses.get('strong_warming', np.nan):.3f} degC")

    # scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    sample = np.random.choice(len(dTt), min(5000, len(dTt)), replace=False)
    ax.scatter(dTt[sample], dTp[sample], alpha=0.15, s=2, color='#2166ac')
    ax.plot([-1, 1], [-1, 1], 'r--', lw=1)
    ax.set_xlabel('Truth dSST/dt (norm)')
    ax.set_ylabel('Pred dSST/dt (norm)')
    ax.set_title(f'dSST/dt (r={overall_corr:.3f})')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "3p1_tendency_scatter.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {"tendency_corr": overall_corr, "bin_rmses": bin_rmses}


# ===========================================================================
# Experiment 3.2 — Gradient-Advection Consistency
# ===========================================================================

def exp_advection_consistency(data, out_dir):
    """Check if model dSST/dt aligns with -grad(SST) (advection signal)."""
    pred = data["pred"][:, :, :, :, 0]    # (N, T, H, W)
    truth = data["truth"][:, :, :, :, 0]
    inp = data["input"]
    omask = _ocean_mask(data)

    # Use truth for advection check (model's internal state is harder to probe)
    # Check: does dSST/dt correlate with U10 * dSST/dx + V10 * dSST/dy?
    # We have U10, V10 in the input (channels 1, 2)
    # For multi-day: use input last day wind as proxy

    N, T_out, H, W = truth.shape
    u10_in = inp[:, -1, :, :, 1]   # last input day U10
    v10_in = inp[:, -1, :, :, 2]   # last input day V10

    cos_consistency = []
    for d in range(T_out - 1):
        dT = truth[:, d+1, :, :] - truth[:, d, :, :]
        gy, gx = np.gradient(truth[:, d, :, :], axis=(1, 2))

        # Advective tendency proxy: -(u * dSST/dx + v * dSST/dy)
        adv = -(u10_in * gx + v10_in * gy)     # (N, H, W)

        # cosine between dT and adv
        for n in range(min(N, 50)):
            a = dT[n][omask].ravel()
            b = adv[n][omask].ravel()
            if a.std() > 0 and b.std() > 0:
                c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                cos_consistency.append(c)

    mean_cos = np.mean(cos_consistency)
    pos_frac = (np.array(cos_consistency) > 0).mean()

    print(f"  [3.2] Mean cos(dSST/dt, -u·∇SST) = {mean_cos:.3f}")
    print(f"  [3.2] Fraction positive = {pos_frac*100:.1f}% "
          f"(>50% = advection signal present)")

    # scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(cos_consistency, bins=40, color='#2166ac', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='red', ls='--', lw=1.5)
    ax.axvline(mean_cos, color='black', ls='-', lw=1.5, label=f'Mean={mean_cos:.3f}')
    ax.set_xlabel('cos(dSST/dt, -u·grad SST)')
    ax.set_ylabel('Frequency')
    ax.set_title('Advection Consistency')
    ax.legend()
    fig.savefig(os.path.join(out_dir, "3p2_advection_consistency.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {"mean_cos": mean_cos, "pos_frac": pos_frac}


# ===========================================================================
# Experiment 3.3 — Perturbation propagation test
# ===========================================================================

def exp_perturbation(model, data, device, out_dir):
    """Inject a local SST anomaly and track how model propagates it."""
    omask = _ocean_mask(data)
    X0 = torch.from_numpy(data["input"][0:1]).float().to(device)  # (1, Tin, H, W, C)

    # pick a perturbation location in Kuroshio region
    # 35N, 145E -> pixel coords
    lat_vals = np.linspace(10, 50, X0.shape[2])
    lon_vals = np.linspace(120, 180, X0.shape[3])
    pert_lat_idx = np.argmin(np.abs(lat_vals - 35))
    pert_lon_idx = np.argmin(np.abs(lon_vals - 145))

    model.eval()
    with torch.no_grad():
        pred_base = model(X0).cpu().numpy()    # (1, T, H, W, 1)

        X_pert = X0.clone()
        X_pert[0, -1, pert_lat_idx, pert_lon_idx, 0] += 1.0   # +1 degC anomaly
        pred_pert = model(X_pert).cpu().numpy()

    delta = (pred_pert - pred_base)[0, :, :, :, 0]   # (T, H, W)

    # track: where does the perturbation go?
    T_out = delta.shape[0]
    pert_mass = []
    for d in range(T_out):
        # total absolute perturbation mass
        pert_mass.append(np.abs(delta[d][omask]).sum())

    # center of mass movement
    hy, wx = np.meshgrid(np.arange(delta.shape[1]), np.arange(delta.shape[2]), indexing='ij')
    com_h, com_w = [], []
    for d in range(T_out):
        w = np.abs(delta[d]) * omask
        if w.sum() > 0:
            com_h.append((hy * w).sum() / w.sum())
            com_w.append((wx * w).sum() / w.sum())
        else:
            com_h.append(pert_lat_idx)
            com_w.append(pert_lon_idx)

    print(f"  [3.3] Perturbation mass decay: {pert_mass[0]:.3f} -> {pert_mass[-1]:.3f}")
    print(f"  [3.3] COM movement: ({com_h[0]:.0f},{com_w[0]:.0f}) "
          f"-> ({com_h[-1]:.0f},{com_w[-1]:.0f}) px")

    # plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    days_plot = [0, 1, 2, 3, 4, 5, 6, -1]  # day numbers, -1 means last
    for idx, d in enumerate(days_plot):
        ax = axes[idx // 4, idx % 4]
        vmax = np.nanpercentile(np.abs(delta), 99)
        im = ax.imshow(delta[d], origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       extent=[120, 180, 10, 50], aspect='auto')
        ax.plot(lon_vals[pert_lon_idx], lat_vals[pert_lat_idx], 'k*', markersize=10)
        ax.set_title(f'Day {d+1 if d >= 0 else T_out}')
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('Perturbation Propagation (delta from +1C anomaly)', fontsize=14)
    fig.savefig(os.path.join(out_dir, "3p3_perturbation_propagation.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {"pert_mass": pert_mass, "com_movement": (com_h, com_w)}


# ===========================================================================
# Decision Tree
# ===========================================================================

def decision_tree(results):
    """Print diagnostic conclusions based on experiment results."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC DECISION TREE")
    print("=" * 70)

    issues = []

    # [1.3] Gradient
    gb = results.get("grad_bias")
    ga = results.get("grad_angle_error")
    if gb is not None and gb < -0.3 and ga is not None and ga > 45:
        issues.append("IMAGE_FITTER: Model oversmooths and loses gradient direction.")
    elif gb is not None and gb < -0.15:
        issues.append("MILD_SMOOTHING: Model slightly oversmooths. Check attention temperature.")

    # [2.2] Autocorrelation
    acf_t = results.get("acf_truth")
    acf_p = results.get("acf_pred")
    if acf_t is not None and acf_p is not None and len(acf_t) > 0:
        if acf_p[-1] < 0.5 * acf_t[-1]:
            issues.append("TIME_MEMORY_LOSS: Temporal autocorrelation decays too fast.")
        elif acf_p[-1] < 0.7 * acf_t[-1]:
            issues.append("MILD_TIME_LOSS: Moderate temporal memory degradation.")

    # [2.1] Slope ratio
    sr = results.get("slope_ratio")
    if sr is not None:
        if sr > 0.8:
            issues.append("PERSISTENCE_LIKE: Error growth similar to persistence. No dynamics learned.")
        elif sr > 0.5:
            issues.append("PARTIAL_DYNAMICS: Some dynamics learned but persistence still competitive.")

    # [2.3] Negative improvement fraction
    nf = results.get("neg_frac")
    if nf is not None:
        if nf > 0.3:
            issues.append("LARGE_NEGATIVE_REGION: >30% pixels worse than persistence.")
        elif nf > 0.15:
            issues.append("MODERATE_NEGATIVE: 15-30% pixels worse than persistence.")

    # [3.1] Tendency correlation
    tc = results.get("tendency_corr")
    if tc is not None:
        if tc < 0.3:
            issues.append("NO_TENDENCY_SKILL: Model cannot predict day-to-day SST changes.")
        elif tc < 0.5:
            issues.append("LOW_TENDENCY_SKILL: Weak day-to-day dynamics.")

    # [3.2] Advection consistency
    pf = results.get("pos_frac")
    if pf is not None:
        if pf < 0.55:
            issues.append("NO_ADVECTION: cos(dSST/dt, advection) near random.")
        elif pf < 0.6:
            issues.append("WEAK_ADVECTION: Slight advection signal detected.")

    # Print
    if not issues:
        print("  Model is within expected bounds for all diagnostics.")
        print("  If RMSE still unsatisfactory, consider: larger dataset, higher resolution.")
    else:
        for i, iss in enumerate(issues):
            print(f"  [{i+1}] {iss}")

    print("=" * 70)

    # Summary
    print("\nSUMMARY:")
    if any("IMAGE_FITTER" in i for i in issues):
        print("  Primary bottleneck: SPATIAL (image fitting, not dynamics)")
    elif any("NO_ADVECTION" in i or "NO_TENDENCY" in i for i in issues):
        print("  Primary bottleneck: PHYSICAL TRANSPORT")
    elif any("TIME_MEMORY" in i for i in issues):
        print("  Primary bottleneck: TEMPORAL DYNAMICS")
    elif any("PERSISTENCE_LIKE" in i for i in issues):
        print("  Primary bottleneck: TEMPORAL DYNAMICS (persistence-like)")
    else:
        print("  No clear single bottleneck — inspect specific regional/lead-day failures.")


# ===========================================================================
# main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="SST model diagnosis suite")
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--ckpt_name", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="datasets/SST-PREDICT/")
    p.add_argument("--cfg", type=str, required=True)
    p.add_argument("--gpu", action="store_true", default=False)
    args = p.parse_args()

    out_dir = os.path.join(args.exp_dir, "diagnosis")
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    # ── Load model ──
    print("Building model...")
    model = _build_model_from_cfg(args.cfg, device=device)
    ckpt_path = os.path.join(args.exp_dir, "checkpoints", args.ckpt_name)
    if not os.path.exists(ckpt_path):
        # try .pt variant
        alt = os.path.join(args.exp_dir, "checkpoints", "best_model.pt")
        if os.path.exists(alt):
            ckpt_path = alt
    print(f"Loading checkpoint: {ckpt_path}")
    sd = _resolve_ckpt(ckpt_path, map_location="cpu")
    # filter out freq_branch keys if present in ckpt but not in model
    model_keys = set(model.state_dict().keys())
    sd = {k: v for k, v in sd.items() if k in model_keys}
    model.load_state_dict(sd, strict=False)
    model.to(device)

    # ── Load data ──
    print("Loading test data...")
    _, _, test_loader, stats_raw = build_dataloaders(
        data_dir=args.data_dir, batch_size=2, num_workers=2)
    stats = {k: float(stats_raw[k]) for k in stats_raw.files} if hasattr(stats_raw, 'files') else {}

    # ── Collect predictions ──
    print("Running inference...")
    data = collect_predictions(model, test_loader, device)
    N, T_out, H, W, _ = data["pred"].shape
    print(f"  Collected {N} samples, {T_out} lead days, {H}x{W} grid")

    results = {}

    # ── Run experiments ──
    print("\n=== I. SPATIAL DIAGNOSIS ===")
    results.update(exp_spatial_error(data, stats, out_dir))
    results.update(exp_regional_rmse(data, stats, out_dir))
    gres = exp_gradient_error(data, stats, out_dir)
    results.update(gres)

    print("\n=== II. TEMPORAL DIAGNOSIS ===")
    lres = exp_lead_day_rmse(data, stats, out_dir)
    results.update(lres)
    ares = exp_autocorr(data, out_dir)
    results.update(ares)
    pres = exp_persistence_improve(data, stats, out_dir)
    results.update(pres)
    tres = exp_temporal_corr(data, out_dir)
    results.update(tres)

    print("\n=== III. PHYSICS DIAGNOSIS ===")
    tenres = exp_tendency(data, stats, out_dir)
    results.update(tenres)
    avres = exp_advection_consistency(data, out_dir)
    results.update(avres)
    perres = exp_perturbation(model, data, device, out_dir)
    results.update(perres)

    # ── Decision tree ──
    decision_tree(results)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
