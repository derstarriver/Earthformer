#!/usr/bin/env python
"""Train Earthformer for NW Pacific daily SSTA prediction.
1213132123132132
Input:  14 days × 161×241 × 4 channels [ssta, u10, v10, sla]
Output:  7 days × 161×241 × 1 channel  [ssta]

Usage:
    TRAIN

    14-3
    python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --save nwp_exp1 --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml



    14-7
    python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --save nwp_7day --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml

13123

 # 断点续训
python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --save nwp_exp1 --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml \
    --ckpt_name last.ckpt


?
        python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --save nwp_7day --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml
    --ckpt_name /home/gmm/zjj/gxy/Earthformer1/scripts/cuboid_transformer/nwp_sst/experiments/nwp_7day/checkpoints/last.ckpt

python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --save nwp_7day --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml \
    --ckpt_name last.ckpt


# 测试
python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --test --save nwp_exp1 --data_dir datasets/SST-PREDICT/ \
    --ckpt_name /home/lab/zhangxm/gxy/Earthformer/scripts/cuboid_transformer/nwp_sst/experiments/nwp_exp1/checkpoints/model-epoch=051.ckpt \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml

# 测试（选最优 epoch）
python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --test --save nwp_7day --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml \
    --ckpt_name model-epoch=066.ckpt
"""
import warnings
import os
import sys
import inspect
import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, Callback
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
from shutil import copyfile

# Project imports
from earthformer.config import cfg
from earthformer.utils.optim import SequentialLR, warmup_lambda
from earthformer.utils.utils import get_parameter_names
from earthformer.utils.checkpoint import pl_ckpt_to_pytorch_state_dict
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from earthformer.datasets.nw_pacific_dataset import build_dataloaders, INPUT_LEN, PRED_LEN

# PyTorch 2.6+ compat
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

try:
    import apex
    from earthformer.utils.apex_ddp import ApexDDPStrategy
    _HAS_APEX = True
except ImportError:
    _HAS_APEX = False


_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
exps_dir = os.path.join(_curr_dir, "experiments")


class EpochProgressBar(TQDMProgressBar):
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        if self.main_progress_bar is not None:
            self.main_progress_bar.set_description(
                f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}", refresh=True)


class NWPPredictionModule(pl.LightningModule):
    """PyTorch Lightning module for NW Pacific daily SSTA prediction."""

    def __init__(self, total_num_steps: int, oc_file: str = None, save_dir: str = None):
        super().__init__()

        # Load config
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self._build_config(oc_from_file)

        # Resolve attention patterns per block
        model_cfg = OmegaConf.to_object(oc.model)
        num_blocks = len(model_cfg["enc_depth"])

        def _resolve_patterns(key):
            val = model_cfg[key]
            if isinstance(val, str):
                return [val] * num_blocks
            if isinstance(val, (list, tuple)):
                return list(val)
            return OmegaConf.to_container(val)

        enc_attn_patterns = _resolve_patterns("self_pattern")
        dec_self_attn_patterns = _resolve_patterns("cross_self_pattern")
        dec_cross_attn_patterns = _resolve_patterns("cross_pattern")

        # ── Build CuboidTransformerModel ──
        self.torch_nn_module = CuboidTransformerModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg.get("block_units"),
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg.get("dec_cross_last_n_frames"),
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg.get("gated_ffn", False),
            norm_layer=model_cfg["norm_layer"],
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg.get("use_global_self_attn", False),
            separate_global_qkv=model_cfg.get("separate_global_qkv", False),
            global_dim_ratio=model_cfg.get("global_dim_ratio", 1),
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            initial_downsample_scale=model_cfg["initial_downsample_scale"],
            initial_downsample_conv_layers=model_cfg["initial_downsample_conv_layers"],
            final_upsample_conv_layers=model_cfg["final_upsample_conv_layers"],
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg.get("checkpoint_level", 0),
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            attn_linear_init_mode=model_cfg.get("attn_linear_init_mode", "0"),
            ffn_linear_init_mode=model_cfg.get("ffn_linear_init_mode", "0"),
            conv_init_mode=model_cfg.get("conv_init_mode", "0"),
            down_up_linear_init_mode=model_cfg.get("down_up_linear_init_mode", "0"),
            norm_init_mode=model_cfg.get("norm_init_mode", "0"),
        )

        self.save_hyperparameters(oc)
        self.oc = oc
        self.total_num_steps = total_num_steps
        self.save_dir = save_dir

        # Metrics
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

        self._setup_save(oc_file)

    def _build_config(self, oc_from_file=None):
        oc = OmegaConf.create()
        layout = OmegaConf.create()
        layout.in_len = INPUT_LEN
        layout.out_len = PRED_LEN
        layout.layout = "NTHWC"
        oc.layout = layout
        oc.optim = self._default_optim()
        oc.logging = self._default_logging()
        oc.trainer = self._default_trainer()
        oc.vis = self._default_vis()
        oc.model = self._default_model()
        oc.dataset = self._default_dataset()
        if oc_from_file is not None:
            oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    def _setup_save(self, cfg_file_path):
        self.save_dir = os.path.join(exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        if cfg_file_path is not None:
            target = os.path.join(self.save_dir, "cfg.yaml")
            if not os.path.exists(target) or not os.path.samefile(cfg_file_path, target):
                copyfile(cfg_file_path, target)

    @staticmethod
    def _default_optim():
        cfg = OmegaConf.create()
        cfg.total_batch_size = 16
        cfg.micro_batch_size = 2
        cfg.seed = 0
        cfg.method = "adamw"
        cfg.lr = 1e-4
        cfg.wd = 1e-5
        cfg.gradient_clip_val = 1.0
        cfg.max_epochs = 50
        cfg.warmup_percentage = 0.1
        cfg.lr_scheduler_mode = "cosine"
        cfg.min_lr_ratio = 1e-3
        cfg.warmup_min_lr_ratio = 0.0
        cfg.early_stop = True
        cfg.early_stop_mode = "min"
        cfg.early_stop_patience = 10
        cfg.save_top_k = 3
        return cfg

    @staticmethod
    def _default_logging():
        cfg = OmegaConf.create()
        cfg.logging_prefix = "NWP_SST"
        cfg.monitor_lr = True
        cfg.monitor_device = False
        cfg.track_grad_norm = -1
        cfg.use_wandb = False
        return cfg

    @staticmethod
    def _default_trainer():
        cfg = OmegaConf.create()
        cfg.check_val_every_n_epoch = 1
        cfg.log_step_ratio = 0.001
        cfg.precision = 32
        return cfg

    @staticmethod
    def _default_vis():
        cfg = OmegaConf.create()
        cfg.eval_example_only = False
        return cfg

    @staticmethod
    def _default_model():
        cfg = OmegaConf.create()
        cfg.data_channels = 4
        cfg.input_shape = (14, 161, 241, 4)
        cfg.target_shape = (7, 161, 241, 1)
        cfg.base_units = 64
        cfg.scale_alpha = 1.0
        cfg.enc_depth = [2, 2, 2]
        cfg.dec_depth = [2, 2, 2]
        cfg.enc_use_inter_ffn = True
        cfg.dec_use_inter_ffn = True
        cfg.dec_hierarchical_pos_embed = True
        cfg.downsample = 2
        cfg.downsample_type = "patch_merge"
        cfg.upsample_type = "upsample"
        cfg.num_global_vectors = 8
        cfg.use_dec_self_global = True
        cfg.dec_self_update_global = True
        cfg.use_dec_cross_global = True
        cfg.use_global_vector_ffn = True
        cfg.use_global_self_attn = False
        cfg.separate_global_qkv = False
        cfg.global_dim_ratio = 1
        cfg.self_pattern = ["axial", "spatial_lg_8", "divided_st"]
        cfg.cross_self_pattern = ["axial", "spatial_lg_8", "divided_st"]
        cfg.cross_pattern = ["cross_1x1", "cross_1x1", "cross_1x1"]
        cfg.dec_cross_last_n_frames = None
        cfg.attn_drop = 0.1
        cfg.proj_drop = 0.1
        cfg.ffn_drop = 0.1
        cfg.num_heads = 4
        cfg.ffn_activation = "gelu"
        cfg.gated_ffn = False
        cfg.norm_layer = "layer_norm"
        cfg.padding_type = "zeros"
        cfg.pos_embed_type = "t+h+w"
        cfg.use_relative_pos = True
        cfg.self_attn_use_final_proj = True
        cfg.dec_use_first_self_attn = False
        cfg.z_init_method = "zeros"
        cfg.initial_downsample_type = "conv"
        cfg.initial_downsample_activation = "leaky"
        cfg.initial_downsample_scale = [1, 4, 4]
        cfg.initial_downsample_conv_layers = 3
        cfg.final_upsample_conv_layers = 2
        cfg.checkpoint_level = 0
        cfg.attn_linear_init_mode = "0"
        cfg.ffn_linear_init_mode = "0"
        cfg.conv_init_mode = "0"
        cfg.down_up_linear_init_mode = "0"
        cfg.norm_init_mode = "0"
        return cfg

    @staticmethod
    def _default_dataset():
        cfg = OmegaConf.create()
        cfg.data_dir = "datasets/SST-PREDICT/"
        cfg.in_len = 14
        cfg.out_len = 3
        return cfg

    # ── Optimizer ──
    def configure_optimizers(self):
        decay_params = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_params = [n for n in decay_params if "bias" not in n]
        groups = [
            {'params': [p for n, p in self.torch_nn_module.named_parameters() if n in decay_params],
             'weight_decay': self.oc.optim.wd},
            {'params': [p for n, p in self.torch_nn_module.named_parameters() if n not in decay_params],
             'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(groups, lr=self.oc.optim.lr, weight_decay=self.oc.optim.wd)

        warmup_steps = int(np.round(self.oc.optim.warmup_percentage * self.total_num_steps))
        warmup = LambdaLR(optimizer, lr_lambda=warmup_lambda(
            warmup_steps=warmup_steps, min_lr_ratio=self.oc.optim.warmup_min_lr_ratio))
        cosine = CosineAnnealingLR(optimizer, T_max=self.total_num_steps - warmup_steps,
                                   eta_min=self.oc.optim.min_lr_ratio * self.oc.optim.lr)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    def _save_hparams(self):
        """Save hparams.json once."""
        import json
        hparams_path = os.path.join(self.save_dir, "hparams.json")
        oc_dict = OmegaConf.to_container(self.oc, resolve=True)
        with open(hparams_path, 'w') as f:
            json.dump(oc_dict, f, indent=2)
        print(f"  Hparams saved: {hparams_path}")

    def on_fit_start(self):
        """Save hyperparameters and CSV header once at training start."""
        self._save_hparams()

        self._csv_path = os.path.join(self.save_dir, "metrics.csv")
        header = "epoch,train_loss,valid_loss,valid_mse,valid_mae,learning_rate\n"
        # Truncate on fresh run (epoch 0); append on resume
        if self.trainer.current_epoch == 0:
            with open(self._csv_path, 'w') as f:
                f.write(header)
        elif not os.path.exists(self._csv_path):
            with open(self._csv_path, 'w') as f:
                f.write(header)
        print(f"  Metrics CSV: {self._csv_path}")

    def on_train_epoch_end(self):
        """Append one row to metrics CSV after each training epoch."""
        if not hasattr(self, '_csv_path'):
            return
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0)
        valid_loss = self.trainer.callback_metrics.get('valid_loss', 0)
        valid_mse = self.trainer.callback_metrics.get('valid_mse_epoch', 0)
        valid_mae = self.trainer.callback_metrics.get('valid_mae_epoch', 0)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        epoch = self.current_epoch

        row = f"{epoch},{float(train_loss):.6f},{float(valid_loss):.6f},{float(valid_mse):.6f},{float(valid_mae):.6f},{lr:.8f}\n"
        with open(self._csv_path, 'a') as f:
            f.write(row)

    # ── Trainer setup ──
    def set_trainer_kwargs(self, **kwargs):
        ckpt_cb = ModelCheckpoint(
            monitor="valid_mse_epoch", dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="model-{epoch:03d}", save_top_k=self.oc.optim.save_top_k,
            save_last=True, mode="min")
        callbacks = kwargs.pop("callbacks", [])
        callbacks += [ckpt_cb, EpochProgressBar()]
        if self.oc.logging.monitor_device:
            callbacks.append(DeviceStatsMonitor())
        if self.oc.optim.early_stop:
            callbacks.append(EarlyStopping(
                monitor="valid_mse_epoch", patience=self.oc.optim.early_stop_patience,
                mode=self.oc.optim.early_stop_mode))

        logger = False   # use custom CSV logging in on_train_epoch_end

        log_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        skip = inspect.signature(Trainer).parameters.keys()
        ret = dict(
            callbacks=callbacks, logger=logger, log_every_n_steps=log_steps,
            default_root_dir=self.save_dir, accelerator="gpu",
            strategy=ApexDDPStrategy(find_unused_parameters=False, delay_allreduce=True) if _HAS_APEX else None,
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            precision=self.oc.trainer.precision)
        ret.update({k: v for k, v in OmegaConf.to_object(self.oc.trainer).items() if k in skip})
        ret.update(kwargs)
        return ret

    @classmethod
    def get_total_num_steps(cls, num_samples, total_batch_size, epoch=None):
        if epoch is None:
            epoch = cls._default_optim().max_epochs
        return int(epoch * num_samples / total_batch_size)

    # ── Forward ──
    def forward(self, X, mask):
        """X: (B, 14, 161, 241, 4) → pred: (B, 3, 161, 241, 1)"""
        return self.torch_nn_module(X)

    def training_step(self, batch, batch_idx):
        X, Y, mask = batch
        pred = self(X, mask)
        B, T = pred.shape[0], pred.shape[1]
        mask_t = mask.reshape(B, 1, mask.shape[1], mask.shape[2], 1)
        loss = ((pred - Y) ** 2 * mask_t).sum() / (mask.sum() * B * T)
        # Entropy regularization to prevent frequency band collapse
        entropy_reg = self.torch_nn_module.freq_branch.entropy_loss(
            self.torch_nn_module._freq_input)
        loss = loss + 1e-4 * entropy_reg
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('entropy_reg', entropy_reg, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        X, Y, mask = batch
        pred = self(X, mask)
        B, T = pred.shape[0], pred.shape[1]
        mask_t = mask.reshape(B, 1, mask.shape[1], mask.shape[2], 1)
        loss = ((pred - Y) ** 2 * mask_t).sum() / (mask.sum() * B * T)

        # Metrics over ocean only
        pred_ocean = pred * mask_t
        Y_ocean = Y * mask_t
        self.valid_mse(pred_ocean, Y_ocean)
        self.valid_mae(pred_ocean, Y_ocean)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_epoch_end(self, outputs):
        self.log('valid_mse_epoch', self.valid_mse.compute(), prog_bar=True)
        self.log('valid_mae_epoch', self.valid_mae.compute(), prog_bar=True)
        self.valid_mse.reset()
        self.valid_mae.reset()

    def on_test_start(self):
        self._save_hparams()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        X, Y, mask = batch
        pred = self(X, mask)
        B, T = pred.shape[0], pred.shape[1]
        mask_t = mask.reshape(B, 1, mask.shape[1], mask.shape[2], 1)
        # Accumulate per-day squared error & absolute error over ocean
        sq_err = ((pred - Y) ** 2 * mask_t).sum(dim=(0,2,3,4))  # (T,)
        abs_err = ((pred - Y).abs() * mask_t).sum(dim=(0,2,3,4))  # (T,)
        n_ocean = mask_t.sum()  # per-sample ocean count (same for all days)
        return {'sq_err': sq_err, 'abs_err': abs_err, 'n_ocean': n_ocean}

    def test_epoch_end(self, outputs):
        # Aggregate across all batches
        sq_err = torch.stack([o['sq_err'] for o in outputs]).sum(dim=0)  # (T,)
        abs_err = torch.stack([o['abs_err'] for o in outputs]).sum(dim=0)  # (T,)
        n_total = sum(o['n_ocean'] for o in outputs)

        mse_per_day = sq_err / n_total  # (T,) in normalized units
        mae_per_day = abs_err / n_total

        # Load normalization stats for degC conversion
        import numpy as np
        stats = np.load(os.path.join(self.oc.dataset.data_dir, "normalization_stats.npz"))
        ssta_std = float(stats['ssta_std'])
        ssta_mean = float(stats['ssta_mean'])

        # Convert to Celsius
        mse_degC = mse_per_day * (ssta_std ** 2)
        rmse_degC = torch.sqrt(mse_degC)
        mae_degC = mae_per_day * ssta_std

        print(f"\n  SSTA std = {ssta_std:.3f}°C (for Celsius conversion)")
        print(f"  {'Day':>6s}  {'MSE( norm )':>12s}  {'MAE( norm )':>12s}  {'RMSE(°C)':>10s}  {'MAE(°C)':>10s}")
        print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}")
        for d in range(len(mse_per_day)):
            print(f"  {d+1:>6d}  {float(mse_per_day[d]):12.6f}  {float(mae_per_day[d]):12.6f}  "
                  f"{float(rmse_degC[d]):10.4f}  {float(mae_degC[d]):10.4f}")
        print(f"  {'avg':>6s}  {float(mse_per_day.mean()):12.6f}  {float(mae_per_day.mean()):12.6f}  "
              f"{float(rmse_degC.mean()):10.4f}  {float(mae_degC.mean()):10.4f}")
        print()

        self.log('test_mse_epoch', mse_per_day.mean(), prog_bar=True)
        self.log('test_mae_epoch', mae_per_day.mean(), prog_bar=True)

        # Save test results CSV
        test_csv = os.path.join(self.save_dir, "test_metrics.csv")
        with open(test_csv, 'w') as f:
            f.write("lead_day,mse_norm,mae_norm,rmse_celsius,mae_celsius\n")
            for d in range(len(mse_per_day)):
                line = (f"{d+1},{float(mse_per_day[d]):.6f},{float(mae_per_day[d]):.6f},"
                        f"{float(rmse_degC[d]):.4f},{float(mae_degC[d]):.4f}\n")
                f.write(line)
            f.write(f"avg,{float(mse_per_day.mean()):.6f},{float(mae_per_day.mean()):.6f},"
                    f"{float(rmse_degC.mean()):.4f},{float(mae_degC.mean()):.4f}\n")
        print(f"  Test results saved: {test_csv}")


# ── Main ──
def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--save', default='nwp_exp1', type=str)
    p.add_argument('--gpus', default=1, type=int)
    p.add_argument('--cfg', default=None, type=str)
    p.add_argument('--data_dir', default='datasets/SST-PREDICT/', type=str)
    p.add_argument('--test', action='store_true')
    p.add_argument('--ckpt_name', default=None, type=str)
    return p


def main():
    args = get_parser().parse_args()

    if args.cfg is not None:
        oc = OmegaConf.load(open(args.cfg, "r"))
        total_bs = oc.optim.total_batch_size
        micro_bs = oc.optim.micro_batch_size
        max_epochs = oc.optim.max_epochs
        seed = oc.optim.seed
    else:
        oc = None
        total_bs = 16
        micro_bs = 2
        max_epochs = 50
        seed = 0

    seed_everything(seed, workers=True)

    # Build dataloaders
    train_loader, val_loader, test_loader, stats = build_dataloaders(
        data_dir=args.data_dir, batch_size=micro_bs, num_workers=4)

    accum = total_bs // (micro_bs * args.gpus)
    n_train_samples = len(train_loader.dataset)
    total_steps = NWPPredictionModule.get_total_num_steps(n_train_samples, total_bs, max_epochs)

    pl_module = NWPPredictionModule(
        total_num_steps=total_steps, oc_file=args.cfg, save_dir=args.save)

    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus, accumulate_grad_batches=accum)
    trainer = Trainer(**trainer_kwargs)

    if args.test:
        assert args.ckpt_name is not None, "--ckpt_name required for test"
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        trainer.test(model=pl_module, dataloaders=test_loader, ckpt_path=ckpt_path)
    else:
        ckpt_path = None
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            if not os.path.exists(ckpt_path):
                warnings.warn(f"ckpt {ckpt_path} not found. Starting fresh.")
                ckpt_path = None
        trainer.fit(model=pl_module, train_dataloaders=train_loader,
                    val_dataloaders=val_loader, ckpt_path=ckpt_path)
        # Save best as plain state_dict
        best_path = trainer.checkpoint_callback.best_model_path
        if best_path:
            sd = pl_ckpt_to_pytorch_state_dict(
                best_path, map_location=torch.device("cpu"),
                delete_prefix_len=len("torch_nn_module."))
            torch.save(sd, os.path.join(pl_module.save_dir, "checkpoints", "best_model.pt"))
        trainer.test(dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
