#!/usr/bin/env python
"""Train Earthformer for multi-variable ENSO/SST prediction.

Data: CMIP_train.nc + CMIP_label.nc (train), SODA_train.nc + SODA_label.nc (val/test)
Variables: sst, t300, ua, va
Grid: auto-detected from data
"""


""" 

TEST
python scripts/cuboid_transformer/enso/train_cuboid_enso.py \
     --gpus 1 --test --save enso_test \
    --data_dir ./datasets/enso_multivar/ \
    --ckpt_name last.ckpt
    
TRAIN   
python scripts/cuboid_transformer/enso/train_cuboid_enso.py \
    --gpus 1 --save enso_test \
    --data_dir ./datasets/enso_multivar/ \
    --cfg scripts/cuboid_transformer/enso/cfg.yaml   
    
  
Continue
python scripts/cuboid_transformer/enso/train_cuboid_enso.py \
    --gpus 1 --save enso_test \
    --data_dir ./datasets/enso_multivar/ \
    --cfg scripts/cuboid_transformer/enso/cfg.yaml \
    --ckpt_name last.ckpt
  
"""
import warnings
from typing import Sequence
from shutil import copyfile
import inspect
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

# PyTorch 2.6+ weights_only fix — PL 1.6.4 ckpt has nested OmegaConf types
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, Callback
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
import os
import argparse
from earthformer.config import cfg
from earthformer.utils.optim import SequentialLR, warmup_lambda
from earthformer.utils.utils import get_parameter_names
from earthformer.utils.checkpoint import pl_ckpt_to_pytorch_state_dict, s3_download_pretrained_ckpt
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from earthformer.datasets.enso.enso_dataloader import (
    ENSOLightningDataModule, NINO_WINDOW_T, DEFAULT_VARS)
from earthformer.metrics.enso import sst_to_nino, compute_enso_score
try:
    import apex
    from earthformer.utils.apex_ddp import ApexDDPStrategy
    _HAS_APEX = True
except ImportError:
    _HAS_APEX = False


class EpochProgressBar(TQDMProgressBar):
    """TQDM bar showing Epoch X/Y."""
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        if self.main_progress_bar is not None:
            self.main_progress_bar.set_description(
                f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}", refresh=True)


_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
exps_dir = os.path.join(_curr_dir, "experiments")
pretrained_checkpoints_dir = cfg.pretrained_checkpoints_dir
pytorch_state_dict_name = "earthformer_enso_multivar.pt"


def _get_grid_info(data_dir, dataset_cfg):
    """Quick peek at data to get grid dimensions and Niño region slices."""
    import xarray as xr
    from pathlib import Path
    from earthformer.datasets.enso.enso_dataloader import find_nino_indices
    import numpy as np

    ds = xr.open_dataset(Path(data_dir) / 'CMIP_train.nc')
    lat_vals = ds.lat.values
    lon_vals = ds.lon.values
    lon_mask = np.logical_and(lon_vals >= 95, lon_vals <= 330)
    lon_vals_filt = lon_vals[lon_mask]
    ds.close()

    n_lat = len(lat_vals)
    n_lon = len(lon_vals_filt)
    nino_lat_slice, nino_lon_slice = find_nino_indices(lat_vals, lon_vals_filt)

    print(f"Grid info: lat={n_lat}, lon={n_lon} "
          f"(after 95E–330E filter, from {len(lon_vals)} original)")
    print(f"Niño 3.4 region: lat[{nino_lat_slice}], lon[{nino_lon_slice}]")

    return n_lat, n_lon, nino_lat_slice, nino_lon_slice


class CuboidENSOPLModule(pl.LightningModule):

    def __init__(self,
                 total_num_steps: int,
                 oc_file: str = None,
                 save_dir: str = None,
                 grid_info: dict = None):
        super().__init__()
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None

        # Override spatial dims from grid_info
        if grid_info is not None:
            base_oc = self.get_base_config()
            base_oc.model.input_shape = (
                base_oc.layout.in_len,
                grid_info['lat'],
                grid_info['lon'],
                base_oc.model.data_channels)
            base_oc.model.target_shape = (
                base_oc.layout.out_len,
                grid_info['lat'],
                grid_info['lon'],
                base_oc.model.data_channels)
            if oc_from_file is not None:
                oc = OmegaConf.merge(base_oc, oc_from_file)
            else:
                oc = base_oc
        else:
            oc = self.get_base_config(oc_from_file=oc_from_file)

        model_cfg = OmegaConf.to_object(oc.model)
        num_blocks = len(model_cfg["enc_depth"])

        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])

        self.torch_nn_module = CuboidTransformerModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
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
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            initial_downsample_scale=model_cfg["initial_downsample_scale"],
            initial_downsample_conv_layers=model_cfg["initial_downsample_conv_layers"],
            final_upsample_conv_layers=model_cfg["final_upsample_conv_layers"],
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )

        self.total_num_steps = total_num_steps
        self.save_hyperparameters(oc)
        self.oc = oc
        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout
        self.channel_axis = self.layout.find("C")
        self.batch_axis = self.layout.find("N")
        self.channels = model_cfg["data_channels"]
        self.max_epochs = oc.optim.max_epochs
        self.total_num_steps = total_num_steps
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix

        # Niño slices from datamodule (set after setup)
        self._nino_lat_slice = grid_info.get('nino_lat_slice', slice(10, 13)) if grid_info else slice(10, 13)
        self._nino_lon_slice = grid_info.get('nino_lon_slice', slice(19, 30)) if grid_info else slice(19, 30)

        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

        self.configure_save(cfg_file_path=oc_file)

    def configure_save(self, cfg_file_path=None):
        self.save_dir = os.path.join(exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.scores_dir = os.path.join(self.save_dir, 'scores')
        os.makedirs(self.scores_dir, exist_ok=True)
        if cfg_file_path is not None:
            cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
            if (not os.path.exists(cfg_file_target_path)) or \
                    (not os.path.samefile(cfg_file_path, cfg_file_target_path)):
                copyfile(cfg_file_path, cfg_file_target_path)
        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)

    def get_base_config(self, oc_from_file=None):
        oc = OmegaConf.create()
        oc.layout = self.get_layout_config()
        oc.optim = self.get_optim_config()
        oc.logging = self.get_logging_config()
        oc.trainer = self.get_trainer_config()
        oc.vis = self.get_vis_config()
        oc.model = self.get_model_config()
        oc.dataset = self.get_dataset_config()
        if oc_from_file is not None:
            oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    @staticmethod
    def get_layout_config():
        cfg = OmegaConf.create()
        cfg.in_len = 12
        cfg.out_len = 26
        cfg.img_height = 24   # detected from actual data
        cfg.img_width = 48    # overridden from actual data
        cfg.layout = "NTHWC"
        return cfg

    @classmethod
    def get_model_config(cls):
        layout_cfg = cls.get_layout_config()
        cfg = OmegaConf.create()
        cfg.data_channels = 4
        cfg.input_shape = (layout_cfg.in_len, layout_cfg.img_height, layout_cfg.img_width, cfg.data_channels)
        cfg.target_shape = (layout_cfg.out_len, layout_cfg.img_height, layout_cfg.img_width, cfg.data_channels)
        cfg.base_units = 64
        cfg.block_units = None
        cfg.scale_alpha = 1.0
        cfg.enc_depth = [1, 1]
        cfg.dec_depth = [1, 1]
        cfg.enc_use_inter_ffn = True
        cfg.dec_use_inter_ffn = True
        cfg.dec_hierarchical_pos_embed = False
        cfg.downsample = 2
        cfg.downsample_type = "patch_merge"
        cfg.upsample_type = "upsample"
        cfg.num_global_vectors = 0
        cfg.use_dec_self_global = False
        cfg.dec_self_update_global = True
        cfg.use_dec_cross_global = False
        cfg.use_global_vector_ffn = False
        cfg.use_global_self_attn = False
        cfg.separate_global_qkv = False
        cfg.global_dim_ratio = 1
        cfg.self_pattern = 'axial'
        cfg.cross_self_pattern = 'axial'
        cfg.cross_pattern = 'cross_1x1'
        cfg.dec_cross_last_n_frames = None
        cfg.attn_drop = 0.1
        cfg.proj_drop = 0.1
        cfg.ffn_drop = 0.1
        cfg.num_heads = 4
        cfg.ffn_activation = 'gelu'
        cfg.gated_ffn = False
        cfg.norm_layer = 'layer_norm'
        cfg.padding_type = 'zeros'
        cfg.pos_embed_type = "t+h+w"
        cfg.use_relative_pos = True
        cfg.self_attn_use_final_proj = True
        cfg.dec_use_first_self_attn = False
        cfg.z_init_method = 'zeros'
        cfg.initial_downsample_type = "conv"
        cfg.initial_downsample_activation = "leaky"
        cfg.initial_downsample_scale = (1, 1, 2)
        cfg.initial_downsample_conv_layers = 2
        cfg.final_upsample_conv_layers = 1
        cfg.checkpoint_level = 0
        cfg.attn_linear_init_mode = "0"
        cfg.ffn_linear_init_mode = "0"
        cfg.conv_init_mode = "0"
        cfg.down_up_linear_init_mode = "0"
        cfg.norm_init_mode = "0"
        return cfg

    @classmethod
    def get_dataset_config(cls):
        cfg = OmegaConf.create()
        cfg.in_len = 12
        cfg.out_len = 26
        cfg.in_stride = 1
        cfg.out_stride = 1
        cfg.train_samples_gap = 1
        cfg.eval_samples_gap = 1
        cfg.cmip6_cutoff = 2265
        cfg.cmip6_years_per_model = 151
        cfg.cmip5_years_per_model = 140
        cfg.soda_val_ratio = 0.5
        cfg.var_names = ['sst', 't300', 'ua', 'va']
        return cfg

    @staticmethod
    def get_optim_config():
        cfg = OmegaConf.create()
        cfg.seed = 0
        cfg.total_batch_size = 64
        cfg.micro_batch_size = 8
        cfg.method = "adamw"
        cfg.lr = 1E-4
        cfg.wd = 1E-5
        cfg.gradient_clip_val = 1.0
        cfg.max_epochs = 100
        cfg.warmup_percentage = 0.2
        cfg.lr_scheduler_mode = "cosine"
        cfg.min_lr_ratio = 1.0e-3
        cfg.warmup_min_lr_ratio = 0.0
        cfg.early_stop = True
        cfg.early_stop_mode = "min"
        cfg.early_stop_patience = 5
        cfg.save_top_k = 5
        return cfg

    @staticmethod
    def get_logging_config():
        cfg = OmegaConf.create()
        cfg.logging_prefix = "Cuboid_ENSO_MultiVar"
        cfg.monitor_lr = True
        cfg.monitor_device = False
        cfg.track_grad_norm = -1
        cfg.use_wandb = False
        return cfg

    @staticmethod
    def get_trainer_config():
        cfg = OmegaConf.create()
        cfg.check_val_every_n_epoch = 5
        cfg.log_step_ratio = 0.001
        cfg.precision = 32
        return cfg

    @staticmethod
    def get_vis_config():
        cfg = OmegaConf.create()
        cfg.train_example_data_idx_list = [0]
        cfg.val_example_data_idx_list = [0]
        cfg.test_example_data_idx_list = [0]
        cfg.eval_example_only = False
        return cfg

    def configure_optimizers(self):
        decay_parameters = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [{
            'params': [p for n, p in self.torch_nn_module.named_parameters() if n in decay_parameters],
            'weight_decay': self.oc.optim.wd
        }, {
            'params': [p for n, p in self.torch_nn_module.named_parameters() if n not in decay_parameters],
            'weight_decay': 0.0
        }]

        if self.oc.optim.method == 'adamw':
            optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                          lr=self.oc.optim.lr,
                                          weight_decay=self.oc.optim.wd)
        else:
            raise NotImplementedError

        warmup_iter = int(np.round(self.oc.optim.warmup_percentage * self.total_num_steps))

        if self.oc.optim.lr_scheduler_mode == 'cosine':
            warmup_scheduler = LambdaLR(optimizer,
                                        lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                                                min_lr_ratio=self.oc.optim.warmup_min_lr_ratio))
            cosine_scheduler = CosineAnnealingLR(optimizer,
                                                 T_max=(self.total_num_steps - warmup_iter),
                                                 eta_min=self.oc.optim.min_lr_ratio * self.oc.optim.lr)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                        milestones=[warmup_iter])
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1}}
        raise NotImplementedError

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    def set_trainer_kwargs(self, **kwargs):
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_sst_mse_epoch",
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="model-{epoch:03d}",
            save_top_k=self.oc.optim.save_top_k,
            save_last=True, mode="min")
        callbacks = kwargs.pop("callbacks", [])
        callbacks += [checkpoint_callback, EpochProgressBar()]
        if self.oc.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step')]
        if self.oc.logging.monitor_device:
            callbacks += [DeviceStatsMonitor()]
        if self.oc.optim.early_stop:
            callbacks += [EarlyStopping(monitor="valid_sst_mse_epoch",
                                        patience=self.oc.optim.early_stop_patience,
                                        mode=self.oc.optim.early_stop_mode)]

        logger = kwargs.pop("logger", [])
        logger += [pl_loggers.TensorBoardLogger(save_dir=self.save_dir),
                   pl_loggers.CSVLogger(save_dir=self.save_dir)]
        if self.oc.logging.use_wandb:
            logger += [pl_loggers.WandbLogger(project=self.oc.logging.logging_prefix, save_dir=self.save_dir)]

        log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        trainer_init_keys = inspect.signature(Trainer).parameters.keys()
        ret = dict(
            callbacks=callbacks, logger=logger,
            log_every_n_steps=log_every_n_steps,
            track_grad_norm=self.oc.logging.track_grad_norm,
            default_root_dir=self.save_dir,
            accelerator="gpu",
            strategy=ApexDDPStrategy(find_unused_parameters=False, delay_allreduce=True) if _HAS_APEX else None,
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            precision=self.oc.trainer.precision)
        oc_trainer_kwargs = {k: v for k, v in OmegaConf.to_object(self.oc.trainer).items() if k in trainer_init_keys}
        ret.update(oc_trainer_kwargs)
        ret.update(kwargs)
        return ret

    @classmethod
    def get_total_num_steps(cls, num_samples, total_batch_size, epoch=None):
        if epoch is None:
            epoch = cls.get_optim_config().max_epochs
        return int(epoch * num_samples / total_batch_size)

    @staticmethod
    def get_enso_datamodule(dataset_cfg, micro_batch_size=1, num_workers=1):
        return ENSOLightningDataModule(
            data_dir=dataset_cfg.get("data_dir"),
            in_len=dataset_cfg["in_len"], out_len=dataset_cfg["out_len"],
            in_stride=dataset_cfg["in_stride"], out_stride=dataset_cfg["out_stride"],
            train_samples_gap=dataset_cfg.get("train_samples_gap", 1),
            eval_samples_gap=dataset_cfg.get("eval_samples_gap", 1),
            cmip6_cutoff=dataset_cfg.get("cmip6_cutoff", 2265),
            cmip6_years_per_model=dataset_cfg.get("cmip6_years_per_model", 151),
            cmip5_years_per_model=dataset_cfg.get("cmip5_years_per_model", 140),
            soda_val_ratio=dataset_cfg.get("soda_val_ratio", 0.5),
            batch_size=micro_batch_size, num_workers=num_workers,
            var_names=dataset_cfg.get("var_names", DEFAULT_VARS))

    @property
    def nino_out_len(self):
        return self.out_len - NINO_WINDOW_T + 1

    def forward(self, batch):
        data_seq, nino_target = batch
        data_seq = data_seq.float()
        in_seq = data_seq[:, :self.in_len, ...]
        target_seq = data_seq[:, self.in_len:self.in_len + self.out_len, ...]
        pred_seq = self.torch_nn_module(in_seq)
        loss = F.mse_loss(pred_seq, target_seq)
        return pred_seq, loss, in_seq, target_seq, nino_target.float()

    def training_step(self, batch, batch_idx):
        pred_seq, loss, in_seq, target_seq, nino_target = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pred_seq, loss, in_seq, target_seq, nino_target = self(batch)
        if self.precision == 16:
            pred_seq = pred_seq.float()
        sst_pred = pred_seq[..., 0:1].contiguous()
        sst_target = target_seq[..., 0:1].contiguous()
        self.valid_mse(sst_pred, sst_target)
        self.valid_mae(sst_pred, sst_target)
        nino_preds = sst_to_nino(sst=pred_seq[..., 0],
                                 lat_slice=self._nino_lat_slice,
                                 lon_slice=self._nino_lon_slice)
        return nino_preds, nino_target

    def validation_epoch_end(self, outputs):
        valid_sst_mse = self.valid_mse.compute()
        valid_sst_mae = self.valid_mae.compute()
        nino_preds_list, nino_target_list = map(list, zip(*outputs))
        nino_preds_list = torch.cat(nino_preds_list, dim=0)
        nino_target_list = torch.cat(nino_target_list, dim=0)
        valid_acc, valid_nino_rmse = compute_enso_score(nino_preds_list, nino_target_list, acc_weight=None)
        valid_weighted_acc, _ = compute_enso_score(nino_preds_list, nino_target_list, acc_weight="default")
        valid_acc /= self.nino_out_len
        valid_nino_rmse /= self.nino_out_len
        valid_weighted_acc /= self.nino_out_len

        self.log('valid_sst_mse_epoch', valid_sst_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_sst_mae_epoch', valid_sst_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_corr_nino3.4_epoch', valid_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_corr_nino3.4_weighted_epoch', valid_weighted_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_nino_rmse_epoch', valid_nino_rmse, prog_bar=True, on_step=False, on_epoch=True)
        self.valid_mse.reset()
        self.valid_mae.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pred_seq, loss, in_seq, target_seq, nino_target = self(batch)
        if self.precision == 16:
            pred_seq = pred_seq.float()
        sst_pred = pred_seq[..., 0:1].contiguous()
        sst_target = target_seq[..., 0:1].contiguous()
        self.test_mse(sst_pred, sst_target)
        self.test_mae(sst_pred, sst_target)
        nino_preds = sst_to_nino(sst=pred_seq[..., 0],
                                 lat_slice=self._nino_lat_slice,
                                 lon_slice=self._nino_lon_slice)
        return nino_preds, nino_target

    def test_epoch_end(self, outputs):
        test_sst_mse = self.test_mse.compute()
        test_sst_mae = self.test_mae.compute()
        nino_preds_list, nino_target_list = map(list, zip(*outputs))
        nino_preds_list = torch.cat(nino_preds_list, dim=0)
        nino_target_list = torch.cat(nino_target_list, dim=0)

        # Per-lead-month correlation
        pred = nino_preds_list - nino_preds_list.mean(dim=0, keepdim=True)
        true = nino_target_list - nino_target_list.mean(dim=0, keepdim=True)
        cor_per_lead = (pred * true).sum(dim=0) / (
            torch.sqrt(torch.sum(pred**2, dim=0) * torch.sum(true**2, dim=0)) + 1e-6)

        test_acc, test_nino_rmse = compute_enso_score(nino_preds_list, nino_target_list, acc_weight=None)
        test_weighted_acc, _ = compute_enso_score(nino_preds_list, nino_target_list, acc_weight="default")
        test_acc /= self.nino_out_len
        test_nino_rmse /= self.nino_out_len
        test_weighted_acc /= self.nino_out_len

        # Print per-lead-month correlations
        print("\nNino3.4 correlation per lead month:")
        for i in range(self.nino_out_len):
            print(f"  lead {i+1:2d} month:  {cor_per_lead[i].item():+.4f}")
        print()

        self.log('test_sst_mse_epoch', test_sst_mse, prog_bar=True)
        self.log('test_sst_mae_epoch', test_sst_mae, prog_bar=True)
        self.log('test_corr_nino3.4_epoch', test_acc, prog_bar=True)
        self.log('test_corr_nino3.4_weighted_epoch', test_weighted_acc, prog_bar=True)
        self.log('test_nino_rmse_epoch', test_nino_rmse, prog_bar=True)
        self.test_mse.reset()
        self.test_mae.reset()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_enso', type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--ckpt_name', default=None, type=str)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        dataset_cfg = OmegaConf.to_object(oc_from_file.dataset)
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
    else:
        dataset_cfg = OmegaConf.to_object(CuboidENSOPLModule.get_dataset_config())
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * args.gpus)
        max_epochs = None
        seed = 0
        oc_from_file = None

    if args.data_dir is not None:
        dataset_cfg["data_dir"] = args.data_dir
    data_dir = dataset_cfg.get("data_dir", "datasets/enso_multivar")

    # Step 1: peek at data to get grid dimensions
    n_lat, n_lon, nino_lat_slice, nino_lon_slice = _get_grid_info(data_dir, dataset_cfg)
    grid_info = {
        'lat': n_lat,
        'lon': n_lon,
        'nino_lat_slice': nino_lat_slice,
        'nino_lon_slice': nino_lon_slice,
    }

    seed_everything(seed, workers=True)

    # Step 2: create datamodule
    dm = CuboidENSOPLModule.get_enso_datamodule(
        dataset_cfg=dataset_cfg,
        micro_batch_size=micro_batch_size,
        num_workers=1)
    dm.prepare_data()
    dm.setup()

    accumulate_grad_batches = total_batch_size // (micro_batch_size * args.gpus)
    total_num_steps = CuboidENSOPLModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size)

    # Step 3: create model with correct grid dims
    pl_module = CuboidENSOPLModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg,
        grid_info=grid_info)

    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
        accumulate_grad_batches=accumulate_grad_batches)
    trainer = Trainer(**trainer_kwargs)

    if args.test:
        assert args.ckpt_name is not None, "args.ckpt_name required!"
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        trainer.test(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)
    else:
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            if not os.path.exists(ckpt_path):
                warnings.warn(f"ckpt {ckpt_path} not found. Starting from epoch 0.")
                ckpt_path = None
        else:
            ckpt_path = None
        trainer.fit(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)
        state_dict = pl_ckpt_to_pytorch_state_dict(
            checkpoint_path=trainer.checkpoint_callback.best_model_path,
            map_location=torch.device("cpu"),
            delete_prefix_len=len("torch_nn_module."))
        torch.save(state_dict, os.path.join(pl_module.save_dir, "checkpoints", pytorch_state_dict_name))
        trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    main()
