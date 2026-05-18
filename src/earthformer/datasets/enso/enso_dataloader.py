"""ENSO multi-variable dataloader with preprocessing cache.

First run: reads raw .nc, transforms, normalizes, saves .npy cache (~30s–2min).
Later runs: loads .npy cache directly (<1s).
"""
import os
import pickle
from typing import Optional, List
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from pytorch_lightning import LightningDataModule
from ...config import cfg


NINO_WINDOW_T = 3
DEFAULT_VARS = ['sst', 't300', 'ua', 'va']
default_data_dir = os.path.join(cfg.datasets_dir, "enso_multivar")


def prepare_inputs_targets(len_time, input_gap, input_length, pred_shift, pred_length, samples_gap):
    assert pred_shift >= pred_length
    input_span = input_gap * (input_length - 1) + 1
    pred_gap = pred_shift // pred_length
    input_ind = np.arange(0, input_span, input_gap)
    target_ind = np.arange(0, pred_shift, pred_gap) + input_span + pred_gap - 1
    ind = np.concatenate([input_ind, target_ind]).reshape(1, input_length + pred_length)
    max_n_sample = len_time - (input_span + pred_shift - 1)
    ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones((1, input_length + pred_length), dtype=int)
    return ind[::samples_gap]


def find_nino_indices(lat_vals, lon_vals):
    lat_min, lat_max = -5.0, 5.0
    lon_min, lon_max = 190.0, 240.0
    lat_idx = np.where((lat_vals >= lat_min) & (lat_vals <= lat_max))[0]
    lon_idx = np.where((lon_vals >= lon_min) & (lon_vals <= lon_max))[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(
            f"Nino 3.4 region not found! "
            f"lat [{lat_vals.min():.1f}, {lat_vals.max():.1f}], "
            f"lon [{lon_vals.min():.1f}, {lon_vals.max():.1f}]")
    lat_slice = slice(lat_idx[0], lat_idx[-1] + 1)
    lon_slice = slice(lon_idx[0], lon_idx[-1] + 1)
    print(f"Nino 3.4 region: lat[{lat_slice}] (values {lat_vals[lat_idx]}), "
          f"lon[{lon_slice}] (values {lon_vals[lon_idx]})")
    return lat_slice, lon_slice


def read_cmip_multivar(data_dir, cmip6_cutoff=2265, cmip6_ypm=151, cmip5_ypm=140):
    """Load preprocessed CMIP data from cache. Run preprocess_enso.py first."""
    data_dir = str(data_dir)
    cache_file = os.path.join(data_dir, ".cache_cmip_all.npz")
    meta_file = os.path.join(data_dir, ".cache_cmip_meta.pkl")

    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"CMIP cache not found: {cache_file}\n"
            f"Run: python scripts/datasets/preprocess_enso.py --data_dir {data_dir}")

    print("Loading CMIP cache...")
    data = np.load(cache_file, mmap_mode='r')
    cmip6_data = data['cmip6_data']
    cmip5_data = data['cmip5_data']
    cmip6_nino = data['cmip6_nino']
    cmip5_nino = data['cmip5_nino']
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    var_stats = meta['var_stats']
    lat_vals = meta['lat_vals']
    lon_vals = meta['lon_vals']

    nino_lat_slice, nino_lon_slice = find_nino_indices(lat_vals, lon_vals)
    print(f"  CMIP6: {cmip6_data.shape}, CMIP5: {cmip5_data.shape}")
    return cmip6_data, cmip5_data, cmip6_nino, cmip5_nino, var_stats, nino_lat_slice, nino_lon_slice


def read_soda_multivar(data_dir, var_stats=None):
    """Load preprocessed SODA data from cache. Run preprocess_enso.py first."""
    data_dir = str(data_dir)
    cache_file = os.path.join(data_dir, ".cache_soda.npz")
    meta_file = os.path.join(data_dir, ".cache_soda_meta.pkl")

    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"SODA cache not found: {cache_file}\n"
            f"Run: python scripts/datasets/preprocess_enso.py --data_dir {data_dir}")

    print("Loading SODA cache...")
    data = np.load(cache_file, mmap_mode='r')
    soda_data = data['soda_data']
    soda_nino = data['soda_nino']
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    lat_vals = meta['lat_vals']
    lon_vals = meta['lon_vals']

    nino_lat_slice, nino_lon_slice = find_nino_indices(lat_vals, lon_vals)
    print(f"  SODA: {soda_data.shape}")
    return soda_data, soda_nino, nino_lat_slice, nino_lon_slice


# ─── Lazy Datasets ───

class MultivarCMIPDataset(Dataset):
    def __init__(self, cmip_data, cmip_nino, samples_gap,
                 in_len=12, out_len=26, in_stride=1, out_stride=1,
                 var_stats=None, var_names=None):
        super().__init__()
        self.data = cmip_data
        self.nino = cmip_nino
        self.nino_idx = slice(in_len, in_len + out_len - NINO_WINDOW_T + 1)
        self.idx_seq = prepare_inputs_targets(
            len_time=cmip_data.shape[0],
            input_length=in_len, input_gap=in_stride,
            pred_shift=out_len * out_stride, pred_length=out_len,
            samples_gap=samples_gap)

    def __len__(self):
        return self.idx_seq.shape[0]

    def __getitem__(self, idx):
        seq_idx = self.idx_seq[idx]
        x = np.array(self.data[seq_idx])
        y = np.array(self.nino[seq_idx[self.nino_idx]])
        return torch.from_numpy(x), torch.from_numpy(y)


class MultivarSODADataset(Dataset):
    def __init__(self, soda_data, soda_nino, samples_gap,
                 in_len=12, out_len=26, in_stride=1, out_stride=1,
                 var_stats=None, var_names=None):
        super().__init__()
        self.data = soda_data
        self.nino = soda_nino
        self.nino_idx = slice(in_len, in_len + out_len - NINO_WINDOW_T + 1)
        self.idx_seq = prepare_inputs_targets(
            len_time=soda_data.shape[0],
            input_length=in_len, input_gap=in_stride,
            pred_shift=out_len * out_stride, pred_length=out_len,
            samples_gap=samples_gap)

    def __len__(self):
        return self.idx_seq.shape[0]

    def __getitem__(self, idx):
        seq_idx = self.idx_seq[idx]
        x = np.array(self.data[seq_idx])
        y = np.array(self.nino[seq_idx[self.nino_idx]])
        return torch.from_numpy(x), torch.from_numpy(y)


# ─── LightningDataModule ───

class ENSOLightningDataModule(LightningDataModule):

    def __init__(self, data_dir=None, in_len=12, out_len=26,
                 in_stride=1, out_stride=1,
                 train_samples_gap=1, eval_samples_gap=1,
                 cmip6_cutoff=2265, cmip6_years_per_model=151, cmip5_years_per_model=140,
                 soda_val_ratio=0.5, batch_size=1, num_workers=1, var_names=None):
        super().__init__()
        if data_dir is None:
            data_dir = default_data_dir
        self.data_dir = data_dir
        self.in_len = in_len
        self.out_len = out_len
        self.in_stride = in_stride
        self.out_stride = out_stride
        self.train_samples_gap = train_samples_gap
        self.eval_samples_gap = eval_samples_gap
        self.cmip6_cutoff = cmip6_cutoff
        self.cmip6_years_per_model = cmip6_years_per_model
        self.cmip5_years_per_model = cmip5_years_per_model
        self.soda_val_ratio = soda_val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.var_names = var_names if var_names else DEFAULT_VARS

    def prepare_data(self):
        for fname in ['CMIP_train.nc', 'CMIP_label.nc', 'SODA_train.nc', 'SODA_label.nc']:
            if not os.path.exists(os.path.join(self.data_dir, fname)):
                raise FileNotFoundError(f"{os.path.join(self.data_dir, fname)} not found!")

    def setup(self, stage: Optional[str] = None):
        cmip6_data, cmip5_data, cmip6_nino, cmip5_nino, var_stats, \
            nino_lat_slice, nino_lon_slice = \
            read_cmip_multivar(self.data_dir, self.cmip6_cutoff,
                               self.cmip6_years_per_model, self.cmip5_years_per_model)
        self.var_stats = var_stats
        self.nino_lat_slice = nino_lat_slice
        self.nino_lon_slice = nino_lon_slice

        soda_data, soda_nino, _, _ = read_soda_multivar(self.data_dir, var_stats=var_stats)

        if stage == "fit" or stage is None:
            cmip_train = np.concatenate([cmip6_data, cmip5_data], axis=0)
            nino_train = np.concatenate([cmip6_nino, cmip5_nino], axis=0)
            self.enso_train = MultivarCMIPDataset(
                cmip_train, nino_train,
                samples_gap=self.train_samples_gap,
                in_len=self.in_len, out_len=self.out_len,
                in_stride=self.in_stride, out_stride=self.out_stride)
            self._setup_soda_val_test(soda_data, soda_nino)

        if stage == "test" or stage is None:
            if not hasattr(self, 'enso_test'):
                self._setup_soda_val_test(soda_data, soda_nino)

        if stage == "predict":
            self.enso_predict = MultivarSODADataset(
                soda_data, soda_nino,
                samples_gap=self.eval_samples_gap,
                in_len=self.in_len, out_len=self.out_len,
                in_stride=self.in_stride, out_stride=self.out_stride)

    def _setup_soda_val_test(self, soda_data, soda_nino):
        soda_full = MultivarSODADataset(
            soda_data, soda_nino,
            samples_gap=self.eval_samples_gap,
            in_len=self.in_len, out_len=self.out_len,
            in_stride=self.in_stride, out_stride=self.out_stride)
        n_total = len(soda_full)
        n_val = int(n_total * self.soda_val_ratio)
        indices = np.arange(n_total)
        self.enso_val = torch.utils.data.Subset(soda_full, indices[:n_val])
        self.enso_test = torch.utils.data.Subset(soda_full, indices[n_val:])

    @property
    def num_train_samples(self):
        return len(self.enso_train)

    @property
    def num_val_samples(self):
        return len(self.enso_val)

    @property
    def num_test_samples(self):
        return len(self.enso_test)

    def train_dataloader(self):
        return DataLoader(self.enso_train, shuffle=True,
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.enso_val, shuffle=False,
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.enso_test, shuffle=False,
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.enso_predict, shuffle=False,
                          batch_size=self.batch_size, num_workers=self.num_workers)
