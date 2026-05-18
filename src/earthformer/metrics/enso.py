"""ENSO/SST prediction metrics."""
from typing import Tuple, Optional, Union
import numpy as np
import torch
from torchmetrics import Metric
from ..datasets.enso.enso_dataloader import NINO_WINDOW_T


def compute_enso_score(y_pred, y_true,
                       acc_weight: Optional[Union[str, np.ndarray, torch.Tensor]] = None):
    pred = y_pred - y_pred.mean(dim=0, keepdim=True)
    true = y_true - y_true.mean(dim=0, keepdim=True)
    cor = (pred * true).sum(dim=0) / (torch.sqrt(torch.sum(pred ** 2, dim=0) * torch.sum(true ** 2, dim=0)) + 1e-6)

    if acc_weight is None:
        acc = cor.sum()
    else:
        nino_out_len = y_true.shape[-1]
        if acc_weight == "default":
            acc_weight = torch.tensor([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * (nino_out_len - 18))[:nino_out_len] \
                         * torch.log(torch.arange(nino_out_len) + 1)
        elif isinstance(acc_weight, np.ndarray):
            acc_weight = torch.from_numpy(acc_weight[:nino_out_len])
        elif isinstance(acc_weight, torch.Tensor):
            acc_weight = acc_weight[:nino_out_len]
        acc_weight = acc_weight.to(y_pred)
        acc = (acc_weight * cor).sum()
    rmse = torch.mean((y_pred - y_true) ** 2, dim=0).sqrt().sum()
    return acc, rmse


def sst_to_nino(sst: torch.Tensor,
                lat_slice: slice = slice(10, 13),
                lon_slice: slice = slice(19, 30),
                detach: bool = True):
    """Convert SST predictions to Niño 3.4 index.

    Parameters
    ----------
    sst: torch.Tensor, shape (N, T, H, W)
    lat_slice, lon_slice: slices for Niño 3.4 region
    """
    if detach:
        nino_index = sst.detach()
    else:
        nino_index = sst
    nino_index = nino_index[:, :, lat_slice, lon_slice].mean(dim=[2, 3])
    nino_index = nino_index.unfold(dimension=1, size=NINO_WINDOW_T, step=1).mean(dim=2)
    return nino_index


class ENSOScore(Metric):

    def __init__(self,
                 layout="NTHWC",
                 out_len=26,
                 lat_slice: slice = slice(10, 13),
                 lon_slice: slice = slice(19, 30)):
        super().__init__()
        self.layout = layout
        self.out_len = out_len
        self.lat_slice = lat_slice
        self.lon_slice = lon_slice
        self.nino_out_len = out_len - NINO_WINDOW_T + 1
        self.nino_weight = torch.from_numpy(
            np.array([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * (self.nino_out_len - 18))
            * np.log(np.arange(self.nino_out_len) + 1))

        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_pixels", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("nino_preds", default=[], dist_reduce_fx="cat")
        self.add_state("nino_target", default=[], dist_reduce_fx="cat")

    def update(self, preds, target, nino_preds=None, nino_target=None):
        if self.layout.endswith("C"):
            sst_preds = preds[..., 0]
            sst_target = target[..., 0]
        else:
            sst_preds = preds
            sst_target = target

        diff = sst_preds - sst_target
        self.sum_squared_error += torch.sum(diff * diff)
        self.num_pixels += sst_target.numel()

        if nino_preds is None:
            nino_preds = sst_to_nino(sst=sst_preds, lat_slice=self.lat_slice, lon_slice=self.lon_slice)
        if nino_target is None:
            nino_target = sst_to_nino(sst=sst_target, lat_slice=self.lat_slice, lon_slice=self.lon_slice)
        self.nino_preds.extend([ele for ele in nino_preds])
        self.nino_target.extend([ele for ele in nino_target])

    def compute(self) -> Tuple[float, float]:
        mse = self.sum_squared_error / self.num_pixels
        y_pred = torch.stack(self.nino_preds, dim=0)
        y_true = torch.stack(self.nino_target, dim=0)
        acc, nino_rmse = compute_enso_score(y_pred=y_pred, y_true=y_true, acc_weight=self.nino_weight)
        return acc.cpu().item(), mse.cpu().item()
