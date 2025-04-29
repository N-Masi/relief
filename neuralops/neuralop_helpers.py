import torch
import pytorch_lightning as pl
import numpy as np
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf
from neuralop.losses.data_losses import LpLoss
import xarray as xr
import zarr
import cfgrib
import gcsfs
import fsspec
import pickle
import hashlib
import json
import os
import pdb
from utils.xarray_funcs import *

class NeuralOpWrapper(pl.LightningModule):
    def __init__(self, model, optimizer_cfg, scheduler_cfg):
        super().__init__()
        self.model = model
        self.loss_fn = LpLoss(d=2)
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        print(f'in training step, device: {self.device}')
        x = batch['x']
        y_hat = self.model(x)
        y = batch['y']
        loss = self.loss_fn(y_hat, y)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        print(f'train/loss: {loss}')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.hparams.optimizer_cfg)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.hparams.scheduler_cfg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }

class HookedLpLoss(object):
    """
    https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/losses/data_losses.py#L16
    """

    def __init__(self, d=1, p=2, measure=1., reduction='sum', eps=1e-8):
        super().__init__()
        self.d = d
        self.p = p
        self.eps = eps
        allowed_reductions = ["sum", "mean"]
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction
        if isinstance(measure, float):
            self.measure = [measure]*self.d
        else:
            self.measure = measure

    def reduce_all(self, x):
        """
        reduce x across the batch according to `self.reduction`

        Params
        ------
        x: torch.Tensor
            inputs
        """
        if self.reduction == 'sum':
            x = torch.sum(x)
        else:
            x = torch.mean(x)
        return x
    
    def rel(self, x, y):
        """
        rel: relative LpLoss
        computes ||x-y||/(||y|| + eps)

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        """
        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                          p=self.p, dim=-1, keepdim=False)
        ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)
        diff = diff/(ynorm + self.eps)
        diff = self.reduce_all(diff).squeeze()
        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.rel(y_pred, y)

class ERA5Dataset(Dataset):

    def __init__(self, data_xarray, vertical_levels, nlat, nlon, autoregressive_steps):
        super().__init__()
        assert autoregressive_steps <= len(data_xarray.time)
        self.nlat = nlat
        self.nlon = nlon
        if 'level' in data_xarray.coords:
            levels = data_xarray.level.values
            for var_name in [x for x in data_xarray.data_vars]:
                if 'level' in data_xarray[var_name].coords:
                    for level in levels:
                        if level in vertical_levels[var_name]:
                            data_xarray[f'{var_name}_{int(level)}hPa'] = data_xarray[var_name].sel(level=level)
                    data_xarray = data_xarray.drop_vars(var_name)
            data_xarray = data_xarray.drop_vars('level')

        # should not have to flip or shift as ERA5 by default ranges -90..90 and 0..360
        # if data_xarray.latitude[0] > 0:
        #     data_xarray = flip_xarray_over_equator(data_xarray)
        assert data_xarray.latitude[0] < 0
        assert 0 <= data_xarray.longitude[0] < 5

        self.ds_x = data_xarray.isel(time=slice(0, -autoregressive_steps))
        self.ds_y = data_xarray.isel(time=slice(autoregressive_steps, None))
        assert self.ds_x.sizes == self.ds_y.sizes
        assert np.allclose(
            self.ds_x.isel(time=autoregressive_steps).to_dataarray(),
            self.ds_y.isel(time=0).to_dataarray())
        assert np.allclose(
            self.ds_x.isel(time=-1).to_dataarray(), 
            self.ds_y.isel(time=-(1+autoregressive_steps)).to_dataarray())

    def __len__(self):
        return len(self.ds_x.time)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            x = self.ds_x.isel(time=idx).to_dataarray().transpose('time', 'variable', 'latitude', 'longitude').data
            y = self.ds_y.isel(time=idx).to_dataarray().transpose('time', 'variable', 'latitude', 'longitude').data
        except:
            x = self.ds_x.isel(time=idx).to_dataarray().transpose('variable', 'latitude', 'longitude').data
            y = self.ds_y.isel(time=idx).to_dataarray().transpose('variable', 'latitude', 'longitude').data
        return {'x': torch.Tensor(x), 'y': torch.Tensor(y)}

def get_era5_datasets(data_gs_dir: str, 
        data_local_dir: str,
        data_vars: List[str],
        data_vars_levels: dict,
        autoregressive_steps: int,
        batch_size: int,
        start_year_train: int,
        end_year_train: int,
        end_year_val: int) -> Tuple[DataLoader]:

    data_gs_filename = os.path.basename(os.path.normpath(data_gs_dir))
    data_vars = sorted(list(data_vars))
    vars_hash = hashlib.md5(
        json.dumps(data_vars, sort_keys=True).encode()
        # Don't currently hash by vertical level too because all data for variables is json dumped,
        # and vertical level filtering is done later in initializing the ERA5Dataset
        # + json.dumps(OmegaConf.to_container(data_vars_levels), sort_keys=True).encode()
        ).hexdigest()
    file_dir = os.getcwd() + data_local_dir + data_gs_filename
    file_path = file_dir + f'/{vars_hash}.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            ds = pickle.load(f)
    else:
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        ds = xr.open_zarr(
            data_gs_dir+'.zarr', 
            storage_options={"token": "anon"}, 
            consolidated=True)
        ds = ds[[*data_vars]]
        with open(file_path, 'wb') as f:
            pickle.dump(ds, f)

    # num_samples = len(ds.time)
    # num_train_samples = round(num_samples*(1-val_ratio))
    nlat = len(ds.latitude)
    nlon = len(ds.longitude)
    era5ds_train = ERA5Dataset(
        ds.isel(time=slice(str(start_year_train)+"-01-01", str(end_year_train)+"-12-31")), 
        vertical_levels=data_vars_levels,
        nlat=nlat, 
        nlon=nlon,
        autoregressive_steps=autoregressive_steps)
    era5ds_val = ERA5Dataset(
        ds.isel(time=slice(str(end_year_train+1)+"-01-01", str(end_year_val)+"-12-31")), 
        vertical_levels=data_vars_levels,
        nlat=nlat, 
        nlon=nlon,
        autoregressive_steps=autoregressive_steps)
    train_loader = DataLoader(
        era5ds_train, 
        batch_size=batch_size,
        shuffle=True)
    val_loader = DataLoader(
        era5ds_val, 
        batch_size=batch_size,
        shuffle=False)
    return (train_loader, val_loader)
