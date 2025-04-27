import torch
from neuralop.models import SFNO
# if the above throws errors, redo: pip install neuraloperator
from neuralop.training import Trainer
from neuralop.losses import LpLoss
import cdsapi
import xarray as xr
import cfgrib
import pdb
import gcsfs
import fsspec
from torch.utils.data import Dataset, DataLoader
import numpy as np

print('running toy sfno example')

assert torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# get data from WB2
ds = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr')
print('loaded in data')
dataset_vars = ['2m_temperature', 'wind_speed'] # cut down to 1 channel
# dataset_vars = [name for name in ds.data_vars] # TODO: normalize data
ds = ds[dataset_vars] 
print(f'reduced data down to vars: {dataset_vars}')

# create train & test torch.utils.data.DataLoader from ds
class ERA5Dataset(Dataset):

    def __init__(self, data_xarray, nlat, nlon):
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon
        if 'level' in data_xarray.coords:
            levels = data_xarray.level.values
            for var_name in [x for x in data_xarray.data_vars]:
                if 'level' in data_xarray[var_name].coords:
                    for level in levels:
                        data_xarray[f'{var_name}{int(level)}'] = data_xarray[var_name].sel(level=level)
                    data_xarray = data_xarray.drop_vars(var_name)
            data_xarray = data_xarray.drop_vars('level')
        self.ds_x = data_xarray.isel(time=slice(0, -1))
        self.ds_y = data_xarray.isel(time=slice(1, None))
        # assert np.allclose(self.ds_x.isel(time=1).to_dataarray(), self.ds_y.isel(time=0).to_dataarray())
        # assert np.allclose(self.ds_x.isel(time=-1).to_dataarray(), self.ds_y.isel(time=-2).to_dataarray())

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

num_samples = len(ds.time)
num_train_samples = round(num_samples*0.8)
era5ds_train = ERA5Dataset(ds.isel(time=slice(0, num_train_samples)), nlat=32, nlon=64)
era5ds_test = ERA5Dataset(ds.isel(time=slice(num_train_samples, None)), nlat=32, nlon=64)
print('created datasets from data')
train_loader = DataLoader(era5ds_train, batch_size=16, shuffle=True)
print('created train_loader from dataset')
test_loader = DataLoader(era5ds_test, batch_size=16, shuffle=True) # TODO: should test loader not be shuffled for autoregressive rollout?
test_loaders = {'32x64': test_loader}
print('created test_loader from dataset')

# modified loss fn to track per gridpoint loss
loss_over_time = torch.empty((0)).to(DEVICE)
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
    
    @property
    def name(self):
        return f"L{self.p}_{self.d}Dloss"
    
    def uniform_quadrature(self, x):
        """
        uniform_quadrature creates quadrature weights
        scaled by the spatial size of ``x`` to ensure that 
        ``LpLoss`` computes the average over spatial dims. 

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        quadrature : list
            list of quadrature weights per-dim
        """
        quadrature = [0.0]*self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j]/x.size(-j)
        
        return quadrature

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

    def abs(self, x, y, quadrature=None):
        """absolute Lp-norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : float or list, optional
            quadrature weights for integral
            either single scalar or one per dimension
        """
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
        
        const = math.prod(quadrature)**(1.0/self.p)
        diff = const*torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                                              p=self.p, dim=-1, keepdim=False)

        diff = self.reduce_all(diff).squeeze()
            
        return diff

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

        # track gridpoint specifc loss
        global loss_over_time
        loss_over_time = torch.cat((loss_over_time, torch.mean(abs(x-y), dim=(0,1)).unsqueeze(0)))
        torch.save(loss_over_time, "loss_over_time.pt")
        print(f"gridpoint loss saved with dimensions: {loss_over_time.shape}")

        diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.rel(y_pred, y)

# more efficient SFNO for demo purposes
print('instantiated model')
model = SFNO(n_modes=(16, 16), in_channels=14, out_channels=14, hidden_channels=64)
hooked_loss_fn = HookedLpLoss(d=2)
def hook_fn(module, input, output):
    # put module in eval mode
    pdb.set_trace()
# model.projection.fcs[1].register_forward_hook(hook_fn)
# print('registered hook on model')

# train on SFNO on ERA5
optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
trainer = Trainer(model=model,
                  n_epochs=1,
                  device=DEVICE,
                  wandb_log=False,
                  use_distributed=False,
                  verbose=True)
print('instantiated trainer')
print('TRAINING COMMENCING')
trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              training_loss = hooked_loss_fn)
print('DONE TRAINING')

# TODO: use RELIEF to get fairness metrics across all val/test data
