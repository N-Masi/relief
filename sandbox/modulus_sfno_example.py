import torch
import cdsapi
import xarray as xr
import cfgrib
import pdb
import gcsfs
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ace_helpers import *
# uses conflicting version of torch-harmonics from neuralop, so
# pip install git+https://github.com/ai2cm/modulus.git@22df4a9427f5f12ff6ac891083220e7f2f54d229


# load in SFNO
'''
# Recreate Bonev (2018) original SFNO paper
model = SFNO(n_modes=(16, 16), in_channels=26, out_channels=26,
                hidden_channels=384, n_layers=8)
# unclear how many n_modes to use in paper, this is package default
'''

print('running toy sfno example')

assert torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# get data from WB2
ds = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr')
print('loaded in data')
dataset_vars = ['2m_temperature']
ds = ds[dataset_vars] # cut down to 1 channel
print(f'reduced data down to vars: {dataset_vars}')
# TODO: normalize data

# create train & test torch.utils.data.DataLoader from ds
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# xarray/geo-data to torchloader: https://zen3geo.readthedocs.io/en/v0.2.0/chipping.html
class ERA5Dataset(Dataset):

    def __init__(self, data_xarray):
        super().__init__()
        if 'level' in data_xarray.coords:
            levels = data_xarray.levels.values
            for var_name in [x for x in data_xarray.data_vars]:
                if 'level' in data_xarray[var_name].coords:
                    for level in levels:
                        data_xarray[f'{var_name}{int(level)}'] = data[var_name].sel(level=level)
                    data_xarray = data_xarray.drop_vars(var_name)
            data_xarray = data_xarray.drop_vars('level')
        self.ds = data_xarray

    def __len__(self):
        return len(self.ds.time)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        samples = self.ds.isel(time=idx).to_dataarray().transpose('time', 'variable', 'latitude', 'longitude')
        return samples.data

era5ds = ERA5Dataset(ds)
print('created dataset from data')
num_samples = len(era5ds)
num_train_samples = round(num_samples*0.8)
train_loader = DataLoader(era5ds[:num_train_samples], batch_size=16, shuffle=True)
print('created train_loader from dataset')
test_loader = DataLoader(era5ds[num_train_samples:], batch_size=16, shuffle=True)
test_loaders = {2048: test_loader}
print('created test_loader from dataset')

# more efficient SFNO for demo purposes
#model = SFNO(n_modes=(16, 16), in_channels=1, out_channels=1, hidden_channels=64) # TODO: neuralop expects data samples to be returned in dict format with 'x' and 'y' keys
#model.projection.fcs[1].register_forward_hook(fn)
model = get_ace_sto_sfno(img_shape=(64,32), in_chans=1, out_chans=1, dropout=0, device=DEVICE)
print('instantiated model')
fn = lambda module, args, output: print(output) # TODO: hook that gets FAIRE loss over training time at each sample
model.module.decoder[2].register_forward_hook(fn)
print('registered hook on model')

# train on SFNO on ERA5

# optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.0)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
# trainer = Trainer(model=model,
#                   n_epochs=1,
#                   device=DEVICE,
#                   wandb_log=False,
#                   use_distributed=False,
#                   verbose=True)
# print('instantiated trainer')
# print('TRAINING COMMENCING')
# trainer.train(train_loader=train_loader,
#               test_loaders=test_loaders,
#               optimizer=optimizer,
#               scheduler=scheduler,
#               regularizer=False)
# print('DONE TRAINING')

# TODO: use RELIEF to get fairness metrics across all val/test data
optimizer = get_ace_optimizer(model)
scheduler = get_ace_lr_scheduler(optimizer)
loss_fn = AceLoss()
print('TRAINING COMMENCING')
model.train()
for batch in train_loader:
    torch.cuda.empty_cache()
    optimizer.zero_grad()
    pdb.set_trace()
    pred = model(batch[0].to(DEVICE, non_blocking=True))
    loss = loss_fn(pred, batch[1].to(DEVICE, non_blocking=True))
    loss.backward()
    optimizer.step()
print('TRAINING DONE')
