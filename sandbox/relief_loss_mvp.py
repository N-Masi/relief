import xarray as xr
import numpy as np
import torch
import zarr
import cfgrib
import gcsfs
import fsspec
import pdb

dspangu = xr.open_zarr('gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr')
dspangu = dspangu.sel(time=slice("2020-01-01", "2020-12-31"))
dspangu['T850'] = dspangu['temperature'].sel(level=850)
dspangu['Z500'] = dspangu['geopotential'].sel(level=500)
dspangu = dspangu[['T850', 'Z500']]

common_lead_times =[np.timedelta64(12,'h'), 
    np.timedelta64(24,'h'), 
    np.timedelta64(36,'h'), 
    np.timedelta64(48,'h'), 
    np.timedelta64(60,'h'), 
    np.timedelta64(72,'h'), 
    np.timedelta64(84,'h'), 
    np.timedelta64(96,'h'), 
    np.timedelta64(108,'h'),
    np.timedelta64(120,'h'), 
    np.timedelta64(132,'h'), 
    np.timedelta64(144,'h'), 
    np.timedelta64(156,'h'), 
    np.timedelta64(168,'h'), 
    np.timedelta64(180,'h'), 
    np.timedelta64(192,'h'), 
    np.timedelta64(204,'h'), 
    np.timedelta64(216,'h'), 
    np.timedelta64(228,'h'), 
    np.timedelta64(240,'h')
]
dspangu = dspangu.sel(prediction_timedelta=dspangu.prediction_timedelta.isin(common_lead_times))

# load in ERA5 data for these two variables in 2020 (up to 2021-01-10T12) at 12h intervals
era5 = xr.open_zarr(
        'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr',
        storage_options={"token": "anon"}, 
        consolidated=True)
era5 = era5.sel(time=slice('2020-01-01T00', '2021-01-10T12', 2)) # downsample to 12hourly, not 6
era5['T850'] = era5['temperature'].sel(level=850)
era5['Z500'] = era5['geopotential'].sel(level=500)
era5 = era5[['T850', 'Z500']]

pdb.set_trace()
# TODO:
# for each lead time
    # for each variable (T850, Z500, ...) [this will generate 366*2 values for each gridpoint]
        # subtract ground truth values (gt values need to be shifted forward from 2020-01-01T00 by lead time)
        # square differences and weight by surface area (l2 loss)
    # for each metadata group
        # for each category
            # get sqrt of average l2 loss (RMSE_l) for gridpoints in that cat
            # save value (how???)
