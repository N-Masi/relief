import xarray as xr
import numpy as np
import torch
import zarr
import cfgrib
import gcsfs
import fsspec
import pdb
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.surface_area import *
import pygmt
import pickle

# dspangu = xr.open_zarr('gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr')
# dspangu = dspangu.sel(time=slice("2020-01-01", "2020-12-31"))
# dspangu['T850'] = dspangu['temperature'].sel(level=850)
# dspangu['Z500'] = dspangu['geopotential'].sel(level=500)
# dspangu = dspangu[['T850', 'Z500']]


models = {
    'keisler': 'gs://weatherbench2/datasets/keisler/2020-240x121_equiangular_with_poles_conservative.zarr',
    'pangu': 'gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr',
    'graphcast': 'gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr',
    'sphericalcnn': 'gs://weatherbench2/datasets/sphericalcnn/2020-240x121_equiangular_with_poles.zarr',
    'fuxi': 'gs://weatherbench2/datasets/fuxi/2020-240x121_equiangular_with_poles_conservative.zarr',
    'neuralgcm': 'gs://weatherbench2/datasets/neuralgcm_deterministic/2020-240x121_equiangular_with_poles_conservative.zarr',
}

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
# dspangu = dspangu.sel(prediction_timedelta=dspangu.prediction_timedelta.isin(common_lead_times))

# load in ERA5 data for these two variables in 2020 (up to 2021-01-10T12) at 12h intervals
era5 = xr.open_zarr(
        'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr',
        storage_options={"token": "anon"}, 
        consolidated=True)
era5 = era5.sel(time=slice('2020-01-01T00', '2021-01-10T12', 2)) # downsample to 12hourly, not 6
era5['T850'] = era5['temperature'].sel(level=850)
era5['Z500'] = era5['geopotential'].sel(level=500)
era5 = era5[['T850', 'Z500']]
era5['latitude'] = np.round(era5['latitude'], 1)
era5['longitude'] = np.round(era5['longitude'], 1)
lat_weights = get_cell_weights((len(era5.longitude), len(era5.latitude)), lat_index=1)

# landmask
with open('metadata/pygmt/landmask_240x121.pkl', 'rb') as f:
    landmask = pickle.load(f)
landmask = landmask.rename({'lat': 'latitude', 'lon': 'longitude'})

for model in models:
    ds = xr.open_zarr(models[model])
    ds = ds.sel(time=slice("2020-01-01", "2020-12-31"))
    ds['T850'] = ds['temperature'].sel(level=850)
    ds['Z500'] = ds['geopotential'].sel(level=500)
    ds = ds[['T850', 'Z500']]
    ds['latitude'] = np.round(ds['latitude'], 1) # TODO: generalize/remove if not 240x121
    ds['longitude'] = np.round(ds['longitude'], 1) # TODO: generalize/remove if not 240x121
    ds = ds.sel(prediction_timedelta=ds.prediction_timedelta.isin(common_lead_times))

    rmses = {'T850': {'oceans': [], 'land': [], 'lakes': []}, 'Z500': {'oceans': [], 'land': [], 'lakes': []}, 'lead_times': list(range(12, 241, 12))}
    for variable in ds.data_vars:
        for lead_time in ds.prediction_timedelta:
            lead_time_str = str(np.timedelta64(lead_time.values, 'h').astype(int))
            # this will generate 366*2 values for each gridpoint
            preds = ds[variable].sel(prediction_timedelta=lead_time)
            # subtract ground truth values (gt values need to be shifted forward from 2020-01-01T00 by lead time)
            gt = era5[variable].sel(time=slice(np.datetime64('2020-01-01T00')+lead_time, np.datetime64('2020-12-31T12')+lead_time))
            gt = gt.sel(time=gt.time.isin(preds.time+lead_time))
            diffs = preds.values - gt.values
            # square differences and weight by surface area (l2 loss)
            weighted_l2 = lat_weights*(diffs**2)
            print(f'Dimensionality of weighted_l2: {weighted_l2.shape}')
            # TODO: have the loss function be a user-provided argument, with weighted_l2 as default
            # for each metadata group
                # for each category: 
                    # get sqrt of average l2 loss (RMSE_l) for gridpoints in that cat
            ocean_rmse = np.sqrt(np.nanmean(weighted_l2[:, landmask.values == 0]))
            rmses[variable]['oceans'].append(ocean_rmse)
            print(f'{model} RMSE ({lead_time_str}hours lead time) for {variable} on OCEANS: {ocean_rmse}')
            land_rmse = np.sqrt(np.nanmean(weighted_l2[:, (landmask.values == 1) | (landmask.values == 3)]))
            rmses[variable]['land'].append(land_rmse)
            print(f'{model} RMSE ({lead_time_str}hours lead time) for {variable} on LAND: {land_rmse}')
            lake_rmse = np.sqrt(np.nanmean(weighted_l2[:, (landmask.values == 2) | (landmask.values == 4)]))
            rmses[variable]['lakes'].append(lake_rmse)
            print(f'{model} RMSE ({lead_time_str}hours lead time) for {variable} on LAKES: {lake_rmse}')

    with open(f'untracked/240x121/{model}_rmses.pkl', 'wb') as f:
        pickle.dump(rmses, f)
    