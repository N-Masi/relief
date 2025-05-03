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

dsgraphcast = xr.open_zarr('gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr')
dsgraphcast = dsgraphcast.sel(time=slice("2020-01-01", "2020-12-31"))

# only has predictions made every other day, not for the whole year either
dsscnn = xr.open_zarr('gs://weatherbench2/datasets/sphericalcnn/2020-240x121_equiangular_with_poles.zarr')

dsneuralgcm = xr.open_zarr('gs://weatherbench2/datasets/neuralgcm_deterministic/2020-240x121_equiangular_with_poles_conservative.zarr')

dsngcmens = xr.open_zarr('gs://weatherbench2/datasets/neuralgcm_ens/2020-240x121_equiangular_with_poles_conservative_mean.zarr')

dsfuxi = xr.open_zarr('gs://weatherbench2/datasets/fuxi/2020-240x121_equiangular_with_poles_conservative.zarr')

common_lead_times = set(dspangu.prediction_timedelta.values) \
    .intersection(set(dsgraphcast.prediction_timedelta.values)) \
    .intersection(set(dsscnn.prediction_timedelta.values)) \
    .intersection(set(dsneuralgcm.prediction_timedelta.values)) \
    .intersection(set(dsngcmens.prediction_timedelta.values)) \
    .intersection(set(dsfuxi.prediction_timedelta.values))
common_lead_times = sorted([np.timedelta64(t,'h') for t in common_lead_times])

pdb.set_trace()
