import torch
import numpy as np
import xarray as xr
import pygeoboundaries_geolab as pgb
import pickle
import geopandas as gpd
from shapely.geometry import Point
import pdb
import gcsfs
import ee

# to activate gcloud command, run the below on OSCAR terminal:
# module load google-cloud-cli/456.0.0
# gcloud config set project earth-engine-457501

def get_weathernext_forecasts():
    ee.Authenticate()
    ee.Initialize(project='earth-engine-457501')

    # access data as an ee.ImageCollection:
    # predsImageCollection = ee.ImageCollection('projects/gcp-public-data-weathernext/assets/59572747_4_0') \
    #     .filter(ee.Filter.date('2020-10-01T06:00:00Z', '2020-10-01T06:01:00Z')) \
    #     .filter(ee.Filter.eq('forecast_hour', 6))

    # access data directly as an xarray via zarr:
    zarr_path = 'gs://weathernext/59572747_4_0/zarr'
    ds = xr.open_zarr(zarr_path, storage_options={'token': 'google_default'})
