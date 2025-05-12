from pygeoboundaries_geolab.pygeoboundaries import get_gdf
import geojson
import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np
import xarray as xr
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

all_territories = ['ABW', 'AFG', 'AGO', 'AIA', 'ALB', 'AND', 'ARE', 'ARG', 'ARM',
    'ASM', 'ATA', 'ATG', 'AUS', 'AUT', 'AZE', 'BDI', 'BEL', 'BEN',
    'BES', 'BFA', 'BGD', 'BGR', 'BHR', 'BHS', 'BIH', 'BLM', 'BLR',
    'BLZ', 'BMU', 'BOL', 'BRA', 'BRB', 'BRN', 'BTN', 'BWA', 'CAF',
    'CAN', 'CHE', 'CHL', 'CHN', 'CIV', 'CMR', 'COD', 'COG', 'COK',
    'COL', 'COM', 'CPV', 'CRI', 'CUB', 'CUW', 'CYM', 'CYP', 'CZE',
    'DEU', 'DJI', 'DMA', 'DNK', 'DOM', 'DZA', 'ECU', 'EGY', 'ERI',
    'ESP', 'EST', 'ETH', 'FIN', 'FJI', 'FLK', 'FRA', 'FRO', 'FSM',
    'GAB', 'GBR', 'GEO', 'GGY', 'GHA', 'GIB', 'GIN', 'GLP', 'GMB',
    'GNB', 'GNQ', 'GRC', 'GRD', 'GRL', 'GTM', 'GUF', 'GUM', 'GUY',
    'HND', 'HRV', 'HTI', 'HUN', 'IDN', 'IMN', 'IRL', 'IRN', 'IRQ',
    'ISL', 'ISR', 'ITA', 'JAM', 'JOR', 'JPN', 'KAZ', 'KEN', 'KGZ',
    'KHM', 'KIR', 'KNA', 'KOR', 'KWT', 'LAO', 'LBN', 'LBR', 'LBY',
    'LCA', 'LIE', 'LKA', 'LSO', 'LTU', 'LUX', 'LVA', 'MAR', 'MCO',
    'MDA', 'MDG', 'MDV', 'MEX', 'MHL', 'MKD', 'MLI', 'MLT', 'MMR',
    'MNE', 'MNG', 'MNP', 'MOZ', 'MRT', 'MSR', 'MTQ', 'MUS', 'MWI',
    'MYS', 'MYT', 'NAM', 'NCL', 'NER', 'NGA', 'NIC', 'NIU', 'NLD',
    'NOR', 'NPL', 'NRU', 'NZL', 'OMN', 'PAK', 'PAN', 'PCN', 'PER',
    'PHL', 'PLW', 'PNG', 'POL', 'PRK', 'PRT', 'PRY', 'PSE', 'PYF',
    'QAT', 'REU', 'ROU', 'RUS', 'RWA', 'SAU', 'SDN', 'SEN', 'SGP',
    'SHN', 'SLB', 'SLE', 'SLV', 'SMR', 'SOM', 'SRB', 'SSD', 'STP',
    'SUR', 'SVK', 'SVN', 'SWE', 'SWZ', 'SYC', 'SYR', 'TCA', 'TCD',
    'TGO', 'THA', 'TJK', 'TKM', 'TLS', 'TON', 'TTO', 'TUN', 'TUR',
    'TUV', 'TWN', 'TZA', 'UGA', 'UKR', 'URY', 'USA', 'UZB', 'VAT',
    'VCT', 'VEN', 'VGB', 'VIR', 'VNM', 'VUT', 'WLF', 'WSM', 'XKX',
    'YEM', 'ZAF', 'ZMB', 'ZWE', 'TKL'
]

all_subregions = ['Caribbean', 'Southern Asia', 'Middle Africa', 'Southern Europe',
    'Western Asia', 'South America', 'Polynesia', 'Antarctica',
    'Australia/New Zealand', 'Western Europe', 'Eastern Africa',
    'Western Africa', 'Eastern Europe', 'Central America',
    'Northern America', 'South-Eastern Asia', 'Southern Africa',
    'Eastern Asia', 'Northern Europe', 'Northern Africa', 'Melanesia',
    'Micronesia', 'Central Asia'
]

all_incomes = ['High-income Countries', 'Low-income Countries',
    'Lower-middle-income Countries', 'Upper-middle-income Countries', ]
    # 'No income group available', ]

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
era5 = era5.assign_coords(longitude=(((era5.longitude + 180) % 360) - 180))
lat_weights = get_cell_weights((len(era5.longitude), len(era5.latitude)), lat_index=1) # TODO: see lat weights equation used in model paper RMSE, how does it normalize?

with open('metadata/pygeoboundaries/gdf_region_income_area.pkl', 'rb') as f:
    gdf = pickle.load(f)

summary_data = {
    'model': [], 
    'lead_time': [], 
    'variable': [], 
    'rmse': [],
    'stdev_of_rmses_territories': [], 
    'max_abs_diff_territories': [],
    'stdev_of_rmses_subregions': [], 
    'max_abs_diff_subregions': [],
    'stdev_of_rmses_incomes': [], 
    'max_abs_diff_incomes': [],
}

territory_data = []
subregion_data = []
income_data = []

for model in models:
    ds = xr.open_zarr(models[model])
    ds = ds.sel(time=slice("2020-01-01", "2020-12-31"))
    ds['T850'] = ds['temperature'].sel(level=850)
    ds['Z500'] = ds['geopotential'].sel(level=500)
    ds = ds[['T850', 'Z500']]
    ds['latitude'] = np.round(ds['latitude'], 1) # TODO: generalize/remove if not 240x121
    ds['longitude'] = np.round(ds['longitude'], 1) # TODO: generalize/remove if not 240x121
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds = ds.sel(prediction_timedelta=ds.prediction_timedelta.isin(common_lead_times))
    pdb.set_trace()

    for variable in ds.data_vars:
        for lead_time in ds.prediction_timedelta:
            lead_time_str = str(np.timedelta64(lead_time.values, 'h').astype(int))

            summary_data['model'].append(model)
            summary_data['lead_time'].append(lead_time_str)
            summary_data['variable'].append(variable)

            preds = ds[variable].sel(prediction_timedelta=lead_time)
            gt = era5[variable].sel(time=slice(np.datetime64('2020-01-01T00')+lead_time, np.datetime64('2020-12-31T12')+lead_time))
            gt = gt.sel(time=gt.time.isin(preds.time+lead_time))
            diffs = preds.values - gt.values
            weighted_l2 = lat_weights*(diffs**2)
            weighted_l2 = weighted_l2.transpose(1,2,0).reshape(-1, weighted_l2.shape[0])

            # make gdf of loss values
            mesh_lon, mesh_lat = np.meshgrid(ds.longitude.values, ds.latitude.values)
            flat_lons = mesh_lon.flatten()
            flat_lats = mesh_lat.flatten()
            
            df = pd.DataFrame({
                "latitude": flat_lats,
                "longitude": flat_lons,
                "weighted_l2": [row.tolist() for row in weighted_l2]
            })
            # TODO: change to polygon
            geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
            points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=gdf.crs)
            # TODO: make sure the below predicate returns a list for all taerritory/subregion/income/landcover that the polygon covers
            joined_gdf = gpd.sjoin(points_gdf, gdf, how="left", predicate="covered_by")

            territories = joined_gdf[joined_gdf['shapeGroup'].notna()]['shapeGroup'].unique()
            subregions = joined_gdf[joined_gdf['UNSDG-subregion'].notna()]['UNSDG-subregion'].unique()
            incomes = joined_gdf[joined_gdf['worldBankIncomeGroup'].notna()]['worldBankIncomeGroup'].unique()
            incomes = [x for x in incomes if not x == 'No income group available']

            # territory_rmses = [np.sqrt(np.nanmean(np.concatenate(joined_gdf[joined_gdf['shapeGroup']==territories[i]]['weighted_l2'].values))) for i in range(len(territories))]
            # subregion_rmses = [np.sqrt(np.nanmean(np.concatenate(joined_gdf[joined_gdf['UNSDG-subregion']==subregions[i]]['weighted_l2'].values))) for i in range(len(subregions))]
            # income_rmses = [np.sqrt(np.nanmean(np.concatenate(joined_gdf[joined_gdf['worldBankIncomeGroup']==incomes[i]]['weighted_l2'].values))) for i in range(len(incomes))]

            # summary_data['rmse'].append(np.sqrt(np.nanmean(np.concatenate(joined_gdf['weighted_l2'].values))))
            # summary_data['stdev_of_rmses_territories'].append(np.nanstd(territory_rmses))
            # summary_data['max_abs_diff_territories'].append(np.abs(np.max(territory_rmses)-np.min(territory_rmses)))
            # summary_data['stdev_of_rmses_subregions'].append(np.nanstd(subregion_rmses))
            # summary_data['max_abs_diff_subregions'].append(np.abs(np.max(subregion_rmses)-np.min(subregion_rmses)))
            # summary_data['stdev_of_rmses_incomes'].append(np.nanstd(income_rmses))
            # summary_data['max_abs_diff_incomes'].append(np.abs(np.max(income_rmses)-np.min(income_rmses)))

            # with open('results/gdf_summary_data_ckpt.pkl', 'wb') as f:
            #     pickle.dump(summary_data, f)

            for i in range(len(territories)):
                territory_data.append({
                    'model': model,
                    'lead_time': lead_time_str,
                    'variable': variable,
                    'territory': territories[i],
                    # TODO: change the == to check if territories[i] in joined_gdf['shapeGroup'] (allows for list of countries/territories)
                    'rmse': np.sqrt(np.nanmean(np.concatenate(joined_gdf[joined_gdf['shapeGroup']==territories[i]]['weighted_l2'].values)))
                })
            with open('results/gdf_territory_data_ckpt.pkl', 'wb') as f:
                pickle.dump(territory_data, f)

            for i in range(len(subregions)):
                subregion_data.append({
                    'model': model,
                    'lead_time': lead_time_str,
                    'variable': variable,
                    'subregion': subregions[i],
                    'rmse': np.sqrt(np.nanmean(np.concatenate(joined_gdf[joined_gdf['UNSDG-subregion']==subregions[i]]['weighted_l2'].values)))
                })
            with open('results/gdf_subregion_data_ckpt.pkl', 'wb') as f:
                pickle.dump(subregion_data, f)

            for i in range(len(incomes)):
                income_data.append({
                    'model': model,
                    'lead_time': lead_time_str,
                    'variable': variable,
                    'income': incomes[i],
                    'rmse': np.sqrt(np.nanmean(np.concatenate(joined_gdf[joined_gdf['worldBankIncomeGroup']==incomes[i]]['weighted_l2'].values)))
                })
            with open('results/gdf_income_data_ckpt.pkl', 'wb') as f:
                pickle.dump(income_data, f)

# with open('results/gdf_data_summary_stats.pkl', 'wb') as f:
#     pickle.dump(data, f)
with open('results/gdf_territory_data.pkl', 'wb') as f:
    pickle.dump(territory_data, f)
with open('results/gdf_subregion_data.pkl', 'wb') as f:
    pickle.dump(subregion_data, f)
with open('results/gdf_income_data.pkl', 'wb') as f:
    pickle.dump(income_data, f)
