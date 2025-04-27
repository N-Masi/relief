import torch
from neuralop.models import SFNO
import cdsapi
import xarray as xr
import cfgrib
import pdb
import gcsfs
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ds_fine = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr')
ds_fine = ds_fine.sel(level=1000)['2m_temperature']
global_variance = np.var(ds_fine, axis=0)
global_variance = np.rot90(global_variance, k=1)
plt.imshow(global_variance)
plt.axis('off')
plt.grid(None)
plt.colorbar(shrink=0.4)
plt.savefig('figs/testfinal.png')
pdb.set_trace()



# plt.imshow(global_variance)
# pdb.set_trace()
# plt.colorbar(shrink=0.4)
# plt.axis('off')
# plt.grid(None)
# plt.savefig('figs/global_variance.png')
# global_variance = global_variance2

# fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
# lats = np.linspace(-90, 90, 121)
# lons = np.linspace(-180, 180, 240)
# lon_grid, lat_grid = np.meshgrid(lons, lats)

# ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.add_feature(cfeature.LAND, edgecolor='black')

# gridlines = ax.gridlines(draw_labels=False)
# gridlines.top_labels = False
# gridlines.right_labels = False
# gridlines.left_labels = False

# pdb.set_trace()

# pcolormesh = ax.pcolormesh(
#     lon_grid, lat_grid, global_variance,
#     transform=ccrs.PlateCarree()
# )

# plt.savefig('figs/global_variance.png')
