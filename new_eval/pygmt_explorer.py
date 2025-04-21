import xarray as xr
import pygmt
import pdb
import numpy as np
from util import get_cell_areas

landmask = pygmt.datasets.load_earth_mask(resolution='15s', registration='pixel') # highest res possible for testing
landmask = landmask.reindex(lat=list(reversed(landmask.lat)))
areas = get_cell_areas(landmask.shape)

ocean_area = np.where(landmask==0, 1, 0)*areas
lake_area = np.where(landmask==2, 1, 0)*areas
lake_in_lake_area = np.where(landmask==4, 1, 0)*areas
# sanity check, this is ~70.8%
print(f'perc water: {(np.sum(ocean_area)+np.sum(lake_area)+np.sum(lake_in_lake_area))/np.sum(areas)}')
