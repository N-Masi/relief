import pygmt
import xarray as xr
import pickle

landmask = pygmt.datasets.load_earth_mask(resolution='30m')
with open('metadata/pygmt/landmask_30min_gridline.pkl', 'wb') as f:
    pickle.dump(landmask, f)

landmask = landmask.sel(lat=slice(-90, 90, 3), lon=slice(-178.5, 180, 3))
landmask = landmask.roll(lon=121, roll_coords=True)
landmask = landmask.transpose('lon', 'lat')
with open('metadata/pygmt/landmask_240x121.pkl', 'wb') as f:
    pickle.dump(landmask, f)
