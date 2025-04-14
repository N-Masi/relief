import xarray as xr
import pygmt
import pdb

landmask = pygmt.datasets.load_earth_mask(resolution='15s', registration='pixel')
num_gridpoints = landmask.shape[0]*landmask.shape[1]
(np.sum(landmask==0)+np.sum(landmask==2)+np.sum(landmask==4))/num_gridpoints
pdb.set_trace()
