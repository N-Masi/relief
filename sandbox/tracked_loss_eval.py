import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import pdb
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree
from geopy.distance import geodesic

'''
This file takes the per-batch (i.e., training step) loss at each gridpoint when training
an SFNO on 2m Temperature in a shuffled manner from 1959-2010 ERA5 data at 64x32 resolution,
see: sfno_example.py
'''

start_test_idx = 4678-1 # num of training batches, subtract 1 because last batch isn't full
training_steps = list(range(start_test_idx)) # x axis of plots
tracked_losses = torch.load('/users/nmasi/FAIREarth/loss_over_time.pt')
train_losses = tracked_losses[:start_test_idx, :, :]

# coordinates for 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr'
latitudes = np.array([-87.1875, -81.5625, -75.9375, -70.3125, -64.6875, -59.0625, -53.4375,
       -47.8125, -42.1875, -36.5625, -30.9375, -25.3125, -19.6875, -14.0625,
        -8.4375,  -2.8125,   2.8125,   8.4375,  14.0625,  19.6875,  25.3125,
        30.9375,  36.5625,  42.1875,  47.8125,  53.4375,  59.0625,  64.6875,
        70.3125,  75.9375,  81.5625,  87.1875])
longitudes = np.array([0.,   5.625,  11.25,  16.875,  22.5,  28.125,  33.75,  39.375,
        45.,  50.625,  56.25,  61.875,  67.5,  73.125,  78.75 ,  84.375,
        90.,  95.625, 101.25, 106.875, 112.5, 118.125, 123.75 , 129.375,
       135., 140.625, 146.25, 151.875, 157.5, 163.125, 168.75 , 174.375,
       180., 185.625, 191.25, 196.875, 202.5, 208.125, 213.75 , 219.375,
       225., 230.625, 236.25, 241.875, 247.5, 253.125, 258.75 , 264.375,
       270., 275.625, 281.25, 286.875, 292.5, 298.125, 303.75 , 309.375,
       315., 320.625, 326.25, 331.875, 337.5, 343.125, 348.75 , 354.375])
lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
grid_points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
grid_shape = lat_grid.shape

highest5losses = []
highest_tups = []
for i in range(5):
    flattened_idx = np.argmax(train_losses[-1,:,:].cpu())
    highest_tups.append(np.unravel_index(flattened_idx, train_losses[-1,:,:].shape))
    highest5losses.append(train_losses[:, highest_tups[-1][0], highest_tups[-1][1]].cpu())
    train_losses[-1, highest_tups[-1][0], highest_tups[-1][1]] = -math.inf
assert len(highest5losses)==5
assert len(highest5losses[0])==start_test_idx
assert len(highest5losses[0])==len(highest5losses[-1])

for (i, j) in highest_tups:
    train_losses[-1, i, j] = math.inf

lowest5losses = []
lowest_tups = []
for i in range(5):
    flattened_idx = np.argmin(train_losses[-1,:,:].cpu())
    lowest_tups.append(np.unravel_index(flattened_idx, train_losses[-1,:,:].shape))
    lowest5losses.append(train_losses[:, lowest_tups[-1][0], lowest_tups[-1][1]].cpu())
    train_losses[-1, lowest_tups[-1][0], lowest_tups[-1][1]] = math.inf
assert len(lowest5losses)==5
assert len(lowest5losses[0])==start_test_idx
assert len(lowest5losses[0])==len(lowest5losses[-1])
assert not lowest5losses[0][-1]==-math.inf
assert not np.allclose(lowest5losses[0], highest5losses[0])

start_plotting_idx = 500
# for i in range(5):
#     plt.plot(training_steps[start_plotting_idx:], highest5losses[i][start_plotting_idx:], c='red')
#     plt.plot(training_steps[start_plotting_idx:], lowest5losses[i][start_plotting_idx:], c='blue')
mean_highest = np.mean(highest5losses, axis=0)
assert mean_highest.shape==(start_test_idx,)
plt.plot(training_steps[start_plotting_idx:], mean_highest[start_plotting_idx:], c='red', label='avg of 5 highest')
mean_lowest = np.mean(lowest5losses, axis=0)
assert mean_lowest.shape==(start_test_idx,)
plt.plot(training_steps[start_plotting_idx:], mean_lowest[start_plotting_idx:], c='blue', label='avg of 5 lowest')

plt.title("Loss by gridpoint when training SFNO on \n 2m temperature at 32x64 resolution")
plt.legend(loc='upper center')
plt.savefig('figs/train_over_step.png')

highest_coords = []
for tup in highest_tups:
    highest_coords.append((latitudes[tup[0]], longitudes[tup[1]]))
lowest_coords = []
for tup in lowest_tups:
    lowest_coords.append((latitudes[tup[0]], longitudes[tup[1]]))

plt.clf()
color_map = np.zeros(grid_shape)
for point in highest_coords:
    distances = np.array([geodesic(point, gp).km for gp in grid_points])
    closest_idx = np.argmin(distances)
    lat_idx, lon_idx = np.unravel_index(closest_idx, grid_shape)
    color_map[lat_idx, lon_idx] = 1
for point in lowest_coords:
    distances = np.array([geodesic(point, gp).km for gp in grid_points])
    closest_idx = np.argmin(distances)
    lat_idx, lon_idx = np.unravel_index(closest_idx, grid_shape)
    color_map[lat_idx, lon_idx] = 2
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.0))
ax.set_global()
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
cmap = ListedColormap(['white', 'red', 'blue'])
mesh = ax.pcolormesh(longitudes, latitudes, color_map,
                     cmap=cmap, vmin=0, vmax=2, shading='auto',
                     transform=ccrs.PlateCarree(central_longitude=180.0))
red_patch = mpatches.Patch(color='red', label='Highest loss')
blue_patch = mpatches.Patch(color='blue', label='Lowest loss')
plt.legend(handles=[red_patch, blue_patch])
plt.title('Top 5 Highest & Lowest Loss Grid Cells from SFNO on 2m Temp')
plt.savefig('figs/colored_extreme_losses.png')
