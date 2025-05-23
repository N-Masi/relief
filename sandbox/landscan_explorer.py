import rasterio
import matplotlib.pyplot as plt
import pdb
import numpy as np
import math
import torch
from util import get_cell_areas

# to use git lfs on oscar:
# module load git-lfs/3.3.0

# rasterio DatasetReader api: 
# https://rasterio.readthedocs.io/en/latest/api/rasterio.io.html#rasterio.io.DatasetReader
tiff_file = 'metadata/landscan-global-2023/landscan-global-2023.tif'
with rasterio.open(tiff_file) as dataset:
    data = dataset.read(1) # "x" coord is lon, "y" coord is lat <-- this is true of any rasterio.io.DatasetReader
    # Manhattan coords: 
    # dataset.index(-73.9822, 40.7685) == (5907, 12722)
    # ^ input is (lon, lat), output is (lat, lon)
    # data[5907, 12722] == 18171 <-- again, the actual data is transposed to have shape (lat, lon)
    # Max value in Dhaka:
    # np.unravel_index(np.argmax(data), data.shape) == (7954, 32446)
    # dataset.xy(7954, 32446) == (90.38749998918445, 23.712500002650515)
    # >90 as first coord means it can't be lon, so x==lon, y==lat
    # now that I know this, I should reverse the order of the input into dataset.xy, since the max index is in (lat, lon) form
    data = np.where(data==dataset.nodata, -math.inf, data)
    areas = get_cell_areas(data.shape)
    density = data/areas
    np.save('metadata/landscan-global-2023/landscan-density.npy', density)
    # to load: np.load('metadata/landscan-global-2023/landscan-density.npy')
    dataset.close()
