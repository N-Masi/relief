import torch
import numpy as np
import xarray as xr
import pygeoboundaries_geolab as pgb
import pickle
import geopandas as gpd
from shapely.geometry import Point
import pdb
import gcsfs
from torch.masked import masked_tensor

def generate_pixeled_metadata(shape: tuple[int], filename: str = None):
    '''
    Takes in the shape of the pixel-registered grid to generate metadata for.
    The top-right corner of the data (i.e., index [0,0]) is -180 lon, 90 lat.
    The saved object is a dictionary of binary masks with the key being the group,
    and the value being the binary mask; in each mask, True represents the metadata
    group applies to the pixel at that index, and False means it does not. Thus,
    if you have a grid of loss at each pixel, you can convert it to a masked tensor.

    Metadata keys include:
        - 'ocean'
        - 'land'
        - 'lake'
        - 'lake_island' (islands within lakes)
        - 'lake_island_lake' (lakes within islands that are within lakes)
        - 'any_water' (ocean, lake, or lake_island_lake)
        - 'any_land' (land or islands within lakes)
        - 'polar' (latitude greater than or equal to 60 or less than or equal to -60)
        - 'nonpolar'
        - 
    '''

    if not filename:
        filename = f'metadata/unified/unified_{shape[0]}x{shape[1]}.pkl'
    

