import torch
import numpy as np
import xarray as xr
import pygeoboundaries_geolab as pgb
import pickle
import geopandas as gpd
from shapely.geometry import Point
import pdb
import gcsfs

def generate_pixeled_metadata(shape: tuple(int), filename: str = f'metadata/unified_{}x{}.pkl') -> str:

    pass
