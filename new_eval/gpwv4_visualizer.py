import xarray as xr
import cfgrib
import os
from shutil import rmtree
import pdb
from huggingface_hub import HfApi
from geotiff import GeoTiff
import rasterio
from typing import Literal
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors

'''
Data is collected at 2.5 arc minute intervals, meaning each degree has 24 samples.
So the shape of grid is (180*24, 360*24) = (4320, 8640), meaning 37,324,800 samples in total. 
1,489,943 of these samples have value of 0. 28,284,041 have the NODATA value.

Index [0,0] starts with the upper-right/northwest corner, so (90, -180)=(90N, 180W)

ex: Central Manhattan is (40.7685N, 73.9822W) at data[1182, 2544] with population of 398,678.44 people

Max value of any pixel: 1710353.1
'''
tiff_file = "/users/nmasi/FAIREarth/metadata/gpwv4/gpw_v4_population_count_rev11_2020_2pt5_min.tif"

def index_to_lat_lon(index_lat: int, index_lon: int) -> tuple[float, float]:
    '''
    Assumes that indices come from a daset with shape (4320, 8640) 
    where each step is 2.5minute arcs, or 1/24th of a degree.
    Data returned is based on degrees North and West, so (lat N, lon W).
    '''
    lat = 90-(index_lat/24)
    lon = 180-(index_lon/24)
    return (lat, lon)
    
def lat_lon_to_index(lat: float, lon: float, lat_dir: Literal["N", "S"], lon_dir: Literal["W", "E"]) -> tuple[int, int]:
    '''
    Assumes that indices will be used on a daset with shape (4320, 8640) 
    where each step is 2.5minute arcs, or 1/24th of a degree.
    Finds the nearest gridpoint.
    '''
    
    if (lat_dir == "N" and lat>=0) or (lat_dir == "S" and lat<0):
        lat = 90 - abs(lat)
    elif (lat_dir == "N" and lat<0) or (lat_dir == "S" and lat>=0):
        lat = 90 + abs(lat)
    
    if (lon_dir == "W" and lon>=0) or (lon_dir == "E" and lon<0):
        lon = 180 - abs(lon)
    elif (lon_dir == "W" and lon<0) or (lon_dir == "E" and lon>=0):
        lon = 180 + abs(lon)

    lat = round(lat*24)
    lon = round(lon*24)
    return (lat, lon)    

with rasterio.open(tiff_file) as dataset:
    data = dataset.read(1)  # Read the first band
    nodata_value = dataset.nodata
    print(f"NODATA value: {nodata_value}")
    print(f"Manhattan pop count: {data[lat_lon_to_index(40.7685, 73.9822, 'N', 'W')]}")
    data[data<=0] = 0
    data[(0<data) & (data<=1)] = 1 # can preproc into buckets to be better
    plt.imshow(data, norm=colors.LogNorm(vmin=1), cmap='copper')
    plt.colorbar(shrink=0.4)
    plt.axis('off')
    plt.grid(None)
    # plt.title('Each pixel is colored with the population of the area it covers on the Earth',fontsize=4)
    # plt.suptitle('North India, East China Most Populated Regions in the World',fontsize=6)
    plt.savefig('figs/pop_count.png')
