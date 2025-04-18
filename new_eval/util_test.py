import pytest
import pdb
import math
import numpy as np
import xarray as xr
import pygmt
from new_eval.util import *

# https://mathworld.wolfram.com/OblateSpheroid.html
# 2*math.pi*a*a+math.pi*(c**2/e)*math.log((1+e)/(1-e)) # inprecise
EARTH_SURFACE_AREA = 510065604944206.145

def test_surface_area_shape():
    areas = get_cell_areas((2,4))
    assert areas.shape == (2,4)

    areas = get_cell_areas((4,8))
    assert areas.shape == (4,8)

    areas = get_cell_areas((32,64))
    assert areas.shape == (32,64)

    areas = get_cell_areas((720,1440))
    assert areas.shape == (720,1440)

    areas = get_cell_areas((4320,8640))
    assert areas.shape == (4320,8640)

    areas = get_cell_areas((21600,43200))
    assert areas.shape == (21600,43200)

    areas = get_cell_areas((10,400))
    assert areas.shape == (10,400)

def test_total_surface_area():
    areas = get_cell_areas((2,4))
    assert np.allclose(np.sum(areas), EARTH_SURFACE_AREA)

    areas = get_cell_areas((4,8))
    assert np.allclose(np.sum(areas), EARTH_SURFACE_AREA)

    areas = get_cell_areas((32,64))
    assert np.allclose(np.sum(areas), EARTH_SURFACE_AREA)

    areas = get_cell_areas((720, 1440))
    assert np.allclose(np.sum(areas), EARTH_SURFACE_AREA)

    areas = get_cell_areas((4320,8640))
    assert np.allclose(np.sum(areas), EARTH_SURFACE_AREA)

    areas = get_cell_areas((21600,43200))
    assert np.allclose(np.sum(areas), EARTH_SURFACE_AREA)

    areas = get_cell_areas((43200,86400))
    assert np.allclose(np.sum(areas), EARTH_SURFACE_AREA)

    areas = get_cell_areas((10,400))
    assert np.allclose(np.sum(areas), EARTH_SURFACE_AREA)

def test_surface_area_max_is_equator():
    areas = get_cell_areas((2,4))
    equator_top_index = (areas.shape[0]//2)-1
    argmax = np.unravel_index(np.argmax(areas), areas.shape)
    assert equator_top_index == argmax[0]
    assert np.allclose(areas[equator_top_index], areas[equator_top_index+1])

    areas = get_cell_areas((4,8))
    equator_top_index = (areas.shape[0]//2)-1
    argmax = np.unravel_index(np.argmax(areas), areas.shape)
    assert equator_top_index == argmax[0]
    assert np.allclose(areas[equator_top_index], areas[equator_top_index+1])

    areas = get_cell_areas((32,64))
    equator_top_index = (areas.shape[0]//2)-1
    argmax = np.unravel_index(np.argmax(areas), areas.shape)
    assert equator_top_index == argmax[0]
    assert np.allclose(areas[equator_top_index], areas[equator_top_index+1])

    areas = get_cell_areas((720, 1440))
    equator_top_index = (areas.shape[0]//2)-1
    argmax = np.unravel_index(np.argmax(areas), areas.shape)
    assert equator_top_index == argmax[0]
    assert np.allclose(areas[equator_top_index], areas[equator_top_index+1])

    areas = get_cell_areas((4320,8640))
    equator_top_index = (areas.shape[0]//2)-1
    argmax = np.unravel_index(np.argmax(areas), areas.shape)
    assert equator_top_index == argmax[0]
    assert np.allclose(areas[equator_top_index], areas[equator_top_index+1])

    areas = get_cell_areas((21600,43200))
    equator_top_index = (areas.shape[0]//2)-1
    argmax = np.unravel_index(np.argmax(areas), areas.shape)
    assert equator_top_index == argmax[0]
    assert np.allclose(areas[equator_top_index], areas[equator_top_index+1])

    areas = get_cell_areas((10,400))
    equator_top_index = (areas.shape[0]//2)-1
    argmax = np.unravel_index(np.argmax(areas), areas.shape)
    assert equator_top_index == argmax[0]
    assert np.allclose(areas[equator_top_index], areas[equator_top_index+1])

def test_surface_area_is_monotonic_over_lat():
    areas = get_cell_areas((2,4))
    for i in range(areas.shape[0]//2-1):
        assert np.all(areas[i] == areas[i, 0])
        assert areas[i, 0] < areas[i+1, 0]
    assert np.all(areas[areas.shape[0]//2-1] == areas[areas.shape[0]//2-1, 0])
    for i in range(areas.shape[0]//2, areas.shape[0]-1):
        assert np.all(areas[i] == areas[i, 0])
        assert areas[i, 0] > areas[i+1, 0]
    assert np.all(areas[areas.shape[0]-1] == areas[areas.shape[0]-1, 0])

    areas = get_cell_areas((4,8))
    for i in range(areas.shape[0]//2-1):
        assert np.all(areas[i] == areas[i, 0])
        assert areas[i, 0] < areas[i+1, 0]
    assert np.all(areas[areas.shape[0]//2-1] == areas[areas.shape[0]//2-1, 0])
    for i in range(areas.shape[0]//2, areas.shape[0]-1):
        assert np.all(areas[i] == areas[i, 0])
        assert areas[i, 0] > areas[i+1, 0]
    assert np.all(areas[areas.shape[0]-1] == areas[areas.shape[0]-1, 0])

    areas = get_cell_areas((32,64))
    for i in range(areas.shape[0]//2-1):
        assert np.all(areas[i] == areas[i, 0])
        assert areas[i, 0] < areas[i+1, 0]
    assert np.all(areas[areas.shape[0]//2-1] == areas[areas.shape[0]//2-1, 0])
    for i in range(areas.shape[0]//2, areas.shape[0]-1):
        assert np.all(areas[i] == areas[i, 0])
        assert areas[i, 0] > areas[i+1, 0]
    assert np.all(areas[areas.shape[0]-1] == areas[areas.shape[0]-1, 0])

    areas = get_cell_areas((720, 1440))
    for i in range(areas.shape[0]//2-1):
        assert np.all(areas[i] == areas[i, 0])
        assert areas[i, 0] < areas[i+1, 0]
    assert np.all(areas[areas.shape[0]//2-1] == areas[areas.shape[0]//2-1, 0])
    for i in range(areas.shape[0]//2, areas.shape[0]-1):
        assert np.all(areas[i] == areas[i, 0])
        assert areas[i, 0] > areas[i+1, 0]
    assert np.all(areas[areas.shape[0]-1] == areas[areas.shape[0]-1, 0])

    areas = get_cell_areas((4320,8640))
    for i in range(areas.shape[0]//2-1):
        assert np.all(areas[i] == areas[i, 0])
        assert areas[i, 0] < areas[i+1, 0]
    assert np.all(areas[areas.shape[0]//2-1] == areas[areas.shape[0]//2-1, 0])
    for i in range(areas.shape[0]//2, areas.shape[0]-1):
        assert np.all(areas[i] == areas[i, 0])
        assert areas[i, 0] > areas[i+1, 0]
    assert np.all(areas[areas.shape[0]-1] == areas[areas.shape[0]-1, 0])

    areas = get_cell_areas((21600,43200))
    for i in range(areas.shape[0]//2-1):
        assert np.all(areas[i] == areas[i, 0])
        assert areas[i, 0] < areas[i+1, 0]
    assert np.all(areas[areas.shape[0]//2-1] == areas[areas.shape[0]//2-1, 0])
    for i in range(areas.shape[0]//2, areas.shape[0]-1):
        assert np.all(areas[i] == areas[i, 0])
        assert areas[i, 0] > areas[i+1, 0]
    assert np.all(areas[areas.shape[0]-1] == areas[areas.shape[0]-1, 0])

    areas = get_cell_areas((10,400))
    for i in range(areas.shape[0]//2-1):
        assert np.all(areas[i] == areas[i, 0])
        assert areas[i, 0] < areas[i+1, 0]
    assert np.all(areas[areas.shape[0]//2-1] == areas[areas.shape[0]//2-1, 0])
    for i in range(areas.shape[0]//2, areas.shape[0]-1):
        assert np.all(areas[i] == areas[i, 0])
        assert areas[i, 0] > areas[i+1, 0]
    assert np.all(areas[areas.shape[0]-1] == areas[areas.shape[0]-1, 0])

def test_surface_area_symmetric_over_equator():
    areas = get_cell_areas((2,4))
    for i in range(areas.shape[0]//2):
        assert np.allclose(areas[i], areas[-(i+1)])

    areas = get_cell_areas((4,8))
    for i in range(areas.shape[0]//2):
        assert np.allclose(areas[i], areas[-(i+1)])
    
    areas = get_cell_areas((32,64))
    for i in range(areas.shape[0]//2):
        assert np.allclose(areas[i], areas[-(i+1)])
    
    areas = get_cell_areas((720, 1440))
    for i in range(areas.shape[0]//2):     
        assert np.allclose(areas[i], areas[-(i+1)])
    
    areas = get_cell_areas((4320,8640)) 
    for i in range(areas.shape[0]//2):
        assert np.allclose(areas[i], areas[-(i+1)])
    
    areas = get_cell_areas((21600,43200)) 
    for i in range(areas.shape[0]//2):
        assert np.allclose(areas[i], areas[-(i+1)])

    areas = get_cell_areas((10,400))
    for i in range(areas.shape[0]//2):
        assert np.allclose(areas[i], areas[-(i+1)])

# def test_water_coverage_sanity_check():
#     landmask = pygmt.datasets.load_earth_mask(resolution='15s', registration='pixel')
#     landmask = landmask.reindex(lat=list(reversed(landmask.lat)))
#     areas = get_cell_areas(landmask)
#     ocean_area = np.where(landmask==0, 1, 0)*areas
#     lake_area = np.where(landmask==2, 1, 0)*areas
#     lake_in_lake_area = np.where(landmask==4, 1, 0)*areas
#     perc_water = (np.sum(ocean_area)+np.sum(lake_area)+np.sum(lake_in_lake_area))/np.sum(areas)
#     assert ~71%