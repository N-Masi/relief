from pygeoboundaries_geolab import get_adm
from pygeoboundaries_geolab import get_gdf
import geopandas as gpd
from shapely.geometry import Point
import pickle
import pdb

gdf = get_gdf('ALL', ['UNSDG-subregion', 'worldBankIncomeGroup', 'maxAreaSqKM'])
with open('~/relief/metadata/pygeoboundaries/gdf_region_income_area.pkl', 'wb') as f:
    pickle.dump(gdf, f)
