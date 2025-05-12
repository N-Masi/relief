
from pygeoboundaries_geolab import get_adm
from pygeoboundaries_geolab import get_gdf
import geopandas as gpd
from shapely.geometry import Point
import pickle
import pdb

# to load in and serialize:
'''
coco = get_adm('ALL', 'ADM0')
with open('coco.pkl', 'wb') as f:
    pickle.dump(coco, f)
'''

# read in and get GeoDataFrame:
'''
with open('coco.pkl', 'rb') as f:
    coco = pickle.load(f)
gdf = gpd.GeoDataFrame.from_features(coco)
gdf = gdf.set_crs('WGS84')
'''

# check which country a coordinate is in
'''
dhaka = Point(90.38749998918445, 23.712500002650515)
containing_country = gdf[gdf.geometry.contains(dhaka)]
assert len(containing_country) <= 1
country_name = containing_country.iloc[0].shapeName # 'Bangledesh'
'''

gdf = get_gdf('ALL', ['UNSDG-subregion', 'worldBankIncomeGroup', 'maxAreaSqKM'])
with open('~/relief/metadata/pygeoboundaries/gdf_region_income_area.pkl', 'wb') as f:
    pickle.dump(gdf, f)
