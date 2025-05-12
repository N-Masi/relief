import selenium # may need to be installed
import folium # may need to be installed
import io
from PIL import Image
import geopandas as gpd
import pickle

with open('metadata/pygeoboundaries/gdf_region_income_area.pkl', 'rb') as f: 
    gdf = pickle.load(f)

m = folium.Map(location=[-13.5, -155.20], zoom_start=5)
for i in range(len(gdf)):
    if gdf.iloc[i]['UNSDG-subregion']=='Polynesia': 
        folium.GeoJson(gdf.iloc[i].geometry).add_to(m)

img_data = m._to_png(10)
img = Image.open(io.BytesIO(img_data))
img.save('untracked/polynesia_map.png')

m = folium.Map(location=[45, -100], zoom_start=3)
counter = 0
for i in range(len(gdf)):
    if gdf.iloc[i]['shapeGroup']=='USA': 
        folium.GeoJson(gdf.iloc[i].geometry).add_to(m)
        counter += 1
assert counter == 1

img_data = m._to_png(10)
img = Image.open(io.BytesIO(img_data))
img.save('untracked/usa_map.png')