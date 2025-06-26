

# Find number of transformers that are in urban and coastal areas
# The transformers in coastal and urban locations would be gas-based switchgear and not air-based.
# Calculate this for both HVAC and MTHVDC scenarios. 

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
from shapely import wkt
from matplotlib.lines import Line2D
import os
from simply_urban_shapefile import get_urban
from Number_transformers import get_number_transformers 
from shapely.wkt import loads
import matplotlib.patches as mpatches

#%% Inputs

## Define scenario name
scenario_name = 'S03' 

## Define datafiles
coastal_shape_ifile = os.path.abspath(os.path.join(os.getcwd(), '..', 'Data/ne_10m_coastline/ne_10m_coastline.shp'))
coast_counties_ifile = os.path.abspath(os.path.join(os.getcwd(), '..', 'Data/Coastal_Counties/Coastal_Counties.shp'))
ifile_urban_shp = os.path.abspath(os.path.join(os.getcwd(), '..', 'Data/tl_2024_us_uac20/tl_2024_us_uac20.shp'))
datapath = os.path.abspath(os.path.join(os.getcwd(), '..', 'Data/R02_Transmission_Expansion/'))

## Read files
gdf_urban = get_urban(ifile_urban_shp) # simplified ifile    

# S01 didn't have locations added to the nodes so I added them with add_geometry_to_HVAC_transformers_simple.py
if scenario_name == 'S01':
    df_HVAC_transformer_nodes_with_location_added = pd.read_pickle('outputs/combined_HVAC_location_simple.pkl')
    # Convert to gdf
    df_HVAC_transformer_nodes_with_location_added['geometry'] = df_HVAC_transformer_nodes_with_location_added.apply(
        lambda row: Point(row['longitude'], row['latitude']),
        axis=1
    )
    gdf_HVAC_transformer_nodes_with_location_added = gpd.GeoDataFrame(
        df_HVAC_transformer_nodes_with_location_added,
        geometry='geometry',
        crs='EPSG:4326'  # WGS84, standard lat/lon
    )
if scenario_name == 'S03':
    df_HVAC_transformer_nodes_with_location_added = pd.read_csv(datapath + '/R02_S03_Transmission_Expansion_CONUS-2.csv')
    # Filter the DataFrame to only include rows where the transformer type is "2TF"
    df_HVAC_transformer_nodes_with_location_added = df_HVAC_transformer_nodes_with_location_added[
    df_HVAC_transformer_nodes_with_location_added['Type'] == '2TF']
    # Convert WKT strings to shapely geometry objects (if not done already)
    df_HVAC_transformer_nodes_with_location_added['geometry'] = (
        df_HVAC_transformer_nodes_with_location_added['geometry']
        .apply(wkt.loads)
    )
    # Extract latitude (Y value) from the first point in the LINESTRING
    df_HVAC_transformer_nodes_with_location_added['latitude'] = (
        df_HVAC_transformer_nodes_with_location_added['geometry']
        .apply(lambda geom: geom.coords[0][1] if geom is not None else None)
    )
    df_HVAC_transformer_nodes_with_location_added['longitude'] = (
    df_HVAC_transformer_nodes_with_location_added['geometry']
    .apply(lambda geom: geom.coords[0][0] if geom is not None else None)
    )
    gdf_HVAC_transformer_nodes_with_location_added = gpd.GeoDataFrame(
    df_HVAC_transformer_nodes_with_location_added,
    geometry='geometry',
    crs='EPSG:4326'  # WGS84, standard lat/lon
    )
        


#%% Get coastal node locations
# Get the boundaries of the coastal counties
gdf_coastal_counties = gpd.read_file(coast_counties_ifile)
# Exclude outside of lower 48 for now
gdf_coastal_counties = gdf_coastal_counties[~gdf_coastal_counties["statename"].isin(["Guam", "Alaska", "Hawaii", "Commonwealth of the Northern Mariana Islands", "American Samoa", "United States Virgin Islands", "Puerto Rico"])]
# Reproject HVAC and coastal boundaries from degrees to meters
node_locations_HVAC = gdf_HVAC_transformer_nodes_with_location_added.to_crs('ESRI:102008')   
#gdf_coastal_us = gdf_coastal_us.to_crs('ESRI:102008')
gdf_coastal_counties = gdf_coastal_counties.to_crs('ESRI:102008')
# Extract the HVAC transformer locations in coastal (this is not number of transformers, see HICOamerica assumptions below) in coastal counties into a separate df
HVAC_nodes_within_coastal_counties = node_locations_HVAC[node_locations_HVAC.geometry.within(gdf_coastal_counties.unary_union)]
# Get the count of HVAC transformer nodes within the buffer
HVAC_nodes_within_coastal_counties_count = len(HVAC_nodes_within_coastal_counties)



#%% Get urban node locations
# Extract the HVAC transformer locations in urban areas
HVAC_nodes_within_urban_areas = node_locations_HVAC[node_locations_HVAC.geometry.within(gdf_urban.unary_union)]
#print(f"Number of locations with transformers within coastal counties: {nodes_within_coastal_counties_count}")
HVAC_nodes_within_urban_count = len(HVAC_nodes_within_urban_areas)

#%% Combine urban and coastal node locations
# merge both df and remove duplicated rows where id is the same
HVAC_nodes_within_both_areas = pd.concat([HVAC_nodes_within_coastal_counties, HVAC_nodes_within_urban_areas], ignore_index=True)
HVAC_nodes_within_both_areas = HVAC_nodes_within_both_areas.drop_duplicates(subset=['id'], keep = 'first')



#%% Find number of transformers that are not in urban or coastal areas but Vn[kV] contain 500 kV. These are also assumed to be SF6 switchgears

df_500kV_HVAC = node_locations_HVAC[node_locations_HVAC['Vn [kV]'].astype(str).str.contains('500', na=False)]
# polygons that are coastal and urban areas
gdf_coastal_and_urban = gpd.GeoDataFrame(pd.concat([gdf_urban, gdf_coastal_counties], ignore_index=True))
# check by plotting
fig, ax = plt.subplots()
gdf_coastal_and_urban.plot(ax=ax)
ax.set_title('Coastal and urban areas')

# Remove rows where geometry is in the urban and coastal areas
gdf_500kV_HVAC = gpd.GeoDataFrame(df_500kV_HVAC, geometry=gpd.points_from_xy(df_500kV_HVAC['longitude'], df_500kV_HVAC['latitude']))
gdf_500kV_HVAC = gdf_500kV_HVAC.set_crs(epsg=4326, allow_override=True)
gdf_500kV_HVAC = gdf_500kV_HVAC.to_crs("ESRI:102008")
gdf_coastal_and_urban = gdf_coastal_and_urban.to_crs('ESRI:102008')

# Spatial join: keep only points that match coastal & urban areas
points_in_coastal = gpd.sjoin(gdf_500kV_HVAC, gdf_coastal_and_urban, how='inner', predicate='within')
# Now, points outside coastal and urban areas:
gdf_500kV_HVAC_outside_urban_and_coastal = gdf_500kV_HVAC[~gdf_500kV_HVAC.index.isin(points_in_coastal.index)]

# Plot the coastal and urban areas with all 500 kV locations
fig, ax = plt.subplots()
gdf_coastal_and_urban.plot(ax=ax, label = 'Coastal and urban areas', alpha=0.7)
gdf_500kV_HVAC.plot(ax=ax, label='All 500 kV locations', color='red', markersize=100)
gdf_500kV_HVAC_outside_urban_and_coastal.plot(ax=ax, label='500 kV outside urban and coastal', color='green')
coastal_patch = mpatches.Patch(color='lightblue', alpha=0.7, label='Coastal and urban areas')
ax.legend(handles=[coastal_patch] + ax.get_legend_handles_labels()[0])
if scenario_name == 'S01':
    scenario_title = 'HVAC Scenario'
if scenario_name == 'S03': 
    scenario_title = 'MTHVDC Scenario'
ax.set_title(scenario_title)



#%% From the number of HVAC transformer node locations, now you need to calculate the number of HVAC transformers based on the Hicoamerica assumptions of capacity for the kV transformer rating

# Create a dataframe with the following assumptions from HICOamerica, since 2000/2400 MVA transformers do not exist
data = {
    "Voltage": ["345/138", "345/230", "500/230*", "500/345**", "765/500"],
    "Phase": [3, 3, 1, 1, 1],
    "Capacity (MVA)": [700, 700, 667, 667, 667],
    "Required units to achieve 2000 MVA": [3, 3, 9, 9, 9]
}
df_transformer_assumptions = pd.DataFrame(data)
# Standardizing voltage names
df_transformer_assumptions['Voltage_cleaned'] = df_transformer_assumptions['Voltage'].replace({
    "500/345**": "500-345", 
    "500/230*": "500-230", 
    "345/138": "345-138", 
    "345/230": "345-230", 
    "765/500": "765-500"
})

# Count number of HVAC node locations that will be multiplied with assumptions above
HVAC_count_coast = HVAC_nodes_within_coastal_counties['Vn [kV]'].value_counts()
HVAC_count_urban = HVAC_nodes_within_urban_areas['Vn [kV]'].value_counts()
HVAC_count_both = HVAC_nodes_within_both_areas['Vn [kV]'].value_counts()
df_500kV_HVAC_outside_urban_and_coastal_count = gdf_500kV_HVAC_outside_urban_and_coastal['Vn [kV]'].value_counts()


HVAC_count_coast.index = HVAC_count_coast.index.map(lambda x: f"{str(x)[:3]}-{str(x)[3:]}")
HVAC_count_urban.index = HVAC_count_urban.index.map(lambda x: f"{str(x)[:3]}-{str(x)[3:]}")
HVAC_count_both.index = HVAC_count_both.index.map(lambda x: f"{str(x)[:3]}-{str(x)[3:]}")
df_500kV_HVAC_outside_urban_and_coastal_count.index = df_500kV_HVAC_outside_urban_and_coastal_count.index.map(lambda x: f"{str(x)[:3]}-{str(x)[3:]}")


# Get number of transformers at each location.  Multiply the number of transformers per voltage by the number of transformers required for each voltage according to MVA capacity assumptions of HICOamerica

def calculate_transformers(df_assumptions, df_counts, label, scenario_name):
    # Merge assumptions with counts
    df_merged = pd.merge(df_assumptions, df_counts, left_on='Voltage_cleaned', right_index=True)
    # Calculate total required units
    df_merged['Total required units'] = df_merged['Required units to achieve 2000 MVA'] * df_merged['count']
    # Print results
    print(scenario_name)
    print(f'Total number of {label} transformers: {df_merged["Total required units"].sum()}')
    # Return relevant columns
    return df_merged[['Voltage', 'Required units to achieve 2000 MVA', 'count', 'Total required units']]

# Coast
df_merged_coast = calculate_transformers(df_transformer_assumptions, HVAC_count_coast, "coastal HVAC", scenario_name)

# Urban
df_merged_urban = calculate_transformers(df_transformer_assumptions, HVAC_count_urban, "urban HVAC", scenario_name)

# Both urban and coast, without duplicates
df_merged_both = calculate_transformers(df_transformer_assumptions, HVAC_count_both, "BOTH urban & coastal HVAC (no duplicates)", scenario_name)

# 500 kV transformers outside urban and coastal areas
df_merged_500kV_HVAC_outside_urban_and_coastal = calculate_transformers(
    df_transformer_assumptions, 
    df_500kV_HVAC_outside_urban_and_coastal_count, 
    "500 kV or higher HVAC outside urban and coastal areas", 
    scenario_name
)




# Combine the two DataFrames and sum the numbers for the same Voltage column
combined_df = pd.concat([df_merged_500kV_HVAC_outside_urban_and_coastal, df_merged_both])

# Group by the 'Voltage' column and sum the other columns
combined_df = combined_df.groupby('Voltage', as_index=False).sum()

# Print the resulting DataFrame
print(f'Total transformers that are either in coastal, urban, or 500 kV outside of coastal and urban.  Note that all 500 kV transformers (even if in a coastal and urban area should also be added when assuming all 500 kV are SF6 switchgears.  Just wanted to see here how many 500 kV are outside urban and coastal as an experiment.) {combined_df}')


print('Note you should include all 500 kV and up on the summary spreadsheet, not just the ones listed here')
print('Just make sure the number of those transformers below 500 kV (if any printed above) are added to the spreadsheet')