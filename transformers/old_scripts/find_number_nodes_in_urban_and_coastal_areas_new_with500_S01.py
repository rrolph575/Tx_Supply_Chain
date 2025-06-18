

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
from Number_transformers import get_number_transformers ## why does this create a plot???
from shapely.wkt import loads


#%% Inputs

# Define datafiles
node_locations_ifile = os.path.abspath(os.path.join(os.getcwd(), '..', 'Data/R02_Transmission_Expansion/CONUS_nodes_UPDATE.csv'))
coastal_shape_ifile = os.path.abspath(os.path.join(os.getcwd(), '..', 'Data/ne_10m_coastline/ne_10m_coastline.shp'))
coast_counties_ifile = os.path.abspath(os.path.join(os.getcwd(), '..', 'Data/Coastal_Counties/Coastal_Counties.shp'))
ifile_urban_shp = os.path.abspath(os.path.join(os.getcwd(), '..', 'Data/tl_2024_us_uac20/tl_2024_us_uac20.shp'))
datapath = os.path.abspath(os.path.join(os.getcwd(), '..', 'Data/R02_Transmission_Expansion/'))

# Read files
gdf_urban = get_urban(ifile_urban_shp) # simplified ifile    
node_locations_all = pd.read_csv(node_locations_ifile) # This needs to be filtered to just transformers.

# Define scenario name
scenario_name = 'S01'  # S01 is only HVAC

#%% Get the locations of the HVAC transformers (with voltages) for AC and DC scenarios
# Generated from add_geometry_to_HVAC_transformers.py
filtered_df_HVAC_transformers = pd.read_pickle(os.path.abspath(os.path.join(os.getcwd(), '..', 'outputs/combined_HVAC_location.pkl'))) 
filtered_df_HVAC_transformers['geometry'] = filtered_df_HVAC_transformers['geometry'].apply(loads)
filtered_df_HVAC_transformers = gpd.GeoDataFrame(filtered_df_HVAC_transformers, geometry='geometry')


# Set crs to lat lon projections in geometry column
node_locations_HVAC = filtered_df_HVAC_transformers.copy()  # this also needs the Vn[kV] so you can calculate the number of transformers using hicoAmerica assumptions
node_locations_HVAC = gpd.GeoDataFrame(node_locations_HVAC, geometry='geometry')
# Filter out the point (0, 0)
node_locations_HVAC = node_locations_HVAC[node_locations_HVAC.geometry != Point(0, 0)]
# Ensure CRS consistency
node_locations_HVAC_epsg4326 = node_locations_HVAC.set_crs('EPSG:4326', allow_override=True, inplace=True)


#%% Get coastal
# Get the boundaries of the coastal counties
gdf_coastal_counties = gpd.read_file(coast_counties_ifile)
# Exclude outside of lower 48 for now
gdf_coastal_counties = gdf_coastal_counties[~gdf_coastal_counties["statename"].isin(["Guam", "Alaska", "Hawaii", "Commonwealth of the Northern Mariana Islands", "American Samoa", "United States Virgin Islands", "Puerto Rico"])]


# Reproject HVAC and coastal boundaries from degrees to meters
node_locations_HVAC = node_locations_HVAC_epsg4326.to_crs('ESRI:102008')   
#gdf_coastal_us = gdf_coastal_us.to_crs('ESRI:102008')
gdf_coastal_counties = gdf_coastal_counties.to_crs('ESRI:102008')
# gdf_urban = gdf_urban.to_crs('ESRI:102008')   ## THis is already in ESRI:102008

# Extract the HVAC transformer locations in coastal (this is not number of transformers, see HICOamerica assumptions below) in coastal counties into a separate df
HVAC_nodes_within_coastal_counties = node_locations_HVAC[node_locations_HVAC.geometry.within(gdf_coastal_counties.unary_union)]

# Get the count of HVAC transformer nodes within the buffer
HVAC_nodes_within_coastal_counties_count = len(HVAC_nodes_within_coastal_counties)



#%% Get urban
# Extract the HVAC transformer locations in urban areas
HVAC_nodes_within_urban_areas = node_locations_HVAC[node_locations_HVAC.geometry.within(gdf_urban.unary_union)]

#print(f"Number of locations with transformers within coastal counties: {nodes_within_coastal_counties_count}")
HVAC_nodes_within_urban_count = len(HVAC_nodes_within_urban_areas)



#%% Both areas
# merge both df and remove duplicated rows where id is the same
HVAC_nodes_within_both_areas = pd.concat([HVAC_nodes_within_coastal_counties, HVAC_nodes_within_urban_areas], ignore_index=True)
HVAC_nodes_within_both_areas = HVAC_nodes_within_both_areas.drop_duplicates(subset=['id'], keep = 'first')



#%% Find number of transformers that are not in urban or coastal areas but Vn[kV] contains 500 kV

# Node: Later addition in code, so coded separately
df_500kV_HVAC = filtered_df_HVAC_transformers[filtered_df_HVAC_transformers['Vn [kV]'].astype(str).str.contains('500', na=False)]

## Find which ones are not in gdf_coastal_counties or gdf_urban
# Combine urban and coastal geometries
gdf_exclude = gpd.GeoDataFrame(pd.concat([gdf_urban, gdf_coastal_counties], ignore_index=True))
# Remove rows where geometry is in the excluded areas
df_500kV_HVAC = gpd.GeoDataFrame(df_500kV_HVAC, geometry='geometry')
df_500kV_HVAC = df_500kV_HVAC.set_crs('EPSG:4326', allow_override=True, inplace=True)
df_500kV_HVAC = df_500kV_HVAC.to_crs('ESRI:102008')
df_500kV_HVAC = df_500kV_HVAC.reset_index(drop=True)
df_500kV_HVAC_outside_urban_and_coastal = df_500kV_HVAC[
    ~df_500kV_HVAC.sjoin(gdf_exclude, how="left", predicate="intersects").index_right.notna()
]



#%% Plot the counties
fig, ax = plt.subplots(figsize=(10, 6))
gdf_coastal_counties.plot(ax=ax, edgecolor="grey", color="lightblue", label='coastal counties')
node_locations_HVAC.plot(ax=ax, color='blue', marker='x', markersize=5, label='Transformer')
#HVAC_nodes_within_coastal_counties.plot(ax=ax, color='orange', marker='o', markersize=10, label = 'Coastal Transformer locations')
#HVAC_nodes_within_urban_areas.plot(ax=ax, color='red', marker='o', markersize=10, label = 'Urban Transformer locations')
HVAC_nodes_within_both_areas.plot(ax=ax, color='yellow', marker='o', markersize=10, label = 'Both Urban & Coastal')
df_500kV_HVAC_outside_urban_and_coastal.plot(ax=ax, color='green', marker='o', markersize=10, label = 'Outside Urban & Coastal')
gdf_urban.plot(ax=ax, color='grey')
# Manually define legend
legend_elements = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor="lightblue", markersize=10, label="Coastal Counties"),
    Line2D([0], [0], marker="x", color="blue", markersize=10, label="Transformer"),
    #Line2D([0], [0], marker="o", color="orange", markersize=10, label="Coastal Transformer"),
    #Line2D([0], [0], marker="o", color="red", markersize=10, label="Urban Transformer"),
    Line2D([0], [0], marker="o", color="yellow", markersize=10, label="Urban plus Coastal"),
    Line2D([0], [0], marker="o", color="green", markersize=10, label="Outside Urban plus Coastal")
]

# Add the legend
ax.legend(handles=legend_elements)
plt.title(scenario_name + ' Node locations HVAC')
#ax.legend()
plt.show()



# From the number of HVAC transformer node locations, now you need to calculate the number of HVAC transformers based on the Hicoamerica assumptions of capacity for the kV transformer rating

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
df_500kV_HVAC_outside_urban_and_coastal_count = df_500kV_HVAC_outside_urban_and_coastal['Vn [kV]'].value_counts()


HVAC_count_coast.index = HVAC_count_coast.index.map(lambda x: f"{str(x)[:3]}-{str(x)[3:]}")
HVAC_count_urban.index = HVAC_count_urban.index.map(lambda x: f"{str(x)[:3]}-{str(x)[3:]}")
HVAC_count_both.index = HVAC_count_both.index.map(lambda x: f"{str(x)[:3]}-{str(x)[3:]}")
df_500kV_HVAC_outside_urban_and_coastal_count.index = df_500kV_HVAC_outside_urban_and_coastal_count.index.map(lambda x: f"{str(x)[:3]}-{str(x)[3:]}")


# Get number of transformers at each location.  Multiply the number of transformers per voltage by the number of transformers required for each voltage according to MVA capacity assumptions of HICOamerica

## Coast
df_merged_coast = pd.merge(df_transformer_assumptions, HVAC_count_coast, left_on='Voltage_cleaned', right_index=True)
df_merged_coast['Total required units'] = df_merged_coast['Required units to achieve 2000 MVA'] * df_merged_coast['count']
df_merged_coast[['Voltage', 'Required units to achieve 2000 MVA', 'count', 'Total required units']]
print(scenario_name)
print('The number of coastal HVAC transformers, using HicoAmerica assumptions to reach 2000 MVA ' + str(df_merged_coast['Total required units'].sum()))

## Urban
df_merged_urban = pd.merge(df_transformer_assumptions, HVAC_count_urban, left_on='Voltage_cleaned', right_index=True)
df_merged_urban['Total required units'] = df_merged_urban['Required units to achieve 2000 MVA'] * df_merged_urban['count']
df_merged_urban[['Voltage', 'Required units to achieve 2000 MVA', 'count', 'Total required units']]
print(scenario_name)
print('The number of urban HVAC transformers, using HicoAmerica assumptions to reach 2000 MVA ' + str(df_merged_urban['Total required units'].sum()))

## Both urban and coast, without duplicates
df_merged_both = pd.merge(df_transformer_assumptions, HVAC_count_both, left_on='Voltage_cleaned', right_index=True)
df_merged_both['Total required units'] = df_merged_both['Required units to achieve 2000 MVA'] * df_merged_both['count']
df_merged_both[['Voltage', 'Required units to achieve 2000 MVA', 'count', 'Total required units']]
print(scenario_name)
print('The number of BOTH urban & coastal HVAC transformers (no duplicates), using HicoAmerica assumptions to reach 2000 MVA ' + str(df_merged_both['Total required units'].sum()))


## The number of 500 kV transformers outside of urban and coastal
df_merged_500kV_HVAC_outside_urban_and_coastal = pd.merge(df_transformer_assumptions, df_500kV_HVAC_outside_urban_and_coastal_count, left_on='Voltage_cleaned', right_index=True)
df_merged_500kV_HVAC_outside_urban_and_coastal['Total required units'] = df_merged_500kV_HVAC_outside_urban_and_coastal['Required units to achieve 2000 MVA']*df_merged_500kV_HVAC_outside_urban_and_coastal['count']
df_merged_500kV_HVAC_outside_urban_and_coastal[['Voltage', 'Required units to achieve 2000 MVA', 'count', 'Total required units']]
print(scenario_name)
print('Total number of transformers 500 kV or higher outside of urban and coastal areas:' + str(df_merged_500kV_HVAC_outside_urban_and_coastal['Total required units'].sum()))




#%% # Print the percent of transformers that are either urban and coastal
## These are taken from runs of Number_transformers.py. 
# HVAC scenario
number_hvac_transformers_for_scenario1 = 1860
# MTHVDC scenario
number_converter_transformers_for_scenario3 = 960
number_hvac_transformers_for_scenario3 = 474

percent_urban_and_coastal_HVAC = df_merged_both['Total required units'].sum()/number_hvac_transformers_for_scenario1 * 100
print(f'Percent urban and coastal HVAC transformers for scenario {scenario_name}: {percent_urban_and_coastal_HVAC}')
#print('Percent urban and coastal HVAC transformers for scenario ' + str(scenario_name) + str(percent_urban_and_coastal_HVAC))
total_500kV_and_higher_outside_urban_and_coastal = df_merged_500kV_HVAC_outside_urban_and_coastal['Total required units'].sum()
total_sf6_transformers = df_merged_both['Total required units'].sum() + total_500kV_and_higher_outside_urban_and_coastal
print('Total number of transformers 500 kV or higher outside of urban and coastal areas:' + str(total_500kV_and_higher_outside_urban_and_coastal))
print(f'Percent HVAC 500 kV or higher outside of urban and coastal areas: {scenario_name}: {total_500kV_and_higher_outside_urban_and_coastal/number_hvac_transformers_for_scenario1*100}')
print(f'Percent HVAC total SF6 transformers: {total_sf6_transformers/number_hvac_transformers_for_scenario1 * 100}')


#total_sf6_transformers = df_merged_both['Total required units'].sum() + total_500kV_and_higher_outside_urban_and_coastal
print(f'breakdown of transformers 500 kV and higher outside urban/coast: {df_merged_500kV_HVAC_outside_urban_and_coastal}')

print(f'breakdown of urban and coastal transformers: {df_merged_both}')



# Combine the two DataFrames and sum the numbers for the same Voltage column
combined_df = pd.concat([df_merged_500kV_HVAC_outside_urban_and_coastal, df_merged_both])

# Group by the 'Voltage' column and sum the other columns
combined_df = combined_df.groupby('Voltage', as_index=False).sum()

# Print the resulting DataFrame
print(combined_df)