

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



# Define datafiles
node_locations_ifile = 'Data/R02_Transmission_Expansion/CONUS_nodes_UPDATE.csv'
coastal_shape_ifile = 'Data/ne_10m_coastline/ne_10m_coastline.shp'
coast_counties_ifile = 'Data/Coastal_Counties/Coastal_Counties.shp'
ifile_urban_shp = 'Data/tl_2024_us_uac20/tl_2024_us_uac20.shp'   
gdf_urban = get_urban(ifile_urban_shp) # simplified ifile    
node_locations_all = pd.read_csv(node_locations_ifile) # This needs to be filtered to just transformers.
basepath = "C:/Users/rrolph/OneDrive - NREL/Projects/FY25/Transmission_Supply_Chain/"
datapath = basepath + "Data/R02_Transmission_Expansion/"



# Read in files
scenario_name = 'S03'  # S01 is only HVAC and S03 has both HVDC and HVAC.
if scenario_name == 'S01':
    filenames = ['R02_S01_Transmission_Expansion_EI-1', 'R02_S01_Transmission_Expansion_ERCOT-1', 'R02_S01_Transmission_Expansion_WECC-1']
if scenario_name == 'S03':
    filenames = ['R02_S03_Transmission_Expansion_CONUS-2']
dfs = []
# Combine df from the S01 scenarios
for file in filenames:
    df = pd.read_csv(datapath + file + '.csv')
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)



##### Get the locations of the HVAC transformers (with voltages) for AC and DC scenarios

#   If scenario 1, the node locations are not given, just the FromBus and ToBus (where the transformer windings are). 
#   For now, I am assuming the transformer is located at the 'ToBus' node. Have to map the 'ToBus' node number to the CONUS_nodes_updates.csv file.
if scenario_name=='S01':
    # Read all of the files 
    filtered_df_HVAC_transformers = pd.read_pickle('combined_HVAC_location.pkl') # Generated from add_geometry_to_HVAC_transformers.py
    filtered_df_HVAC_transformers['geometry'] = filtered_df_HVAC_transformers['geometry'].apply(loads)
    # Ensure it's a GeoDataFrame
    filtered_df_HVAC_transformers = gpd.GeoDataFrame(filtered_df_HVAC_transformers, geometry='geometry')

#   If scenario 3, the node locations for HVAC transformers are given in filtered_df_HVAC_transformers['geometry']   
if scenario_name == 'S03':
    # Get the nodes df
    filtered_df, filtered_df_HVAC_transformers = get_number_transformers(df_all, scenario_name)
    filtered_df_HVAC_transformers['geometry'] = filtered_df_HVAC_transformers['geometry'].apply(loads)  # Convert WKT to geometry
    filtered_df_HVAC_transformers['geometry'] = filtered_df_HVAC_transformers['geometry'].apply(lambda geom: Point(geom.coords[0]) if geom and not geom.is_empty else None)


### Set crs to lat lon projections in geometry column
# These node_locations should be the filtered only HVAC transformers
node_locations_HVAC = filtered_df_HVAC_transformers.copy()  # this also needs the Vn[kV] so you can calculate the number of transformers using hicoAmerica assumptions
node_locations_HVAC = gpd.GeoDataFrame(node_locations_HVAC, geometry='geometry')
# Filter out the point (0, 0)
node_locations_HVAC = node_locations_HVAC[node_locations_HVAC.geometry != Point(0, 0)]
# Reorder columns so that longitude comes first
#node_locations['geometry'] = node_locations['geometry'].apply(lambda point: Point(point.y, point.x))
# Ensure CRS consistency.
node_locations_HVAC_epsg4326 = node_locations_HVAC.set_crs('EPSG:4326', allow_override=True, inplace=True)


# Get HVDC node locations
if scenario_name == 'S03':
    df_all_HVDC_transformer_locations = filtered_df[filtered_df['Rate1[MVA]'].isin([4000, 2000])]


#### Coastal #### 
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
# Extract the HVAC transformer locations in urban areas
HVAC_nodes_within_urban_areas = node_locations_HVAC[node_locations_HVAC.geometry.within(gdf_urban.unary_union)]
# merge both df and remove duplicated rows where id is the same
HVAC_nodes_within_both_areas = pd.concat([HVAC_nodes_within_coastal_counties, HVAC_nodes_within_urban_areas], ignore_index=True)
HVAC_nodes_within_both_areas = HVAC_nodes_within_both_areas.drop_duplicates(subset=['id'], keep = 'first')
### !!! Get total number of HVAC nodes everywehere that are 500 kV and up
total_500andhigher_in_dataset_HVAC_locations = node_locations_HVAC[node_locations_HVAC['Vn [kV]'].astype(str).str.contains('500', na=False)]



# Get the count of HVAC transformer nodes within the buffer
HVAC_nodes_within_coastal_counties_count = len(HVAC_nodes_within_coastal_counties)
#print(f"Number of locations with transformers within coastal counties: {nodes_within_coastal_counties_count}")
HVAC_nodes_within_urban_count = len(HVAC_nodes_within_urban_areas)


#####  Check HVAC transformer locations on coast by plotting
# Plot (on one plot):
#   - locations of all HVAC transformers # node_locations
#   - all HVAC transformers in coastal counties # nodes_within_coastal_counties
#   - plot coastal county boundaries  # gdf_coastal_counties

# Plot the counties
fig, ax = plt.subplots(figsize=(10, 6))
gdf_coastal_counties.plot(ax=ax, edgecolor="grey", color="lightblue", label='coastal counties')
node_locations_HVAC.plot(ax=ax, color='blue', marker='x', markersize=5, label='Transformer')
#HVAC_nodes_within_coastal_counties.plot(ax=ax, color='orange', marker='o', markersize=10, label = 'Coastal Transformer locations')
#HVAC_nodes_within_urban_areas.plot(ax=ax, color='red', marker='o', markersize=10, label = 'Urban Transformer locations')
HVAC_nodes_within_both_areas.plot(ax=ax, color='yellow', marker='o', markersize=10, label = 'Both Urban & Coastal')
# Manually define legend
legend_elements = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor="lightblue", markersize=10, label="Coastal Counties"),
    Line2D([0], [0], marker="x", color="blue", markersize=10, label="Transformer"),
    #Line2D([0], [0], marker="o", color="orange", markersize=10, label="Coastal Transformer"),
    #Line2D([0], [0], marker="o", color="red", markersize=10, label="Urban Transformer"),
    Line2D([0], [0], marker="o", color="yellow", markersize=10, label="Urban plus Coastal")
]

# Add the legend
ax.legend(handles=legend_elements)
plt.title(scenario_name + ' Node locations HVAC')
#ax.legend()
plt.show()



###### From the number of HVAC transformer node locations, now you need to calculate the number of HVAC transformers based on the Hicoamerica assumptions of capacity for the kV transformer rating

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
HVAC_500andHigher_count = total_500andhigher_in_dataset_HVAC_locations['Vn [kV]'].value_counts()

HVAC_count_coast.index = HVAC_count_coast.index.map(lambda x: f"{str(x)[:3]}-{str(x)[3:]}")
HVAC_count_urban.index = HVAC_count_urban.index.map(lambda x: f"{str(x)[:3]}-{str(x)[3:]}")
HVAC_count_both.index = HVAC_count_both.index.map(lambda x: f"{str(x)[:3]}-{str(x)[3:]}")
HVAC_500andHigher_count.index = HVAC_500andHigher_count.index.map(lambda x: f"{str(x)[:3]}-{str(x)[3:]}")


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
### Total 500 and up 
df_merged_500andup = pd.merge(df_transformer_assumptions, HVAC_500andHigher_count, left_on='Voltage_cleaned', right_index=True)
df_merged_500andup['Total required units'] = df_merged_500andup['Required units to achieve 2000 MVA'] * df_merged_500andup['count']
total_number_HVAC_transformers_500andup = df_merged_500andup['Total required units'].sum()


####### Get the node locations of HVDC transformers in coastal areas ################
if scenario_name == 'S03':
    # Get node locations of HVDC transformers. Separate the geometries into two columns so the ends of the HVDC lines (which correspond to transformer locations) are separated to determine if they are in coastal or urban locations. 
    # Convert LINESTRING text to shapely LineString objects
    df_all_HVDC_transformer_locations['geometry'] = df_all_HVDC_transformer_locations['geometry'].apply(loads)

    # Extract first and second coordinates. These are the points at each end of the line.
    df_all_HVDC_transformer_locations['coords1'] = df_all_HVDC_transformer_locations['geometry'].apply(lambda line: line.coords[0])
    df_all_HVDC_transformer_locations['coords2'] = df_all_HVDC_transformer_locations['geometry'].apply(lambda line: line.coords[1])

    # Plot the HVDC locations
    # Convert coords into Point geometry
    df_all_HVDC_transformer_locations['coords1'] = df_all_HVDC_transformer_locations['coords1'].apply(lambda coord: Point(coord))
    df_all_HVDC_transformer_locations['coords2'] = df_all_HVDC_transformer_locations['coords2'].apply(lambda coord: Point(coord))
    # 
    transformer_locations_HVDC_side1 = gpd.GeoDataFrame(df_all_HVDC_transformer_locations, geometry='coords1')
    transformer_locations_HVDC_side2 = gpd.GeoDataFrame(df_all_HVDC_transformer_locations, geometry='coords2')

    # Plot the locations of HVDC transformers. Have to do this separate for each side1 and side2
    transformer_locations_HVDC_side1_epsg4326 = transformer_locations_HVDC_side1.set_crs('EPSG:4326', allow_override=True, inplace=True)
    transformer_locations_HVDC_side2_epsg4326 = transformer_locations_HVDC_side2.set_crs('EPSG:4326', allow_override=True, inplace=True)

    transformer_locations_HVDC_side1 = transformer_locations_HVDC_side1_epsg4326.to_crs('ESRI:102008') 
    transformer_locations_HVDC_side2 = transformer_locations_HVDC_side2_epsg4326.to_crs('ESRI:102008')   
    #transformer_locations_HVDC_side1and2 = pd.concat([transformer_locations_HVDC_side1, transformer_locations_HVDC_side2], ignore_index=True)
    # gdf_coastal_counties = gdf_coastal_counties.to_crs('ESRI:102008')
    # gdf_urban = gdf_urban.to_crs('ESRI:102008')   ## THis is already in ESRI:102008

    ### Find all the DC transformers that are 500 and above
    HVDC_locations_500kV_and_higher_total_side1 = transformer_locations_HVDC_side1[transformer_locations_HVDC_side1['Vn [kV]'].astype(str).str.contains('500', na=False)]
    HVDC_locations_500kV_and_higher_total_side2 = transformer_locations_HVDC_side2[transformer_locations_HVDC_side2['Vn [kV]'].astype(str).str.contains('500', na=False)]



    #### Find the DC transformers in coastal and urban (and both areas) 
    hvdc_transformer_locations_within_coastal_counties1 = transformer_locations_HVDC_side1[transformer_locations_HVDC_side1.geometry.within(gdf_coastal_counties.unary_union)]
    hvdc_transformer_locations_within_coastal_counties2 = transformer_locations_HVDC_side2[transformer_locations_HVDC_side2.geometry.within(gdf_coastal_counties.unary_union)]
    hvdc_transformer_locations_within_urban1 = transformer_locations_HVDC_side1[transformer_locations_HVDC_side1.geometry.within(gdf_urban.unary_union)]
    hvdc_transformer_locations_within_urban2 = transformer_locations_HVDC_side2[transformer_locations_HVDC_side2.geometry.within(gdf_urban.unary_union)]
    # Find converter transformers that are in both urban and coastal together
    hvdc_nodes_within_coastal_counties = pd.concat([hvdc_transformer_locations_within_coastal_counties1.drop(columns=['coords2']), hvdc_transformer_locations_within_coastal_counties2.drop(columns=['coords1'])], ignore_index=True)
    hvdc_nodes_within_urban = pd.concat([hvdc_transformer_locations_within_urban1.drop(columns=['coords2']), hvdc_transformer_locations_within_urban2.drop(columns=['coords1'])], ignore_index=True)
    
    # Have to drop duplicates that are the same lat lon, not the same id, because there are same id's for each line, each line has 2 ends which correspond to different converter-transformers
    df_hvdc_nodes_urban_and_coastal = pd.concat([hvdc_nodes_within_coastal_counties, hvdc_nodes_within_urban], ignore_index=True)   
    # fill in coords1 where none with coords2 values
    df_hvdc_nodes_urban_and_coastal['geometry_transformer_locations'] = df_hvdc_nodes_urban_and_coastal['coords1'].fillna(df_hvdc_nodes_urban_and_coastal['coords2'])
    # drop duplicate node locations only if they are the same line.  some lines are different but have the same geometry_transformer_location becuase different line ends can be in the same place
    hvdc_nodes_within_both_areas = df_hvdc_nodes_urban_and_coastal.drop_duplicates(subset=['FromBus', 'ToBus', 'geometry_transformer_locations'], keep='first')
    hvdc_nodes_within_both_areas = hvdc_nodes_within_both_areas.set_geometry('geometry_transformer_locations')
    # hvdc_nodes_within_both_areas = hvdc_nodes_within_both_areas.drop_duplicates(subset=['id'], keep = 'first')
    hvdc_nodes_within_both_areas = hvdc_nodes_within_both_areas.to_crs('ESRI:102008')

    # Plot the DC transformers in coastal counties 
    fig, ax = plt.subplots(figsize=(10, 6))
    gdf_coastal_counties.plot(ax=ax, edgecolor="grey", color="lightblue", label='coastal counties')
    #transformer_locations_HVDC_side1and2.plot(ax=ax, color='blue', marker='x', markersize=5, label='Converter-Transformer')
    transformer_locations_HVDC_side1.plot(ax=ax, color='blue', marker='x', markersize=5, label='Converter-Transformer')
    transformer_locations_HVDC_side2.plot(ax=ax, color='blue', marker='x', markersize=5, label='Converter-Transformer')
    #hvdc_transformer_locations_within_coastal_counties1.plot(ax=ax, color='orange', marker='o', markersize=25, label = 'Coastal Converter-Transformer')
    #hvdc_transformer_locations_within_coastal_counties2.plot(ax=ax, color='orange', marker='o', markersize=25, label = 'Coastal Converter-Transformer')
    #hvdc_transformer_locations_within_urban1.plot(ax=ax, color='red', marker='o', markersize=25, label = 'Urban Converter-Transformer')
    #hvdc_transformer_locations_within_urban2.plot(ax=ax, color='red', marker='o', markersize=25, label = 'Urban Converter-Transformer')
    hvdc_nodes_within_both_areas.plot(ax=ax, color='yellow', marker='o', markersize=25, label = 'Urban & Coastal Converter-Transformer')
    HVDC_locations_500kV_and_higher_total_side1.plot(ax=ax, color='green')
    HVDC_locations_500kV_and_higher_total_side2.plot(ax=ax, color='green')
    # Manually define legend
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="lightblue", markersize=10, label="Coastal Counties"),
        Line2D([0], [0], marker="x", color="blue", markersize=10, label="Converter-Transformer"),
        #Line2D([0], [0], marker="o", color="orange", markersize=10, label="Coastal Converter-Transformer"),
        #Line2D([0], [0], marker="o", color="red", markersize=10, label="Urban Converter-Transformer"),
        Line2D([0], [0], marker="o", color="yellow", markersize=10, label="Coastal plus Urban Converter-Transformer")
    ]

    # Add the legend
    ax.legend(handles=legend_elements)
    plt.title(scenario_name + ' Converter Transformer locations')
    #ax.legend()
    plt.show()

    number_converter_transformers_per_HVDC_line_end = 12
    # Calculate the total number of converter transformers:
    total_number_converter_transformers_coastal = number_converter_transformers_per_HVDC_line_end*(hvdc_transformer_locations_within_coastal_counties1.shape[0] + hvdc_transformer_locations_within_coastal_counties2.shape[0])
    print('Total number converter-transformers in coastal areas ' + str(scenario_name) + ': ' + str(total_number_converter_transformers_coastal))
    total_number_converter_transformers_urban = number_converter_transformers_per_HVDC_line_end*(hvdc_transformer_locations_within_urban1.shape[0] + hvdc_transformer_locations_within_urban2.shape[0])
    print('Total number converter-transformers in urban areas ' + str(scenario_name) + ': ' + str(total_number_converter_transformers_urban))
    total_number_converter_transformers_both = number_converter_transformers_per_HVDC_line_end * hvdc_nodes_within_both_areas.shape[0]
    print('Total number converter-transformers both urban plus coastal (no duplicates)' + str(total_number_converter_transformers_both))

    total_number_500andup = number_converter_transformers_per_HVDC_line_end*(HVDC_locations_500kV_and_higher_total_side1.shape[0] + HVDC_locations_500kV_and_higher_total_side2.shape[0])
## These are taken from runs of Number_transformers.py:
# HVAC scenario
number_hvac_transformers_for_scenario1 = 1860
# MTHVDC scenario
number_converter_transformers_for_scenario3 = 960
number_hvac_transformers_for_scenario3 = 474

# Print the percent of transformers that are either urban and coastal

if scenario_name == 'S01':
    percent_urban_and_coastal_HVAC = df_merged_both['Total required units'].sum()/number_hvac_transformers_for_scenario1 * 100
    print(f'Percent urban and coastal HVAC transformers for scenario {scenario_name}: {percent_urban_and_coastal_HVAC}')
    #print('Percent urban and coastal HVAC transformers for scenario ' + str(scenario_name) + str(percent_urban_and_coastal_HVAC))

if scenario_name == 'S03':
    percent_urban_and_coastal_HVAC = df_merged_both['Total required units'].sum()/number_hvac_transformers_for_scenario3 * 100
    print('Percent urban and coastal HVAC transformers for scenario ' + str(scenario_name) +': ' + str(percent_urban_and_coastal_HVAC))
    percent_urban_and_coastal_HVDC = total_number_converter_transformers_both/number_converter_transformers_for_scenario3 * 100
    print('Percent urban and coastal MTHVDC converter-transformers for scenario ' + str(scenario_name) + ': ' + str(percent_urban_and_coastal_HVDC))


###### To get the total number of 500 kV and higher outside of the coast and urban = number of 500 in the total dataset - number in both (above)
# Then total number of SF6 = total number of 500 kV outside of coast and urban + number in both

print('the following is only valid for scenario 3')
# HVAC 
#total_500andhigher_in_dataset_HVAC_locations = node_locations_HVAC[node_locations_HVAC['Vn [kV]'].astype(str).str.contains('500', na=False)]
total_number_HVAC_transformers_500andup = df_merged_500andup['Total required units'].sum()

HVAC_total_number_500kV_and_higher_outside_coast_and_urban = total_number_HVAC_transformers_500andup - df_merged_both['Total required units'].sum()
total_number_SF6_HVAC = HVAC_total_number_500kV_and_higher_outside_coast_and_urban + df_merged_both['Total required units'].sum()

# HVDC
#total_500andhigher_in_dataset_HVDC = 
HVDC_500kV_and_higher_outside_coast_and_urban = total_number_500andup - total_number_converter_transformers_both
total_number_SF6_HVDC = HVDC_500kV_and_higher_outside_coast_and_urban + total_number_converter_transformers_both 