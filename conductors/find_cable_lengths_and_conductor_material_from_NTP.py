
# Find cable lengths from NTP study, based on location of buses
# Also calculates the kg of Aluminum and Steel for each cable rating.
# !! Note.. some nicer plots are in plot_cable_lengths.py

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
from matplotlib.ticker import MaxNLocator
from shapely.geometry import Point, LineString
import re

### Define paths
basepath = 'C:/Users/rrolph/OneDrive - NREL/Projects/FY25/Transmission_Supply_Chain/'
datapath = basepath + 'Data/R02_Transmission_Expansion/'
# plotpath = basepath + ''
percent_added_to_compensate_for_straightline = .3  # 30% increase
percent_added_to_compensate_for_sag = .04 # 4% increase
total_multiplier_for_length = 1 + percent_added_to_compensate_for_straightline + percent_added_to_compensate_for_sag
scenario_name = 'S03'
use_avg = False # False if you want to use specified number of bundles per cable, True if average number of bundles per cable. (Cable types are consistently Bluejay, even with NTP assumptions.)


### Read data
if scenario_name == 'S01':
    filenames = ['R02_S01_Transmission_Expansion_EI-1', 'R02_S01_Transmission_Expansion_ERCOT-1', 'R02_S01_Transmission_Expansion_WECC-1']
if scenario_name == 'S03':
    filenames = ['R02_S03_Transmission_Expansion_CONUS-2']


dfs = []
# Combine df from the S01 scenarios
for file in filenames:
    df = pd.read_csv(datapath + file + '.csv')
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)  # df1
# rename
linestring_df = df_all


# Load node locations
node_locations_file = datapath + 'CONUS_nodes_UPDATE.csv'
node_locations_df = pd.read_csv(node_locations_file)  # df2



### Calculate lengths HVAC
# Filter to HVAC where column 'Type' = OHL
OHL_df = linestring_df[(linestring_df['Type']=='OHL') & (linestring_df['Add [Y/N]']=='Y')]




# Function to clean the WKT string by removing the comma between coordinates
def clean_wkt_string(wkt_string):
    # Replace the comma in the WKT string with a space
    cleaned_string = wkt_string.replace(',', ' ')
    return cleaned_string


def correct_coordinates(point):
    if isinstance(point, Point):
        # Get the current lat/lon (lat is x and lon is y for reversed coordinates)
        lon, lat = point.x, point.y
        # Return a new Point with the reversed coordinates
        return Point(lat, lon)
    return point  # Return the original value if it's not a Point



### If HVAC scenario 1, add FromBus and ToBus locations to OHL_df into new columns called 
# FromBusLocation and ToBusLocation
if scenario_name == 'S01':
    # Extract last 6 digits of `node_id` in `node_locations_df`
    node_locations_df['last_6_digits_node_id'] = node_locations_df['node_id'].astype(str).str[-6:]

    ## Merge node_locations_df with OHL_df based on the last 6 digits for FromBus and ToBus
    # FromBus
    OHL_df = OHL_df.merge(
        node_locations_df[['last_6_digits_node_id', 'geometry']], 
        left_on=OHL_df['FromBus'].astype(str).str[-6:], 
        right_on='last_6_digits_node_id', 
        how='left'
    ).rename(columns={'geometry': 'FromBusLocation'}).drop(columns=['last_6_digits_node_id'])

    # ToBus 
    OHL_df = OHL_df.merge(
        node_locations_df[['last_6_digits_node_id', 'geometry']], 
        left_on=OHL_df['ToBus'].astype(str).str[-6:], 
        right_on='last_6_digits_node_id', 
        how='left'
    ).rename(columns={'geometry': 'ToBusLocation'}).drop(columns=['last_6_digits_node_id'])


    # Apply the cleaning function to the 'FromBusLocation' and 'ToBusLocation' columns
    OHL_df['FromBusLocation'] = OHL_df['FromBusLocation'].apply(lambda x: clean_wkt_string(x) if isinstance(x, str) else x)
    OHL_df['ToBusLocation'] = OHL_df['ToBusLocation'].apply(lambda x: clean_wkt_string(x) if isinstance(x, str) else x)


    # Now convert the cleaned WKT strings to Point objects
    OHL_df['FromBusLocation'] = OHL_df['FromBusLocation'].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)
    OHL_df['ToBusLocation'] = OHL_df['ToBusLocation'].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)

    # Reverse the coords so lat and lon are in the correct positions
    OHL_df['FromBusLocation'] = OHL_df['FromBusLocation'].apply(correct_coordinates)
    OHL_df['ToBusLocation'] = OHL_df['ToBusLocation'].apply(correct_coordinates)

    # For those lines where Length[km]=0, add 'Geometry' column so you can calculate the distance of the lines. 
    # For some lines where Length[km] is given, there is not a way to calculate the geometry/distance becuase 
    # some 'ToBus' node locations are not provided, which should be fine becuase the lengths are already given 
    # for those lines.

    OHL_df['geometry'] = OHL_df.apply(lambda row: LineString([ 
                            (row['FromBusLocation'].x, row['FromBusLocation'].y),  # (longitude, latitude)
                            (row['ToBusLocation'].x, row['ToBusLocation'].y)      # (longitude, latitude)
                        ]) if pd.notna(row['FromBusLocation']) and pd.notna(row['ToBusLocation'])
                        else None, axis=1)



### Calculate the lengths of each HVAC lines
OHL_df['geometry'] = OHL_df['geometry'].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)
# Create geodataframe
gdf_OHL = gpd.GeoDataFrame(OHL_df, geometry='geometry')
# Set the CRS to WGS84 (lat/lon)
gdf_OHL.set_crs('EPSG:4326', allow_override=True, inplace=True)

# Reproject to a suitable CRS for distance calculations (e.g., EPSG 3857)
#gdf_OHL = gdf_OHL.to_crs(epsg=3857)  
gdf_OHL = gdf_OHL.to_crs(epsg=32614) # meters that are more focused on center US

# Calculate the distance of each LineString geometry
gdf_OHL['distance'] = gdf_OHL.geometry.length/1e3*total_multiplier_for_length # Convert meters to km and apply length multipliers
gdf_OHL['distance'] = gdf_OHL['distance']*0.62 # convert km to miles


# !! Added on 1 May 2025.  If the lengths are provided already, then use those and not the calculated distance 
if scenario_name == 'S01':
    gdf_OHL['distance'] = gdf_OHL.apply(lambda row: row['Length[km]'] if row['Length[km]'] != 0 else row['distance'], axis=1
)

# If the lengths are not provided in S01, you also don't have the coordinates of both endpoints, so use the lengths
# provided in the S01 dataset.  This is not for very many points.
if scenario_name == 'S01':
    gdf_OHL['distance'] = gdf_OHL['Length[km]'].where(gdf_OHL['Length[km]'] != 0, gdf_OHL['distance'])*0.62*total_multiplier_for_length # Convert from km to miles
    # This is only happening once



### Remove the long line lengths created by duplicate matching last 
### 6 digit identifiers from conus nodes location file and original datafile
### that does not contain all lat lon information
# Drop rows with NaN in 'id' or 'distance' columns
gdf_cleaned = gdf_OHL.dropna(subset=['id', 'distance'])
# Identify rows where 'id' is not duplicated
non_duplicated = gdf_cleaned[gdf_cleaned.duplicated('id', keep=False) == False]
# For the duplicated 'id' values, select the row with the lowest 'distance' value
duplicated = gdf_cleaned[gdf_cleaned.duplicated('id', keep=False)]
lowest_distance = duplicated.loc[duplicated.groupby('id')['distance'].idxmin()]
# Concatenate both the non-duplicated and the lowest distance rows
final_gdf = pd.concat([non_duplicated, lowest_distance])
# Sort the final dataframe based on the 'id' 
final_gdf = final_gdf.sort_values(by='id')
# Reset index 
gdf_OHL = final_gdf.reset_index(drop=True)



### Plot the HVAC lengths results, separated by Vn[kV]
# Group by 'Vn [kV]' and calculate the mean or sum of 'distance'
# Convert 'Vn [kV]' values to integers if they are not already integers
gdf_OHL['Vn [kV]'] = gdf_OHL['Vn [kV]'].apply(lambda x: int(x) if not isinstance(x, int) else x)
grouped_data = gdf_OHL.groupby('Vn [kV]')['distance'].sum().reset_index() 
grouped_data = grouped_data.sort_values(by='Vn [kV]')
# Create the bar plot
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 18})
# Create the bar plot with 'Vn [kV]' as x and 'distance' as y
plt.bar(grouped_data['distance'].index, grouped_data['distance']/1e3, color='skyblue')
# Set x-ticks to correspond to 'Vn [kV]' labels
plt.xticks(grouped_data['distance'].index, grouped_data['Vn [kV]'])
# Add labels and title
plt.xlabel('Voltage [kV]')
plt.ylabel('Total Distance HVAC [thosands of miles]')
plt.ylim([0,110])
#plt.title('Total line lengths HVAC (Demand through 2035, NTP data)')
plt.tight_layout()
# Show the plot
plt.show()



# Calculate total length of HVAC cable in km
total_AC = grouped_data['distance'].sum()
print(scenario_name)
print('Total AC cable lengths in miles ' + str(total_AC))
print('Total AC lengths in miles grouped by voltage: ', str(grouped_data))

""" ### Plot the differences in lengths
if scenario_name == 'S03':  # Only applicable to S03, because the Length[km] had to be used for S01 (see comment above)
    # For those lengths available from the dataset, calculate the absolute differences
    gdf.loc[gdf['Length[km]'] != 0, 'Difference_in_length'] = gdf['Length[km]'] - gdf['distance']
    # Group by 'Vn [kV]' and calculate the mean or sum of 'distance'
    grouped_data_diff = gdf.groupby('Vn [kV]')['Difference_in_length'].sum().reset_index()
    grouped_data_diff = grouped_data_diff.sort_values(by='Vn [kV]')
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 18})
    # Create the bar plot with 'Vn [kV]' as x and 'distance' as y
    plt.bar(grouped_data_diff['Difference_in_length'].index, grouped_data_diff['Difference_in_length']/grouped_data['distance']*100, color='skyblue')
    # Set x-ticks to correspond to 'Vn [kV]' labels
    plt.xticks(grouped_data_diff['Difference_in_length'].index, grouped_data_diff['Vn [kV]'])
    # Add labels and title
    plt.xlabel('Voltage [kV]')
    plt.ylabel('Length difference [%]')
    #plt.title('% Difference in NTP-provided HVAC line lengths and \ncalculated line lengths [km] (where provided)')
    plt.tight_layout()
    # Show the plot
    plt.show()
 """


### Plot number of HVAC lines, with x axis as line length and categories of stacked bar are kV
# Create bins for the distance
bins = [0, 50, 100, 150, 999999999]  # Adjust the bins as needed
labels = ['0-50', '100-150', '150-200', '200+']
df = gdf_OHL
df['Distance_Bin'] = pd.cut(df['distance'], bins=bins, labels=labels, right=False)
# Group by 'Distance_Bin' and 'Vn [kV]', and count the occurrences
grouped = df.groupby(['Distance_Bin', 'Vn [kV]']).size().unstack(fill_value=0)
if scenario_name == 'S01':
    pd.to_pickle(grouped,basepath + 'HVAC_Scenario_HVAC_lines' + str(use_avg) + '_avg_num_bundles.pkl')
if scenario_name == 'S03':
    pd.to_pickle(grouped,basepath+'MTHVDC_Scenario_HVAC_lines' + str(use_avg) + '_avg_num_bundles.pkl')
# Plot the stacked bar plot 
""" # Get the colors used in the plot
ax = grouped.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
colors = ax.patches[0].get_facecolor()  # Get the color of the first patch to find the colormap
colormap = plt.cm.get_cmap('tab20')
# If there are multiple voltage levels, each will have a unique color
color_list = [colormap(i) for i in range(len(grouped.columns))] """
color_list_hvac = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
 (0.6823529411764706, 0.7803921568627451, 0.9098039215686274, 1.0),
 (1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
 (1.0, 0.7333333333333333, 0.47058823529411764, 1.0),
 (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0)]
color_list_hvdc =  [(0.6823529411764706, 0.7803921568627451, 0.9098039215686274, 1.0),
 (1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
 (1.0, 0.7333333333333333, 0.47058823529411764, 1.0),
 (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0)]
if scenario_name == 'S01':
    ax = grouped.plot(kind='bar', stacked=True, figsize=(10, 6), color=color_list_hvac)
if scenario_name == 'S03':
    ax = grouped.plot(kind='bar', stacked=True, figsize=(10, 6), color=color_list_hvdc)  ## this is the HVAC lines in the HVDC scenario
# Set the plot labels and title
ax.set_xlabel('HVAC Line length [miles]')
ax.set_ylabel('Number of HVAC Lines')
# Adjust the y-axis to show only integer values
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
#ax.set_title('HVAC: Number of Transmission Lines Grouped by Distance and Voltage')
plt.xticks(rotation=45)
plt.ylim([0,550])
plt.legend(title='Voltage [kV]', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



# At this point in the code, gdf_OHL is the HVAC dataframe, either in scenario HVAC or MTHVDC, depending on scenario_name parameter value.



### Calculate the HVDC line lengths
if scenario_name == 'S03':
    #### Calculate line lengths for HVDC
    # Filter to DC
    DC_df = linestring_df[(linestring_df['Type']=='HVDC') & (linestring_df['Add [Y/N]']=='Y')]
    DC_df['geometry'] = DC_df['geometry'].apply(wkt.loads)

    # Convert to right reference system
    gdf_DC = gpd.GeoDataFrame(DC_df, geometry='geometry')
    gdf_DC.set_crs('EPSG:4326', allow_override=True, inplace=True)

    # Reproject to a suitable CRS for distance calculations (e.g., EPSG 3857)
    #gdf_DC = gdf_DC.to_crs(epsg=3857)  # meters
    gdf_DC = gdf_DC.to_crs(epsg=32614) # meters that are more focused on center US

    # Calculate the distance of each LineString geometry
    gdf_DC['distance'] = gdf_DC.geometry.length/1e3*total_multiplier_for_length # convert m to km and apply length multipliers
    gdf_DC['distance'] = gdf_DC['distance']*0.62 # convert km to miles
    #print(gdf[['geometry', 'distance']])


    #### !! Checked if there are any ids with duplicated values
    # gdf_cleaned[gdf_cleaned.duplicated('id', keep=False) == True] ## This is an empty df, so there are no duplicated id values


    ### Plot lengths of HVDC lines, separated by Vn[kV]
    # Group by 'Vn [kV]' and calculate the mean or sum of 'distance'
    grouped_data = gdf_DC.groupby('Vn [kV]')['distance'].sum().reset_index()
    grouped_data = grouped_data.sort_values(by='Vn [kV]')

    # Total length of HVDC cable
    total_DC = grouped_data['distance'].sum()
    print('Total DC cable lengths ' + str(total_DC))
    print('DC cable lengths, grouped by voltage:', str(grouped_data))

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 18})
    # Create the bar plot with 'Vn [kV]' as x and 'distance' as y
    plt.bar(grouped_data['distance'].index, grouped_data['distance']/1e3, color='skyblue')
    # Set x-ticks to correspond to 'Vn [kV]' labels
    plt.xticks(grouped_data['distance'].index, grouped_data['Vn [kV]'])
    # Add labels and title
    plt.xlabel('Voltage [kV]')
    plt.ylabel('Total Distance HVDC [thosands of miles]')
    # Adjust the y-axis to show only integer values
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    #ax.set_yticks(range(0, int(grouped_data['distance'].max() / 1e3) + 1, 1))  # Adjust based on your data
    #plt.title('Total line lengths HVDC (Demand through 2035, NTP data)')
    plt.tight_layout()
    # Show the plot
    plt.show()



    ### Plot number of HVDC lines, with x axis as line length and categories of stacked bar are kV
    # Create bins for the distance
    bins = [0, 250, 500, 750, 999999999]  # Adjust the bins as needed
    labels = ['0-250', '250-500', '500-750', '750+']
    df = gdf_DC
    df['Distance_Bin'] = pd.cut(df['distance'], bins=bins, labels=labels, right=False)
    # Group by 'Distance_Bin' and 'Vn [kV]', and count the occurrences
    grouped = df.groupby(['Distance_Bin', 'Vn [kV]']).size().unstack(fill_value=0)
    print(f'this is mthvdc_dc: {grouped}')
    pd.to_pickle(grouped, basepath + 'MTHVDC_Scenario_HVDC_lines' + str(use_avg) + '_avg_num_bundles.pkl')
    # Plot the stacked bar plot
    ax = grouped.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
    # Set the plot labels and title
    ax.set_xlabel('Line length HVDC [miles]')
    ax.set_ylabel('Number of HVDC Lines')
    #ax.set_title('HVDC: Number of Transmission Lines Grouped by Distance and Voltage')
    plt.xticks(rotation=45)
    # Adjust the y-axis to show only integer values
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(title='Voltage [kV]', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()



    # Data for AC and DC
    labels = ['HVAC', 'HVDC']
    distances_thousands_of_miles = [total_AC/1e3, total_DC/1e3]
    # Create the bar plot
    plt.bar(labels, distances_thousands_of_miles, color=['blue', 'orange'])
    # Adding titles and labels
    plt.title('Distances for AC and DC')
    #plt.xlabel('Type')
    plt.ylabel('Distance (thousands of miles)')
    # Display the plot
    plt.show()


""" 
    ## Plot in miles
    distances_miles= [total_AC/1e3*0.621371, total_DC/1e3*0.621371]
    # Create the bar plot
    plt.bar(labels, distances_miles, color=['blue', 'orange'])
    # Adding titles and labels
    plt.title('Distances for AC and DC')
    #plt.xlabel('Type')
    plt.ylabel('Distance (thousands of miles)')
    # Display the plot
    plt.show() """



##### Find total kg of Aluminum and Steel ####### 

# gdf is the dataframe that has the line lengths calculated in the column 'distance'
# You need to create a new column that takes the multiplier from 'Conductor' column if it exists
#gdf['conductor_type_multiplier'] 


# If 'conductor' is NaN, then take the voltage value and map it to the conductor type in the following table
data = {
    'Voltage [kV]': [138, 230, 230, 345, 345, 500, 500, 525, 765],
    'Option': ['High Capacity', 'Standard', 'High Capacity', 'Standard', 'High Capacity', 'Standard', 'High Capacity', 'High Capacity', 'High Capacity'],
    'Conductor/Bundle': ['1 x Wolf', '2 x Grosbeak', '2 x Bluejay', '2 x Bluejay', '3 x Bluejay', '4 x Grosbeak', '6 x Bluejay', '10 x Bluejay', '6 x Bluejay']
}

# Create DataFrame
conductor_assumptions_df = pd.DataFrame(data)


# gdf_OHL is the filtered df for HVAC lines
# gdf_DC is the filtered df for HVDC lines

# Add a conductor value to those where it is missing, based on voltage assumptions
def add_conductor_value_where_missing(gdf):
    # Iterate over each row in gdf where 'Conductor' is NaN
    for idx, row in gdf[gdf['Conductor'].isna()].iterrows():
        # Get the matching row from conductor_assumptions_df where Voltage and Option = 'High Capacity'
        matching_row = conductor_assumptions_df[(conductor_assumptions_df['Voltage [kV]'] == row['Vn [kV]']) & 
                                                (conductor_assumptions_df['Option'] == 'High Capacity')]

        # If a match is found, update the 'Conductor' column in gdf
        if not matching_row.empty:
            gdf.at[idx, 'Conductor'] = matching_row['Conductor/Bundle'].values[0]
        if matching_row.empty:
            print(idx)
            print(row)
            print('empty')
    
    return gdf

# Add for both gdf_OHL and gdf_DC
gdf_OHL = add_conductor_value_where_missing(gdf_OHL)
if scenario_name == 'S03':
    gdf_DC = add_conductor_value_where_missing(gdf_DC)
    gdf_DC['Vn_HVDC'] = 525 # Add a column for the voltage level of HVDC. The existing voltage level in NTP dataset refers not to the DC conductor but the AC line that the DC conductor is connected to.


# Take the birdtype and multiply the digit from that by the weights from ACSR.
bird_dict = {
    'Bluejay': {
        'lb per 1000 ft Al': 1048,
        'lb per 1000 ft Stl': 205
    },
    'Grosbeak': {
        'lb per 1000 ft Al': 599,
        'lb per 1000 ft Stl': 275
    },
    'Bluejay_ACSS': {
        'lb per 1000 ft Al': 1048,
        'lb per 1000 ft Stl': 205
    }
}


# Calculate average Al and Stl across bird types
al_values = [bird['lb per 1000 ft Al'] for bird in bird_dict.values()]
avg_al_weight_per_1000ft = sum(al_values) / len(al_values)
stl_values = [bird['lb per 1000 ft Stl'] for bird in bird_dict.values()]
avg_stl_weight_per_1000ft = sum(stl_values)/ len(stl_values)


# This is per conductor! not yet including the 2x for DC bipole or 3 phase AC. That is taken into account in plot_al_and_stl...
def get_al_and_stl_weights(conductor_str, bird_dict, use_avg):
    if use_avg==False:
        # This means to use individual weight densities by cable type
        # Check if the conductor string contains 'Bluejay' or 'Grosbeak'
        for bird_type in bird_dict:
            if bird_type in conductor_str:
                # Extract the quantity from the string, assuming format is like "4 x Grosbeak"
                quantity = int(conductor_str.split(' x ')[0])  
                # Get the weights per 1000 ft 
                al_weight_per_1000ft = bird_dict[bird_type]['lb per 1000 ft Al']
                stl_weight_per_1000ft = bird_dict[bird_type]['lb per 1000 ft Stl']
                # Calculate total aluminum weight (convert to kg by dividing by 2.20462)
                total_al_weight = quantity * al_weight_per_1000ft / 2.20462  # Converting lb to kg
                total_stl_weight = quantity * stl_weight_per_1000ft / 2.20462  # Converting lb to kg
                return total_al_weight, total_stl_weight
    if use_avg==True:
        # This means to use the average cable weight density, across all types.
        # Despite the assumptions for NTP having multiple cable types (e.g. Grosbeak, Bluejay, ...) the assumptions when applied result in only 'Bluejay' type conductors, with varying numbers of conductors per bundle (2,3,4, and 6).  The average number of lines in a bundle (unweighted) is (2+3+4+6)/4 = 3.75. Rounding up is 4 cables per bundle.  We can use this as a sensitivity if we assume all cable bundles are 4xBluejay. 
        al_weight_per_1000ft = 1048
        stl_weight_per_1000ft = 205
        total_al_weight = 4 *al_weight_per_1000ft / 2.20462 # See above comment, 4 is the average number of cables per bundle.
        total_stl_weight = 4 *stl_weight_per_1000ft / 2.20462
        return total_al_weight, total_stl_weight


    return 0  # Return 0 if no matching bird type is found


def plot_al_and_stl_for_each_kv_and_type(cable_type, gdf, scenario_name, use_avg):
    # Calculate the weight for aluminum and steel for each row. 5280/1000 is converting miles to 1000 ft bc weight is in kg/1000ft
    # 3 * for 3 phase HVAC. The 2* for DC bipole is taken into account inthe if statment below.
    gdf['Total Al [kg]'] = gdf.apply(
        lambda row: 3* get_al_and_stl_weights(row['Conductor'], bird_dict, use_avg)[0] * row['distance'] * 5280 / 1000, 
        axis=1
    )

    gdf['Total Stl [kg]'] = gdf.apply(
        lambda row: 3 * get_al_and_stl_weights(row['Conductor'], bird_dict, use_avg)[1] * row['distance'] * 5280 / 1000,
        axis=1
    )

    # This overwrites the gdf Al and Stl if DC . 
    if gdf.iloc[0]['Remarks'] == 'New HVDC bipole':
        # We assume 5 x Bluejay for each HVDC pole. so 2 x 5 x Bluejay for each HVDC bipole. 
        # 5280/1000 is converting miles to 1000s of ft. if we assume Bluejay as conductors (1092 A thermal rating, derated for conservative design to 80%) and 525 kV for the HVDC design, this would mean # 2000 MW monopole: I = 2000/525 = 3.8 kA i.e. 5 Bluejay conductor bundle. # 4000 MW bipole: I = 4000/(525 –(-525)) = 3.8 kA i.e. 5 Bluejay conductor bundle per pole.

        total_aluminum_weight_in_kg_per_1000ft_DC = 2* 5* bird_dict['Bluejay']['lb per 1000 ft Al']/ 2.20462  # Converting lb to kg
        total_steel_weight_in_kg_per_1000ft_DC = 2* 5* bird_dict['Bluejay']['lb per 1000 ft Stl']/ 2.20462  # Converting lb to kg
        gdf['Total Al [kg]'] = gdf.apply(
            lambda row: total_aluminum_weight_in_kg_per_1000ft_DC * row['distance'] * 5280 / 1000,
            axis=1
        )
        gdf['Total Stl [kg]'] = gdf.apply(
            lambda row: total_steel_weight_in_kg_per_1000ft_DC * row['distance'] * 5280 / 1000,
            axis=1
        )
        

    # Group by 'kV' and sum the results
    result = gdf.groupby('Vn [kV]')[['Total Al [kg]', 'Total Stl [kg]']].sum().reset_index()
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Creating a stacked bar chart
    result.set_index('Vn [kV]')[['Total Al [kg]', 'Total Stl [kg]']].plot(kind='bar', stacked=True, ax=ax)
    result.to_pickle(basepath + f'{cable_type}_{scenario_name}_total_weights_by_voltage_avg_{str(use_avg)}.pkl')

    # Adding labels and title
    if scenario_name == 'S03':
        plt.ylim(0,4e8)
        if cable_type == 'HVAC':
            scenario_name_for_plot = 'MTHVDC Scenario'
        elif cable_type == 'HVDC':
            scenario_name_for_plot = 'MTHVDC Scenario'
        
    if scenario_name == 'S01':
        scenario_name_for_plot = 'HVAC Scenario'

    ax.set_xlabel('Voltage Level [kV]')
    ax.set_ylabel('Total Weight [kg]')
    ax.set_title(cable_type + ' in ' + scenario_name_for_plot + ' using average cable weight = ' + str(use_avg))

    # Show plot
    plt.tight_layout()
    plt.show()
    print(cable_type)
    print(result)


    # Print the sums of the Al and Stl in kg for each scenario

    # Calculate the sums of 'Total Al [kg]' and 'Total Stl [kg]'
    total_al_sum = gdf['Total Al [kg]'].sum()
    total_stl_sum = gdf['Total Stl [kg]'].sum()
    print('Total Al [kg] in ', scenario_name, ' for ', cable_type, ': ', str(total_al_sum))
    print('Total Stl [kg] in ', scenario_name, ' for ', cable_type, ': ', str(total_stl_sum))

    # Prepare data for plotting
    categories = ['Total Al [kg]', 'Total Stl [kg]']
    values = [total_al_sum, total_stl_sum]

    # Create a bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(categories, values, color=['blue', 'green'])
    plt.xlabel('Wire Component')
    plt.ylabel('Total Weight (kg)')
    if scenario_name == 'S03':
        plt.ylim(0,6e8)
    plt.title(scenario_name_for_plot + ' ' + cable_type + ': \nTotal Weight of Aluminum and Steel Wires')
    plt.show()

    ## Plot separately by kV


    return


efficiency_factor = 0.9

if scenario_name == 'S01':
    plot_al_and_stl_for_each_kv_and_type('HVAC', gdf_OHL, scenario_name, use_avg)
    # Calculate the conductor length in TW-miles
    gdf_OHL['TW-miles'] = gdf_OHL['distance']*gdf_OHL['Rate1[MVA]']*efficiency_factor/1e6  # conversion factors are from MW to TW 
    # gdf_OHL['Conductor'].unique() = ['4 x Bluejay', '3 x Bluejay', '2 x Bluejay', '6 x Bluejay']
if scenario_name == 'S03':
    plot_al_and_stl_for_each_kv_and_type('HVAC', gdf_OHL, scenario_name, use_avg)
    gdf_OHL['TW-miles'] = gdf_OHL['distance']*gdf_OHL['Rate1[MVA]']*efficiency_factor/1e6
    # gdf_OHL['Conductor'].unique() = ['4 x Bluejay', '6 x Bluejay', '3 x Bluejay', '2 x Bluejay']

    plot_al_and_stl_for_each_kv_and_type('HVDC', gdf_DC, scenario_name, use_avg)
    gdf_DC['TW-miles'] = gdf_DC['distance']*gdf_DC['Rate1[MVA]']*efficiency_factor/1e6  #
    # gdf_OHL['Conductor'].unique() = ['6 x Bluejay', '3 x Bluejay', '2 x Bluejay']


# Despite the assumptions for NTP having multiple cable types (e.g. Grosbeak, Bluejay, ...) the assumptions when applied result in only 'Bluejay' type conductors, with varying numbers of conductors per bundle (2,3,4, and 6).  The average number of lines in a bundle (unweighted) is (2+3+4+6)/4 = 3.75. Rounding up is 4 cables per bundle.  We can use this as a sensitivity if we assume all cable bundles are 4xBluejay. 



