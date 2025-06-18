
# Adds locations to HVAC transformers from NTP data.

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
scenario_name = 'S01'   # This script for adding geometry is only needed for S01 because S03 file contains locatoins
 


### Read data
if scenario_name == 'S01':
    filenames = ['R02_S01_Transmission_Expansion_EI-1', 'R02_S01_Transmission_Expansion_ERCOT-1', 'R02_S01_Transmission_Expansion_WECC-1']
    #filenames = ['R02_S01_Transmission_Expansion_WECC-1']
    filename_str = filenames[0].split('_')[-1].split('-')[0]  # return, e.g. ERCOT for that filename
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
# filter the node locations by region
if scenario_name == 'S01':
    node_locations_df = node_locations_df[node_locations_df['area'].str.contains(filename_str, na=False)]




### Calculate lengths HVAC
# Filter to HVAC where column 'Type' = OHL
OHL_df = linestring_df[(linestring_df['Type']=='2TF') & (linestring_df['Add [Y/N]']=='Y')]




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



### If HVAC scenario 1, match the node id 'ToBus' (where the transformer is located) of the scenario run file (no location info) to the node_id in CONUS_updates (location info)

number_last_digits_to_match = 6

if scenario_name == 'S01':
    # Extract last 6 digits of `node_id` in `node_locations_df`
    node_locations_df['last_digits_node_id'] = node_locations_df['node_id'].astype(str).str[-number_last_digits_to_match:]

    ## Merge node_locations_df with OHL_df based on the last 6 digits for ToBus
    # ToBus 
    OHL_df = OHL_df.merge(
        node_locations_df[['last_digits_node_id', 'geometry']], 
        left_on=OHL_df['ToBus'].astype(str).str[-number_last_digits_to_match:], 
        right_on='last_digits_node_id', 
        how='left'
    ).drop(columns=['last_digits_node_id'])

    blank_geometry_rows = OHL_df[OHL_df['geometry'].isna() | (OHL_df['geometry'] == '')]

    while not blank_geometry_rows.empty:

        # print('no location matched becuase not enough digits for the row id to match the location file id')

        # Increase the number of digits that must match for identifiers to reduce multiple matches and run again.
        number_last_digits_to_match = number_last_digits_to_match - 1
        # Extract locations that match
        node_locations_df['last_digits_node_id'] = node_locations_df['node_id'].astype(str).str[-number_last_digits_to_match:]
        ## Merge node_locations_df with OHL_df based on the last 6 digits for ToBus
        # Clear OHL_df 
        OHL_df = linestring_df[(linestring_df['Type']=='2TF') & (linestring_df['Add [Y/N]']=='Y')]
        OHL_df = OHL_df.merge(
            node_locations_df[['last_digits_node_id', 'geometry']], 
            left_on=OHL_df['ToBus'].astype(str).str[-number_last_digits_to_match:], 
            right_on='last_digits_node_id', 
            how='left'
        ).drop(columns=['last_digits_node_id'])

        # Update to check if blank geometry rows have now been matched
        blank_geometry_rows = OHL_df[OHL_df['geometry'].isna() | (OHL_df['geometry'] == '')]

            

    # Apply the cleaning function to the 'FromBusLocation' and 'ToBusLocation' columns
    OHL_df['geometry'] = OHL_df['geometry'].apply(lambda x: clean_wkt_string(x) if isinstance(x, str) else x)

    # Now convert the cleaned WKT strings to Point objects
    OHL_df['geometry'] = OHL_df['geometry'].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)

    # Reverse the coords so lat and lon are in the correct positions
    OHL_df['geometry'] = OHL_df['geometry'].apply(correct_coordinates)



OHL_df['geometry'] = OHL_df['geometry'].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)
# Create geodataframe
gdf_OHL = gpd.GeoDataFrame(OHL_df, geometry='geometry')
# Set the CRS to WGS84 (lat/lon)
gdf_OHL.set_crs('EPSG:4326', allow_override=True, inplace=True)


# save the dataframe that has the locations added in the column called 'ToBusLocation'
gdf_OHL.to_csv(filenames[0] + '_HVAC_location.csv', index=False)




'''

# 
## After running commented code above, combine all the dataframes into one
filenames = ['R02_S01_Transmission_Expansion_EI-1', 'R02_S01_Transmission_Expansion_ERCOT-1', 'R02_S01_Transmission_Expansion_WECC-1']
df_list = [pd.read_csv(f + '_HVAC_location.csv') for f in filenames]
combined_df = pd.concat(df_list, ignore_index=True)
combined_df.to_pickle('outputs/combined_HVAC_location.pkl')

'''