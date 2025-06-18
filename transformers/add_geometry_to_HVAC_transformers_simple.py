
# Adds locations to HVAC tranthedfsformers from NTP data.

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
from matplotlib.ticker import MaxNLocator
from shapely.geometry import Point, LineString
import re


pd.set_option('display.max_columns', None)  # Show all columns

#%% Define paths
basepath = 'C:/Users/rrolph/OneDrive - NREL/Projects/FY25/Transmission_Supply_Chain/'
datapath = basepath + 'Data/R02_Transmission_Expansion/'
#%% Load node locations
node_locations_file = datapath + 'CONUS_nodes_UPDATE.csv'
node_locations_df_all = pd.read_csv(node_locations_file) 

# Define scenario name
scenario_name = 'S01'   # This script for adding geometry is only needed for S01 because S03 file contains locatoins
 
# Define filenaems
filenames = ['R02_S01_Transmission_Expansion_EI-1', 'R02_S01_Transmission_Expansion_ERCOT-1', 'R02_S01_Transmission_Expansion_WECC-1']

merged_dfs = []
for file in filenames:
    filename_str = file.split('_')[-1].split('-')[0]  # return, e.g. ERCOT for that filename
    print(filename_str)

    df = pd.read_csv(datapath + file + '.csv')
    #print(df.iloc[0])

    # Filter for the HVAC transformers
    df_all_HVAC_transformers = df[(df['Type'] == '2TF') & (df['Add [Y/N]'] == 'Y')]

    # Have to group or filter more the df_all_HVAC_transformers becuase the ToBus is not enough digits to uniquely match a node_id

    # filter the node locations by current filename
    node_locations_df = node_locations_df_all[(node_locations_df_all['area'].str.contains(filename_str, na=False)) & (node_locations_df_all['country'] == 'USA')]

    #  Match the node id 'ToBus' (where the transformer is located) of the scenario run file (no location info) to the node_id in CONUS_updates (location info)
    number_last_digits_to_match = 6  # have to filter more the df_all_HVAC_transformers becuase the ToBus is not enough digits to uniquely match a node_id
    # Extract the last 6 digits of `node_id` in `node_locations_df`
    node_locations_df['last_digits_node_id'] = node_locations_df['node_id'].astype(str).str[-number_last_digits_to_match:]
    #print(node_locations_df.iloc[0])


    # Extract the last 6 digits of `ToBus` in `df_all_HVAC_transformers`
    df_all_HVAC_transformers['last_digits_ToBus'] = df_all_HVAC_transformers['ToBus'].astype(str).str[-number_last_digits_to_match:]

    # Filter df_all_HVAC_transformers to only those rows where 'last_digits_ToBus' is contained in any 'last_digits_node_id'
    #filtered_df = df_all_HVAC_transformers[
    #    df_all_HVAC_transformers['last_digits_ToBus'].apply(
    #        lambda tobus: node_locations_df['last_digits_node_id'].str.contains(tobus).any()
    #    )
    #]
    filtered_df = df_all_HVAC_transformers[
    df_all_HVAC_transformers['last_digits_ToBus'].isin(node_locations_df['last_digits_node_id'])]

    merged_df = pd.merge(
        node_locations_df, filtered_df,
        how='inner',
        left_on=node_locations_df['last_digits_node_id'].apply(lambda x: next((tobus for tobus in filtered_df['last_digits_ToBus'] if tobus in x), None)),
        right_on='last_digits_ToBus'
    )

    # save the dataframe that has the locations added in the column called 'ToBusLocation'
    merged_df.to_csv(file + '_HVAC_location.csv', index=False)
    #print(merged_df.head())
    print(merged_df.shape)
    merged_dfs.append(merged_df.copy()) 

combined_df = pd.concat(merged_dfs, ignore_index=True).drop_duplicates()
combined_df.to_pickle('outputs/combined_HVAC_location_simple.pkl')
