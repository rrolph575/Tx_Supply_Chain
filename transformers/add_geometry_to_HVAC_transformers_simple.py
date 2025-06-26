
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
node_locations_CONUS_df_all = pd.read_csv(node_locations_file) 
voltages_to_keep = [138, 230, 345, 500, 765]
node_locations_CONUS_df_all = node_locations_CONUS_df_all[node_locations_CONUS_df_all['voltage'].isin(voltages_to_keep)].copy()

# Define scenario name
scenario_name = 'S01'   # This script for adding geometry is only needed for S01 because S03 file contains locatoins
 
# Define filenaems
filenames = ['R02_S01_Transmission_Expansion_EI-1', 'R02_S01_Transmission_Expansion_ERCOT-1', 'R02_S01_Transmission_Expansion_WECC-1']

merged_dfs = []  
for file in filenames:    # [1:2]
    filename_str = file.split('_')[-1].split('-')[0]  # return, e.g. ERCOT for that filename
    print(filename_str)

    df = pd.read_csv(datapath + file + '.csv')
    #print(df.iloc[0])

    # Filter for the HVAC transformers in the region 
    df_all_HVAC_transformers = df[(df['Type'] == '2TF') & (df['Add [Y/N]'] == 'Y')].copy()
    print(f'Size of all HVAC transformers in input file for {filename_str}: {df_all_HVAC_transformers.shape}')


    df_all_HVAC_transformers['Vn [kV]'] = df_all_HVAC_transformers['Vn [kV]'].astype(str).str.replace('/', '', regex=False)  # remove slashes from voltage values


    # filter the CONUS node locations by current filename
    node_locations_CONUS_df = node_locations_CONUS_df_all[(node_locations_CONUS_df_all['area'].str.contains(filename_str, na=False)) & (node_locations_CONUS_df_all['country'] == 'USA')].copy()


    # Match the node id 'ToBus' (where the transformer is located) of the scenario run file (no location info) to the node_id in CONUS_updates (location info)
    number_last_digits_to_match = 6  # have to filter more the df_all_HVAC_transformers becuase the ToBus is not enough digits to uniquely match a node_id
    # Extract the last digits of `node_id` in `node_locations_CONUS_df`
    node_locations_CONUS_df['last_digits_node_id'] = node_locations_CONUS_df['node_id'].astype(str).str[-number_last_digits_to_match:]
    #print(node_locations_CONUS_df.iloc[0])


    # Extract the last digits of `ToBus` in `df_all_HVAC_transformers`
    df_all_HVAC_transformers['last_digits_ToBus'] = df_all_HVAC_transformers['ToBus'].astype(str).str[-number_last_digits_to_match:]

    # Filter df_all_HVAC_transformers to only those rows where 'last_digits_ToBus' is contained in any 'last_digits_node_id'
    filtered_df = df_all_HVAC_transformers[
    df_all_HVAC_transformers['last_digits_ToBus'].isin(node_locations_CONUS_df['last_digits_node_id'])]


    # filtered_df should be the same size as df_all_HVAC_transformers['last_digits_ToBus'] 
    # this would mean that all ToBus locations have been matched

    # if filtered_df.shape is larger, then that means more than one entry of node_id in CONUS matched the file df_all_HVAC*. 
    # so you have to then take 
    while df_all_HVAC_transformers['last_digits_ToBus'].shape[0] != filtered_df.shape[0]:
        print('df_all_HVAC_transformers last digits ToBus:')
        print(df_all_HVAC_transformers['last_digits_ToBus'].shape[0])
        print('filtered_df from CONUS file shape:')
        print(filtered_df.shape[0]) ### !!! why is this zero for ERCOT maybe 6 is not good number of ints to match? 
        number_last_digits_to_match = number_last_digits_to_match - 1 # reduce number ints to match to get more results
        # Extract the last digits of `node_id` in `node_locations_CONUS_df`
        node_locations_CONUS_df['last_digits_node_id'] = node_locations_CONUS_df['node_id'].astype(str).str[-number_last_digits_to_match:].copy()
        #print(node_locations_CONUS_df.iloc[0])
        # Extract the last int digits of `ToBus` in `df_all_HVAC_transformers`
        df_all_HVAC_transformers['last_digits_ToBus'] = df_all_HVAC_transformers['ToBus'].astype(str).str[-number_last_digits_to_match:]
        # Filter df_all_HVAC_transformers to only those rows where 'last_digits_ToBus' is contained in any 'last_digits_node_id'
        filtered_df = df_all_HVAC_transformers[
        df_all_HVAC_transformers['last_digits_ToBus'].isin(node_locations_CONUS_df['last_digits_node_id'])]


    merged_df = pd.merge(
        node_locations_CONUS_df, filtered_df,
        how='inner',
        left_on=node_locations_CONUS_df['last_digits_node_id'].apply(lambda x: next((tobus for tobus in filtered_df['last_digits_ToBus'] if tobus in x), None)),
        right_on='last_digits_ToBus'
    )



    # save the dataframe that has the locations added in the column called 'ToBusLocation'
    merged_df.to_csv(file + '_HVAC_location.csv', index=False)
    #print(merged_df.head())
    print(f'This should be the same as the input file {merged_df.shape[0]}')
    merged_dfs.append(merged_df.copy()) 

combined_df = pd.concat(merged_dfs, ignore_index=True).drop_duplicates()



combined_df.to_pickle('outputs/combined_HVAC_location_simple.pkl')



# %%
