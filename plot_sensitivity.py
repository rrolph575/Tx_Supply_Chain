import pandas as pd
import matplotlib.pyplot as plt


# Plot the difference in weights of Al and Stl when an average number of lines are used per bundle vs. the NTP assumptions.



# Load files
datapath = "C:/Users/rrolph/OneDrive - NREL/Projects/FY25/Transmission_Supply_Chain/"


# Dictionary to store DataFrames
dataframes = {}

for scenario in ['S01', 'S03']:
    if scenario == 'S01':
        cable_types = ['HVAC'] 
    else:
        cable_types = ['HVAC', 'HVDC']
    for cable_type in cable_types:
        for use_avg_num_cable_bundles in ['True', 'False']:
            # Construct the filename dynamically
            filename = f"{cable_type}_{scenario}_total_weights_by_voltage_avg_{use_avg_num_cable_bundles}.pkl"
            
            # Read the .pkl file into a DataFrame
            df = pd.read_pickle(datapath + filename)
            
            # Store the DataFrame in the dictionary with a unique key
            dataframes[f"{cable_type}_{scenario}_{use_avg_num_cable_bundles}"] = df

# Unpack dfames into individual variables
for key, df in dataframes.items():
    globals()[f"df_{key}"] = df
# e.g. df_HVAC_S01_True, ...


# Calculate the differences in weights using average number of cables in bundle vs not
# Where False means using the NTP assumptions and no average was taken. 
# Subtract DataFrames while leaving the 'Vn[kV]' column unchanged
# List of DataFrame pairs and their resulting variable names
dataframe_pairs = [
    ('df_HVAC_S01_False', 'df_HVAC_S01_True', 'df_HVAC_S01_False_minus_True'),
    ('df_HVAC_S03_False', 'df_HVAC_S03_True', 'df_HVAC_S03_False_minus_True'),
    ('df_HVDC_S03_False', 'df_HVDC_S03_True', 'df_HVDC_S03_False_minus_True')
]

# Perform subtraction while leaving the 'Vn[kV]' column unchanged
results = {}
for false_df, true_df, result_name in dataframe_pairs:
    results[result_name] = eval(false_df).copy()
    results[result_name].loc[:, eval(false_df).columns != 'Vn[kV]'] = (
        eval(false_df).loc[:, eval(false_df).columns != 'Vn[kV]'] -
        eval(true_df).loc[:, eval(true_df).columns != 'Vn[kV]']
    )
    results[result_name]['Vn [kV]'] = eval(false_df)['Vn [kV]']

    # results['df_HVAC_S01_False_minus_True']


for key in results.keys():
    if 'S03' in key:
        scenario = 'MTHVDC Scenario'
    else:
        scenario = 'HVAC Scenario'
    line_type = 'HVDC lines' if 'HVDC' in key else 'HVAC lines'

    fig, ax = plt.subplots()
    plt.title(f'{scenario}, {line_type}. Difference of weights when using NTP assumptions \n in conductors per bundle minus average number of conductors per bundle')
    # Convert from kg to metric tonnes
    df = results[key].set_index('Vn [kV]')[['Total Al [kg]', 'Total Stl [kg]']]/1e6
    df.plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel('Weight [thousands of metric tonnes]')
    ax.set_xlabel('Voltage [kV]')
    ax.legend(['Aluminum', 'Steel'])
    ax.set_ylim([-90,50])

    plt.tight_layout()
    plt.savefig(f'{key}_plot.png')  # Save each plot with a unique filename
    plt.show()


## Total weight difference
# total_al_sum = gdf['Total Al [kg]'].sum()
# total_stl_sum = gdf['Total Stl [kg]'].sum()