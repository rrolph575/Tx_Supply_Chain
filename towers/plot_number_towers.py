import pandas as pd
import matplotlib.pyplot as plt
import os

# Input dataframes calculated from number_and_materials_towers.py. 


projpath = os.path.dirname(os.getcwd())
datapath  = os.path.join(projpath,'towers','data')
figpath = os.path.join(projpath,'plots')



def load_structure_data(scenario, conductor, datapath):
    """Load structure data and create a DataFrame."""
    file_name = f'number_structures_{scenario}_scenario_{conductor}_lines.csv'
    data = pd.read_csv(os.path.join(datapath, file_name))
    df_number_struct = pd.DataFrame({
        'Type': list(data.keys())[1:],
        f'{scenario} Scenario {conductor} Conductors': data.sum()[1:]
    }).reset_index(drop=True)

    df_struct_by_voltage = pd.DataFrame({
    'Voltage': data['Unnamed: 0'],
    f'{scenario} Scenario {conductor} Conductors': data.iloc[:,1:].sum(axis=1)
    })

    """Load amt steel data and create a DataFrame."""
    file_name = f'steel_weights_{scenario}_scenario_{conductor}_lines.csv'
    data = pd.read_csv(os.path.join(datapath, file_name))
    df_amt_steel = pd.DataFrame({
        'Type': list(data.keys())[1:],
        f'Millions_US_tons_Stl_{scenario}_lines_{conductor}_Scenario': data.sum()[1:]
    }).reset_index(drop=True)

    df_stl_by_voltage = pd.DataFrame({
    'Voltage': data['Unnamed: 0'],
    f'{scenario} Scenario {conductor} Conductors': data.iloc[:,1:].sum(axis=1)
    })

    return df_number_struct, df_struct_by_voltage, df_amt_steel, df_stl_by_voltage

# Define combinations of scenarios and conductors
combinations = [
    ('AC', 'AC'),
    ('DC', 'AC'),
    ('DC', 'DC')
]


# Access DataFrames
# Initialize dictionaries to store DataFrames
df_structures_by_type = {}
df_struct_by_voltage = {}
df_amt_steel = {}
df_stl_by_voltage = {}

# Load data for each combination
for scenario, conductor in combinations:
    key = f'{scenario}_{conductor}'
    df_structures_by_type[key], df_struct_by_voltage[key], df_amt_steel[key], df_stl_by_voltage[key] = load_structure_data(scenario, conductor, datapath)

# Merge all scenarios on the 'Type' column
df_total_towers = df_structures_by_type['AC_AC'].merge(
    df_structures_by_type['DC_AC'], on='Type').merge(
    df_structures_by_type['DC_DC'], on='Type'
)




# %% Plot
# Reshape the DataFrame for plotting
df_total_towers_melted = df_total_towers.melt(
    id_vars='Type', 
    var_name='Scenario', 
    value_name='Count'
)

# Create a stacked bar plot
df_pivot = df_total_towers_melted.pivot(index='Type', columns='Scenario', values='Count')
df_pivot.plot(kind='bar', stacked=True, figsize=(10, 6))

# Add labels and title
#plt.xlabel('Tower Type', fontsize=14)
plt.xlabel('')
plt.ylabel('Count', fontsize=16)
plt.title('Total Towers by Scenario', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=16)
plt.tick_params(axis='y', labelsize=16)

# Adjust legend
plt.legend(title='Scenario',title_fontsize =16, bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 16})

# Show the plot
plt.tight_layout()
plt.show()



#%% Plot total amount of steel 
# Merge all scenarios on the 'Type' column
df_total_stl = df_amt_steel['AC_AC'].merge(
    df_amt_steel['DC_AC'], on='Type').merge(
    df_amt_steel['DC_DC'], on='Type'
)


# Reshape the DataFrame for plotting
df_total_stl_melted = df_total_stl.melt(
    id_vars='Type', 
    var_name='Scenario', 
    value_name='Count'
)

# Create a stacked bar plot
df_pivot = df_total_stl_melted.pivot(index='Type', columns='Scenario', values='Count')
df_pivot.plot(kind='bar', stacked=True, figsize=(10, 6))

# Add labels and title
#plt.xlabel('Tower Type', fontsize=14)
plt.xlabel('')
plt.ylabel('Count', fontsize=16)
plt.title('Total Steel by Scenario', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=16)
plt.tick_params(axis='y', labelsize=16)

# Adjust legend
plt.legend(title='Scenario',title_fontsize =16, bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 16})

# Show the plot
plt.tight_layout()
plt.show()



## Plot number of structures by voltage
df_struct_by_voltage_combined = df_struct_by_voltage['AC_AC'] \
    .merge(df_struct_by_voltage['DC_AC'], on='Voltage', how='outer') \
    .merge(df_struct_by_voltage['DC_DC'], on='Voltage', how='outer')


# Melt into long format
df_melted = df_struct_by_voltage_combined.melt(id_vars='Voltage', var_name='Scenario', value_name='Count')

# Replace NaN with 0 for plotting
df_melted['Count'] = df_melted['Count'].fillna(0)
# Wrap long x-axis labels by inserting '\n'
def replace_second_whitespace(label):
    parts = label.split(' ', 2)
    if len(parts) == 3:
        return f'{parts[0]} {parts[1]}\n{parts[2]}'
    return label  # if fewer than 2 underscores, leave unchanged

df_melted['Scenario'] = df_melted['Scenario'].apply(replace_second_whitespace)   

# Pivot for stacked bar
df_pivot = df_melted.pivot(index='Scenario', columns='Voltage', values='Count')

# Plot
plt.rcParams.update({'font.size': 16}) 
df_pivot.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
plt.ylabel('Total number structures')
plt.title('Total number structures by Scenario and Voltage')
plt.xlabel(' ')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Voltage (kV)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

