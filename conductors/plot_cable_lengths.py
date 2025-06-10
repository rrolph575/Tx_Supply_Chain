import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors


datapath = "C:/Users/rrolph/OneDrive - NREL/Projects/FY25/Transmission_Supply_Chain/"

# read pkl files saved from find_cable_lengths*py
use_avg = False # True if use avg number of cables in bundle.
print('If plots are using avg number of bundles of cable per line: ' + str(use_avg))
hvac_data = pd.read_pickle(datapath+'outputs/HVAC_Scenario_HVAC_lines' + str(use_avg) + '_avg_num_bundles.pkl')
mthvdc_dc_data = pd.read_pickle(datapath+'outputs/MTHVDC_Scenario_HVDC_lines' + str(use_avg) + '_avg_num_bundles.pkl')
mthvdc_ac_data = pd.read_pickle(datapath+'outputs/MTHVDC_Scenario_HVAC_lines' + str(use_avg) + '_avg_num_bundles.pkl')

dataframes = [hvac_data, mthvdc_dc_data, mthvdc_ac_data]

plt.rcParams.update({'font.size': 16})


# List of DataFrames to process
dataframes = [hvac_data, mthvdc_dc_data, mthvdc_ac_data]
# List to store the transformed dictionaries
output_data = []

# Iterate through each DataFrame in the list
for df in dataframes:
    # Rename columns
    df.rename(columns={230: '230 kV', 345: '345 kV', 500: '500 kV', 765: '765 kV'}, inplace=True)
    
    # Reset index and convert to dictionary
    output_data.append(df.reset_index().to_dict(orient='list'))

# Separate dictionaries for each DataFrame
hvac_data = output_data[0]
mthvdc_dc_data = output_data[1]
mthvdc_ac_data = output_data[2]

# Convert to DataFrame
df_hvac = pd.DataFrame(hvac_data)
df_mthvdc_ac = pd.DataFrame(mthvdc_ac_data)
df_mthvdc_dc = pd.DataFrame(mthvdc_dc_data)

# Rename datasets for easier distinction later
df_hvac['Dataset'] = 'HVAC'
df_mthvdc_ac['Dataset'] = 'MTHVDC_AC'
df_mthvdc_dc['Dataset'] = 'MTHVDC_DC'

# Combine the AC datasets (HVAC + MTHVDC_AC)
df_ac = pd.concat([df_hvac, df_mthvdc_ac], ignore_index=True)

# Combine the DC dataset (MTHVDC_DC)
df_dc = df_mthvdc_dc
# Sum the HVAC voltages DC is connected to, to the actual NTP assumed voltage for DC lines
df_dc['525 kV'] = df_dc.select_dtypes(include='number').sum(axis=1)
# Drop the other columns
df_dc = df_dc[['Distance_Bin', '525 kV', 'Dataset']]
# Replace the '0-250' label with '100-250' in the 'Distance_Bin' column
df_dc['Distance_Bin'] = df_dc['Distance_Bin'].replace('0-250', '100-250')
#df_dc.loc["100-250"] = df_dc.loc["0-250"]
#df_dc.drop("0-250", inplace=True)

# Set the 'Distance_Bin' as the index for better plotting control
df_ac.set_index('Distance_Bin', inplace=True)
df_dc.set_index('Distance_Bin', inplace=True)


""" # Plotting AC stacked bar plot with separate bars for HVAC and MTHVDC_AC
fig, ax = plt.subplots(figsize=(12, 6))
# Number of distance bins
n_bins = len(df_ac.index)
# Define positions for bars (split each bin into two bars)
bar_width = 0.35  # width of each bar
indices = np.arange(n_bins)  # positions of the bars on the x-axis
# Generate a blue gradient for the voltage levels
blue_colors = [cm.Blues(i) for i in [0.8, 0.6, 0.4, 0.2]]  # Darkest to lightest blue
# Plot HVAC dataset
df_hvac.set_index(['Dataset', df_hvac.index]).unstack(level=0).plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], hatch = '//', width=bar_width, position=0)
# Plot MTHVDC_AC dataset (shifted by -bar_width/2)
df_mthvdc_ac.set_index(['Dataset', df_mthvdc_ac.index]).unstack(level=0).plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], width=bar_width, position=1)
# Set x-axis labels manually, ensuring no repeated labels
ax.set_xticks(indices)  # set tick positions at the center of each group
ax.set_xticklabels(df_ac.index)  # set the labels to the distance bins
#ax.set_title('Stacked Bar Plot of HVAC and MTHVDC_AC by Distance Bin and Voltage Level')
ax.set_xlabel('Distance (miles)')
ax.set_ylabel('Number of Lines')
# Remove the default legend entries
ax.get_legend().remove()
# Create custom legend entries for the voltage levels (colors) and hatch pattern for HVAC
handles = []
# Create color patches for each voltage level (230 kV, 345 kV, 500 kV, 765 kV) with no hatching
color_legend_230 = mpatches.Patch(color=blue_colors[0], label='230 kV')
color_legend_345 = mpatches.Patch(color=blue_colors[1], label='345 kV')
color_legend_500 = mpatches.Patch(color=blue_colors[2], label='525 kV')
color_legend_765 = mpatches.Patch(color=blue_colors[3], label='765 kV')
# Create a custom hatch pattern legend entry for HVAC
hvac_legend = mpatches.Patch(hatch=None, label='HVAC Scenario', edgecolor='black', facecolor='white')
# Create a custom colorless patch for MTHVDC_AC scenario
mthvdc_legend = mpatches.Patch(hatch=None, label='MTHVDC Scenario', edgecolor='black', facecolor='white')
# Combine the custom legend entries
handles.extend([color_legend_230, color_legend_345, color_legend_500, color_legend_765, hvac_legend, mthvdc_legend])
# Add the updated legend with only the custom entries
ax.legend(handles=handles, title="Voltage Level (kV)", bbox_to_anchor=(0.70, 1), loc='upper left')
# Get the current x-tick positions
ticks = ax.get_xticks()
# Take only the first 4 tick positions
new_ticks = ticks[:4]
# Get the first 4 labels
new_labels = [label.get_text() for label in ax.get_xticklabels()][:4]
# Set the new ticks and labels
ax.set_xticks(new_ticks)
ax.set_xticklabels(new_labels, rotation=45)
# Get the first and last tick positions
x_min = new_ticks[0] - 0.5
x_max = new_ticks[-1] + 0.5
# Set the x-axis limits to the first and last tick positions
ax.set_xlim(x_min, x_max)
plt.tight_layout()
plt.show() 




# Plotting DC stacked bar plot
fig, ax = plt.subplots(figsize=(10, 6))
# Sum the HVAC voltages DC is connected to, to the actual NTP assumed voltage for DC lines
df_dc['525 kV'] = df_dc.select_dtypes(include='number').sum(axis=1)
# Drop the other columns
df_dc = df_dc[['525 kV', 'Dataset']]
df_dc.plot(kind='bar', stacked=True, ax=ax, color=['blue'])
# Customizing the DC plot
#ax.set_title('Stacked Bar Plot of MTHVDC_DC by Distance Bin and Voltage Level')
ax.set_xlabel('Distance (miles)')
ax.set_ylabel('Number of Lines')
ax.set_xticklabels(df_dc.index, rotation=45)  # Ensure DC distance bins are labeled correctly
# Add the custom hatch legend to the DC plot
ax.legend(title="Voltage Level (kV)", bbox_to_anchor=(0.70, 1), loc='upper left')
#ax.get_legend().set_visible(False)
plt.tight_layout()
plt.show()
"""



# Plotting AC stacked bar plot with separate bars for HVAC and MTHVDC_AC
blue_colors = [cm.Blues(i) for i in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.2]]
#blue_colors = [cm.Blues(i) for i in [0.9, 0.7, 0.6, 0.4, 0.2]]  # Darkest to lightest blue
fig, ax = plt.subplots(figsize=(12, 6))
# Number of distance bins
n_bins = 8 #len(df_ac.index)
# Set x-axis labels manually, ensuring no repeated labels
bar_width = 0.35  # width of each bar
#group_spacing = 2  # Increase this value for larger spacing
#indices = np.arange(n_bins) * (1 + group_spacing)  # Add spacing between groups
# Add HVAC and MTHVDC labels under each group of bars

full_plot_index = pd.RangeIndex(start=0, stop=8, step=1)  # Create index from 0 to 8
df_hvac_reindexed = df_hvac.reindex(full_plot_index, fill_value=0)  # Fill missing indices with 0

# Plot HVAC dataset 
df_hvac_reindexed.set_index(['Dataset', df_hvac_reindexed.index]).unstack(level=0).plot(
    kind='bar', stacked=True, ax=ax,
    color=blue_colors,
    width=bar_width, position=0
)

#ax.set_xticks(full_plot_index) 

# Plot MTHVDC_AC dataset
df_mthvdc_ac_reindexed = df_mthvdc_ac.reindex(full_plot_index, fill_value=0)  # Fill missing indices with 0

df_mthvdc_ac_reindexed.set_index(['Dataset', df_mthvdc_ac_reindexed.index]).unstack(level=0).plot(
    kind='bar', stacked=True, ax=ax,
    color=blue_colors,
    width=bar_width, position=1
)

# Plot MTHVDC_DC dataset
df_dc_plot = df_dc.copy()
df_dc_plot.reset_index(drop=True, inplace=True)
df_dc_plot.index = pd.RangeIndex(start=4, stop=4 + len(df_dc_plot), step=1)

df_dc_plot_reindexed = df_dc_plot.reindex(full_plot_index, fill_value=0)  # Fill missing indices with 0
# Rename the column '525 kV' to '525 kV DC'
df_dc_plot_reindexed.rename(columns={'525 kV': '525 kV DC'}, inplace=True)

df_dc_plot_reindexed.set_index(['Dataset', df_dc_plot_reindexed.index]).unstack(level=0).plot(
    kind='bar', stacked=True, ax=ax,
    color='#B4C8D2',
    width=bar_width, position=1
)

print(ax.get_xticks())
ax.set_ylabel('Number of Lines', fontweight='bold', fontsize=18)

indices = np.arange(n_bins)  # positions of the bars on the x-axis
ax.set_xticks(indices)
ax.set_xticklabels([''] * len(indices))  # Remove default tick labels

# Remove the default legend entries
#ax.get_legend().remove()
# Create custom legend entries only for the voltage levels (colors)
#handles = [
#    mpatches.Patch(color=blue_colors[0], label='230 kV'),
#    mpatches.Patch(color=blue_colors[1], label='345 kV'),
#    mpatches.Patch(color=blue_colors[2], label='500 kV'),
#    mpatches.Patch(color=blue_colors[3], label='765 kV'),
#    mpatches.Patch(color=blue_colors[4], label='525 kV DC')
#]


# Get the current legend handles and labels
handles, labels = ax.get_legend_handles_labels()

# Filter out entries where the label contains '0)'
filtered_handles_labels = [
    (handle, label) for handle, label in zip(handles, labels) if '0)' not in label
]

# Unzip the filtered handles and labels
filtered_handles, filtered_labels = zip(*filtered_handles_labels)

# Modify the labels to remove the "(HVAC)" or similar suffix
new_labels = [label.split(',')[0].strip('()') for label in filtered_labels]


# Remove duplicates by creating a mapping of unique labels to their first associated handle
unique_labels = {}
for handle, label in zip(filtered_handles, new_labels):
    if label not in unique_labels:
        unique_labels[label] = handle

# Set the updated legend with unique labels and their corresponding handles
ax.legend(unique_labels.values(), unique_labels.keys(), title="Voltage Level (kV)", loc='upper right')

#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles[::-1], labels[::-1])  # Reverse the legend order
#ax.legend(handles=handles, title="Voltage Level (kV)", loc='upper right')

# Adjust y-axis to make room for the scenario labels
ax.set_ylim(bottom=-70)
ax.spines['bottom'].set_position(('data', 0))  # Move the x-axis line to y=0

df_ac_dc_indices = list(df_ac.index[0:4]) + list(df_dc.index)
#df_ac_indices = list(df_ac.index[0:4]) + ['']*4
df_dc_indices = (['']*4) + list(df_dc.index)
for i in range(n_bins):
    if i in range(4):
        ax.text(indices[i] - bar_width / 2, -15, 'HVAC', ha='center', va='top', fontsize=14, rotation=90)
        ax.text(indices[i] + bar_width / 2, -15, 'MT-HVDC', ha='center', va='top', fontsize=14, rotation=90)
        ax.text(indices[i], -165, df_ac_dc_indices[i], ha='center', va='top', fontsize=16)  
    if i in range(4, n_bins):
        ax.text(indices[i], -15, 'MT-HVDC', ha='center', va='top', fontsize=14, rotation=90)
        ax.text(indices[i], -165, df_ac_dc_indices[i], ha='center', va='top', fontsize=16)  

ax.set_xlabel('Distance (miles)', labelpad=110, fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()



# !!!! Add this to the plot above 

# Sum the HVAC voltages DC is connected to, to the actual NTP assumed voltage for DC lines
#df_dc['525 kV'] = df_dc.select_dtypes(include='number').sum(axis=1)
# Drop the other columns
#df_dc = df_dc[['525 kV', 'Dataset']]
# Update the label of df_dc from "0-250" to "100-250" using .loc
#df_dc.loc["100-250"] = df_dc.loc["0-250"]
#df_dc.drop("0-250", inplace=True)
# The min value is 111 miles.. so adjust the label 0-250 to 100-250
""" fig, ax = plt.subplots(figsize=(10, 6))
df_dc.plot(kind='bar', stacked=True, ax=ax, color=['blue'])
# Customizing the DC plot
#ax.set_title('Stacked Bar Plot of MTHVDC_DC by Distance Bin and Voltage Level')
ax.set_xlabel('Distance (miles)')
ax.set_ylabel('Number of Lines')
ax.set_xticklabels(df_dc.index, rotation=45)  # Ensure DC distance bins are labeled correctly
# Add the custom hatch legend to the DC plot
ax.legend(title="Voltage Level (kV)", bbox_to_anchor=(0.70, 1), loc='upper left')
#ax.get_legend().set_visible(False)
plt.tight_layout()
plt.show() """