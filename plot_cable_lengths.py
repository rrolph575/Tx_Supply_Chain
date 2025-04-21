import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


datapath = "C:/Users/rrolph/OneDrive - NREL/Projects/FY25/Transmission_Supply_Chain/"

# read pkl files saved from find_cable_lengths*py
use_avg = False # True if use avg number of cables in bundle.
print('If plots are using avg number of bundles of cable per line: ' + str(use_avg))
hvac_data = pd.read_pickle(datapath+'HVAC_Scenario_HVAC_lines' + str(use_avg) + '_avg_num_bundles.pkl')
mthvdc_dc_data = pd.read_pickle(datapath+'MTHVDC_Scenario_HVDC_lines' + str(use_avg) + '_avg_num_bundles.pkl')
mthvdc_ac_data = pd.read_pickle(datapath+'MTHVDC_Scenario_HVAC_lines' + str(use_avg) + '_avg_num_bundles.pkl')

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

# Set the 'Distance_Bin' as the index for better plotting control
df_ac.set_index('Distance_Bin', inplace=True)
df_dc.set_index('Distance_Bin', inplace=True)

# Plotting AC stacked bar plot with separate bars for HVAC and MTHVDC_AC
fig, ax = plt.subplots(figsize=(12, 6))

# Number of distance bins
n_bins = len(df_ac.index)

# Define positions for bars (split each bin into two bars)
bar_width = 0.35  # width of each bar
indices = np.arange(n_bins)  # positions of the bars on the x-axis

# Plot HVAC dataset with hatch pattern (shifted by bar_width/2)
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
color_legend_230 = mpatches.Patch(color='#1f77b4', label='230 kV')
color_legend_345 = mpatches.Patch(color='#ff7f0e', label='345 kV')
color_legend_500 = mpatches.Patch(color='#2ca02c', label='500 kV')
color_legend_765 = mpatches.Patch(color='#d62728', label='765 kV')

# Create a custom hatch pattern legend entry for HVAC
hatch_legend = mpatches.Patch(hatch='//', label='HVAC Scenario', edgecolor='black', facecolor='white')

# Create a custom colorless patch for MTHVDC_AC scenario
mthvdc_legend = mpatches.Patch(hatch=None, label='MTHVDC Scenario', edgecolor='black', facecolor='white')

# Combine the custom legend entries
handles.extend([color_legend_230, color_legend_345, color_legend_500, color_legend_765, hatch_legend, mthvdc_legend])

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
df_dc.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

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
