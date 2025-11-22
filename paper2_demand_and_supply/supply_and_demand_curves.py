import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pickle
import matplotlib.lines as mlines
from matplotlib.ticker import ScalarFormatter


calculate_material_supply_and_demand = False # Calculte material supply and demand weights and save to pickle files

def plot_demand_and_supply_curve(selected_material):


    df_all = []

    sheets = ['MTGrid_2024MidCase', 'ACGrid_2024MidCase']
    linestyles = {'MTGrid_2024MidCase': '-', 'ACGrid_2024MidCase': '--'}
    linewidths = {'MTGrid_2024MidCase': 2, 'ACGrid_2024MidCase': 3}  # AC thicker

    # Set up figure
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 15,
        'legend.title_fontsize': 18,
    })
    fig, ax = plt.subplots(figsize=(13.5, 8))  

    color_map = {}
    color_cycle = plt.cm.tab10.colors
    color_index = 0

    for sheet in sheets:
        df = pd.read_excel('Grid_RING_inputs_v2.xlsx', sheet_name=sheet, skiprows=6)
        df_filtered = df[df['Refined Material'] == selected_material]

        # Only keep the specific row for ACGrid
        if sheet == 'ACGrid_2024MidCase':
            df_filtered = df_filtered[df_filtered['Term'] == 'RING projected demand']

        year_cols = [col for col in df.columns if str(col).isdigit()]

        for term, row in df_filtered.groupby('Term'):
            y_values = row[year_cols].values[0] / 1e6

            df_all.append({
                'case': sheet,
                'data source': term,
                'years': year_cols,
                'values': y_values
            })

            if term not in color_map:
                color_map[term] = color_cycle[color_index % len(color_cycle)]
                color_index += 1

            plt.plot(
                year_cols,
                y_values,
                marker='o',
                linestyle=linestyles[sheet],
                linewidth=linewidths[sheet],
                color=color_map[term],
                label=term if sheet == 'MTGrid_2024MidCase' else None,
            )
        df_all.append({'material': selected_material})

    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave room for right-side legend

    # Create MT/AC custom legend
    line_mt = Line2D([0], [0], color='orange', linewidth=2, linestyle='-', label='MT')
    line_ac = Line2D([0], [0], color='orange', linewidth=3, linestyle='--', label='AC')

    # Main legend (terms)
    term_legend = plt.legend(loc='upper left')

    # Add MT/AC legend to the right
    plt.gca().add_artist(term_legend)  # keep term legend visible
    fig.canvas.draw()  # ensure positions are updated
    bbox = term_legend.get_window_extent()
    inv = fig.transFigure.inverted()
    bbox_fig = inv.transform(bbox)

    x_right = bbox_fig[1, 0] + 0.015  # small horizontal gap
    y_top = bbox_fig[1, 1] 

    ax.legend(
        handles=[line_mt, line_ac],
        loc='upper left',
        bbox_to_anchor=(x_right, y_top + 0.04),
        frameon=True
    )

    plt.show()
    return df_all


### Run for both materials
if calculate_material_supply_and_demand == True:
    materials = ['copper', 'aluminum']
    for material in materials:
        df_all = plot_demand_and_supply_curve(material)
        with open(f'supply_and_demand_{material}.pkl', 'wb') as f:
            pickle.dump(df_all, f)
    


### Calculate GOES supply and demand and save to pickle
material = ['GOES']
cases = ['MTGrid_2024MidCase', 'ACGrid_2024MidCase']
years = [str(y) for y in range(2020, 2036)]
# Define demand GOES per case
demand_dict = {
    'MTGrid_2024MidCase': 85.250,
    'ACGrid_2024MidCase': 103.450
}
df_all = []
for case in cases:
    # Supply GOES
    df_all.append({
        'case': case,
        'data source': 'USMCA GOES supply',
        'years': years,
        'values': np.full(len(years), 576.207, dtype=int),
        'material': material
    })
    # Demand GOES
    df_all.append({
        'case': case,
        'data source': 'RING projected demand',
        'years': years,
        'values': np.full(len(years), demand_dict[case], dtype=int),
        'material': material
    })
# Save to pickle
with open('supply_and_demand_GOES.pkl', 'wb') as f:
    pickle.dump(df_all, f)




### Create separate demand and supply plots, also difference plots
materials = ['copper', 'aluminum', 'GOES']
all_rows = []
for material in materials:
    with open(f'supply_and_demand_{material}.pkl', 'rb') as f:
        dict_material = pickle.load(f)
        series_blocks = [d for d in dict_material if 'values' in d] # Extract all dicts that have values
        for series in series_blocks:
            for year, value in zip(series['years'], series['values']):
                all_rows.append({
                    'material': material,
                    'case': series['case'],
                    'data source': series['data source'],
                    'year': year,
                    'value': value
                })  
df = pd.DataFrame(all_rows)


materials = df['material'].unique()
case_linestyle = {'MTGrid_2024MidCase': '-', 'ACGrid_2024MidCase': '--'}

# Map full case names to short labels
case_label_map = {
    'MTGrid_2024MidCase': 'MT',
    'ACGrid_2024MidCase': 'AC'
}

# Assign colors to data sources
data_sources = df['data source'].unique()
colors = plt.cm.tab10.colors
ds_colors = {ds: colors[i % len(colors)] for i, ds in enumerate(data_sources)}

fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)

# Row y-axis labels
#y_labels = ['Supply (millions of metric tons)', 'Demand (millions of metric tons)', 'Supply - Demand (metric tons)']
y_labels_base = ['Supply', 'Demand', 'Supply - Demand']

for row_idx in range(3):
    for col_idx, material in enumerate(materials):
        df_mat = df[df['material'] == material]
        
        if row_idx == 0:  # Supply
            ax = axes[row_idx, col_idx]
            df_supply = df_mat[df_mat['data source'].str.contains('supply', case=False)]
            for ds in df_supply['data source'].unique():
                df_ds = df_supply[df_supply['data source'] == ds]
                for case in df_ds['case'].unique():
                    df_case = df_ds[df_ds['case'] == case]
                    ax.plot(df_case['year'], df_case['value'],
                            color=ds_colors[ds],
                            linestyle=case_linestyle.get(case, '-'))
            ax.set_title(material if material == 'GOES' else material.capitalize())

        elif row_idx == 1:  # Demand
            ax = axes[row_idx, col_idx]
            df_demand = df_mat[df_mat['data source'].str.contains('demand', case=False)]
            for ds in df_demand['data source'].unique():
                df_ds = df_demand[df_demand['data source'] == ds]
                for case in df_ds['case'].unique():
                    df_case = df_ds[df_ds['case'] == case]
                    ax.plot(df_case['year'], df_case['value'],
                            color=ds_colors[ds],
                            linestyle=case_linestyle.get(case, '-'))

        elif row_idx == 2:  # Difference
            ax = axes[row_idx, col_idx]
            df_supply = df_mat[df_mat['data source'].str.contains('supply', case=False)]
            df_demand = df_mat[df_mat['data source'].str.contains('demand', case=False)]
            for ds_supply in df_supply['data source'].unique():
                df_s = df_supply[df_supply['data source'] == ds_supply]
                for ds_demand in df_demand['data source'].unique():
                    df_d = df_demand[df_demand['data source'] == ds_demand]
                    df_diff = pd.merge(df_s, df_d, on=['year', 'case', 'material'],
                                       suffixes=('_supply', '_demand'))
                    for case in df_diff['case'].unique():
                        df_case = df_diff[df_diff['case'] == case]
                        ax.plot(df_case['year'], df_case['value_supply'] - df_case['value_demand'],
                                color=ds_colors[ds_demand],
                                linestyle=case_linestyle.get(case, '-'))

        # Set y-axis label for the row
        # Determine units based on material
        units = 'thousands of metric tons' if material == 'GOES' else 'millions of metric tons'
        ax.set_ylabel(f"{y_labels_base[row_idx]} ({units})")

        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=90)

# --- Create shared legends ---
# Data source legend (colors)
ds_handles = [mlines.Line2D([], [], color=ds_colors[ds], label=ds) for ds in data_sources]
fig.legend(handles=ds_handles, loc='upper center', ncol=5, fontsize=10)

# Case legend (linestyle)
case_handles = [
    mlines.Line2D([], [], color='black', linestyle=case_linestyle[case], label=case_label_map[case])
    for case in case_linestyle
]
fig.legend(handles=case_handles, title='Case', loc='upper right', fontsize=10)
for row in axes:
    for ax in row:
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.tick_params(axis='x', rotation=90)  # keep x-axis rotated
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
