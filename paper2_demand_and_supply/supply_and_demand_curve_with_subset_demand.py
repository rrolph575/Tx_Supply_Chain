import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import FuncFormatter


# --- Parameters ---
scenario = 'MT'  # 'AC' or 'MT'
sheet = scenario+'Grid_2024MidCase'  # main sheet
selected_material = 'aluminum' # 'copper' or 'aluminum'

if selected_material == 'copper':
    ring_breakdown_sheet = scenario + 'Grid_2024MidCase_Cu'
else:
    ring_breakdown_sheet = scenario + 'Grid_2024MidCase_Al'

# --- Figure setup ---
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 18,
    'axes.labelsize': 35,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 25,
    'legend.title_fontsize': 18,
})

fig, ax = plt.subplots(figsize=(16, 12))

# --- Read main data ---
#df = pd.read_excel('Grid_RING_inputs_v2.xlsx', sheet_name=sheet, skiprows=6)
df = pd.read_excel('Grid_RING_inputs_v2_updated_with_MCS2025.xlsx', sheet_name=sheet, skiprows=6)
df_filtered = df[df['Refined Material'] == selected_material]

# --- Year columns & numeric x positions ---
year_cols = [col for col in df.columns if str(col).isdigit() and int(col) >= 2026]
x = np.arange(len(year_cols))  # numeric x positions

# --- Line styles and labels ---
linestyle_map = {
    'RING projected demand': '-',
    'Total domestic supply': '-',
    'USGS projected demand': ':'
    }
label_map = {
    'RING projected demand': 'Electricity sector demand (RING)',
    'Total domestic supply': 'Total domestic supply',
    'USGS projected demand': 'All sectors demand (USGS)'
}

legend_handles = []
df_all = []

# --- Plot main lines ---
# Remove CMA 
df_filtered = df_filtered[df_filtered['Term'] != 'DOE CMA project demand']
for term, row in df_filtered.groupby('Term'):
    linestyle = linestyle_map.get(term, '-')
    new_label = label_map.get(term, term)
    y_values = row[year_cols].values[0] / 1e6

    df_all.append({'case': sheet, 'data source': term, 'years': year_cols, 'values': y_values})
    color = 'grey' if term == 'Total domestic supply' else 'orange'

    ax.plot(
        x,
        y_values,
        linestyle=linestyle,
        linewidth=5,
        color=color,
        label=new_label
    )
    df_all.append({'material': selected_material})

    # Add manual legend handle
    handle = Line2D([0], [0], color=color, linestyle=linestyle, linewidth=5, label=new_label)
    legend_handles.append(handle)

# --- X-axis formatting ---
ax.set_xticks(x)
ax.tick_params(axis='x')
ax.tick_params(axis='y')
ax.set_xticklabels(year_cols, rotation=90)
ax.set_ylabel(f'{selected_material.capitalize()} (million metric tons)')
#ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))

def smart_formatter(y, pos):
    yticks = ax.get_yticks()
    if np.allclose(yticks, np.round(yticks)):
        return f'{int(y)}'
    else:
        return f'{y:.1f}'

ax.yaxis.set_major_formatter(FuncFormatter(smart_formatter))

ax.set_xlim(x[0]-0.2, x[-1]+0.5)
ax.grid(True)

# Legend in desired order
desired_order = [
    'All sectors demand (USGS)',
    'Electricity sector demand (RING)',
    'Total domestic supply'
]
handle_dict = {h.get_label(): h for h in legend_handles}
legend_handles_sorted = [handle_dict[label] for label in desired_order if label in handle_dict]
ax.legend(handles=legend_handles_sorted, loc='upper left', bbox_to_anchor=(0.05,0.35), handlelength=6)

# --- RING values for pointer lines ---
df_ring = df_filtered[df_filtered['Term'] == 'RING projected demand']
ring_2026_value = df_ring['2026'].values[0] / 1e6
ring_2035_value = df_ring['2035'].values[0] / 1e6
x_2026 = np.where(np.array(year_cols) == '2026')[0][0]
x_2035 = np.where(np.array(year_cols) == '2035')[0][0]

# --- Read tech breakdown ---
df_tech = pd.read_excel('Grid_RING_inputs_v2_updated_with_MCS2025.xlsx', sheet_name=ring_breakdown_sheet, skiprows=6)
df_filtered_tech = df_tech[df_tech['Refined Materials'] == selected_material].copy()

# Combine tech categories
df_filtered_tech["tech_combined"] = np.select(
    [df_filtered_tech["Tech"] == "Circuit breakers",
     df_filtered_tech["Tech"] == "Transmission Lines"],
    ["Circuit breakers", "Transmission"],
    default="Generation and Storage"
)

#scenario = 'MT'  # 'AC' or 'MT'
#sheet = scenario+'Grid_2024MidCase'  # main sheet
#selected_material = 'copper'

plt.savefig(f'figures/supply_demand_{selected_material}_{scenario}.png', bbox_inches='tight')


plt.show()
