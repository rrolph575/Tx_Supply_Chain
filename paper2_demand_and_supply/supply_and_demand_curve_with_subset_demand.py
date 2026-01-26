import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch, Rectangle

# --- Parameters ---
scenario = 'MT'  # 'AC' or 'MT'
sheet = scenario+'Grid_2024MidCase'  # main sheet
selected_material = 'copper' # 'copper' or 'aluminum'

if selected_material == 'copper':
    ring_breakdown_sheet = scenario + 'Grid_2024MidCase_Cu'
else:
    ring_breakdown_sheet = scenario + 'Grid_2024MidCase_Al'

# --- Figure setup ---
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 15,
    'legend.title_fontsize': 18,
})

fig, ax = plt.subplots(figsize=(16, 12))

# --- Read main data ---
#df = pd.read_excel('Grid_RING_inputs_v2.xlsx', sheet_name=sheet, skiprows=6)
df = pd.read_excel('Grid_RING_inputs_v2_updated_with_MCS2025.xlsx', sheet_name=sheet, skiprows=6)
df_filtered = df[df['Refined Material'] == selected_material]

# --- Year columns & numeric x positions ---
year_cols = [col for col in df.columns if str(col).isdigit() and int(col) >= 2025]
x = np.arange(len(year_cols))  # numeric x positions

# --- Line styles and labels ---
linestyle_map = {
    'RING projected demand': '--',
    'Total domestic supply': '-',
    'USGS projected demand': ':',
    'DOE CMA project demand': '-.',
}
label_map = {
    'RING projected demand': 'Electricity sector demand (RING)',
    'Total domestic supply': 'Total domestic supply',
    'USGS projected demand': 'All sectors demand (USGS)',
    'DOE CMA project demand': 'All sectors demand (DOE CMA)',
}

legend_handles = []
df_all = []

# --- Plot main lines ---
for term, row in df_filtered.groupby('Term'):
    linestyle = linestyle_map.get(term, '-')
    new_label = label_map.get(term, term)
    y_values = row[year_cols].values[0] / 1e6

    df_all.append({'case': sheet, 'data source': term, 'years': year_cols, 'values': y_values})
    color = 'blue' if term == 'Total domestic supply' else 'orange'

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
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_xticklabels(year_cols, rotation=90)
ax.set_ylabel(f'{selected_material.capitalize()} (million metric tons)', fontsize=25)
ax.set_xlim(x[0]-0.2, x[-1]+0.5)
ax.grid(True)

# Legend in desired order
desired_order = [
    'All sectors demand (DOE CMA)',
    'All sectors demand (USGS)',
    'Electricity sector demand (RING)',
    'Total domestic supply'
]
handle_dict = {h.get_label(): h for h in legend_handles}
legend_handles_sorted = [handle_dict[label] for label in desired_order if label in handle_dict]
ax.legend(handles=legend_handles_sorted, loc='upper left', handlelength=6)

# --- RING values for pointer lines ---
df_ring = df_filtered[df_filtered['Term'] == 'RING projected demand']
ring_2025_value = df_ring['2025'].values[0] / 1e6
ring_2035_value = df_ring['2035'].values[0] / 1e6
x_2025 = np.where(np.array(year_cols) == '2025')[0][0]
x_2035 = np.where(np.array(year_cols) == '2035')[0][0]

# --- Read tech breakdown ---
df_tech = pd.read_excel('Grid_RING_inputs_v2.xlsx', sheet_name=ring_breakdown_sheet, skiprows=6)
df_filtered_tech = df_tech[df_tech['Refined Materials'] == selected_material].copy()

# Combine tech categories
df_filtered_tech["tech_combined"] = np.select(
    [df_filtered_tech["Tech"] == "Circuit breakers",
     df_filtered_tech["Tech"] == "Transmission Lines"],
    ["Circuit breakers", "Transmission"],
    default="Generation and Storage"
)

# --- Define consistent colors for each tech ---
tech_colors = {
    'Circuit breakers': '#3a3a3a',        # dark grey
    'Transmission': '#969696',            # medium grey
    'Generation and Storage': '#d9d9d9'  # light grey
}

# --- Function to create percent stacked bar inset ---
def plot_inset_percent(ax_main, df_tech_data, year, inset_pos):
    df_year = df_tech_data[['tech_combined', year]].groupby('tech_combined', as_index=False).sum()
    df_year[year] = df_year[year] / 1e6
    df_year_filtered = df_year[df_year[year] > 1e-10]

    total = df_year_filtered[year].sum()
    if total == 0:
        return None  # skip if nothing to plot

    ax_inset = ax_main.inset_axes(inset_pos)

    bottom = 0
    x_bar = 0
    for tech, row in df_year_filtered.set_index('tech_combined').iterrows():
        height = row[year] / total * 100  # percent
        if height <= 0.06:
            continue

        ax_inset.bar(
            x_bar,
            height,
            bottom=bottom,
            color=tech_colors.get(tech, 'gray')
        )

        # Label with newline if >2 words
        words = tech.split()
        label = '\n'.join(words) if len(words) > 2 else tech

        # Compute y position for label
        y_label = bottom + 0.6 * height
        if tech == 'Circuit breakers':
            y_label += 8  # move up slightly

        ax_inset.text(
            x_bar,
            y_label,
            label + f" ({height:.1f}%)",
            ha='center',
            va='center',
            fontsize=18,
            fontweight='bold',
            color='black'
        )

        bottom += height

    # Format inset
    ax_inset.set_yticks([])
    ax_inset.set_xticks([])
    for spine in ax_inset.spines.values():
        spine.set_visible(False)

    return ax_inset

# --- Function to add vertical rectangle ---
def add_vertical_rect(ax_main, x_pos, height, width=0.3, color='black', alpha=0.2):
    rect = Rectangle(
        (x_pos - width/2, 0),
        width,
        height,
        color=color,
        alpha=alpha
    )
    ax_main.add_patch(rect)
    return rect

# --- 2025 rectangle and inset connection ---
rect_2025 = add_vertical_rect(ax, x_2025, ring_2025_value)
if selected_material == 'aluminum':
    ax_inset_2025 = plot_inset_percent(ax, df_filtered_tech, '2025', [0.03, 0.1, 0.25, 0.3])
if selected_material =='copper':
    ax_inset_2025 = plot_inset_percent(ax, df_filtered_tech, '2025', [0.05, 0.15, 0.25, 0.15])
con1 = ConnectionPatch(
    xyA=(0.5, 0.5),               # center of inset
    coordsA=ax_inset_2025.transAxes,
    xyB=(x_2025, ring_2025_value),  # top of rectangle
    coordsB=ax.transData,
    color='black',
    linewidth=2,
    linestyle='--'
)
ax.add_artist(con1)

# --- 2035 rectangle and inset connection ---
rect_2035 = add_vertical_rect(ax, x_2035, ring_2035_value)
if selected_material=='aluminum':
    ax_inset_2035 = plot_inset_percent(ax, df_filtered_tech, '2035', [0.6, 0.1, 0.35, 0.3])
if selected_material=='copper':
    ax_inset_2035 = plot_inset_percent(ax, df_filtered_tech, '2035', [0.70, 0.15, 0.25, 0.15])
# Make inset bars appear on top
ax_inset_2035.set_zorder(5)
ax_inset_2025.set_zorder(5)


con2 = ConnectionPatch(
    xyA=(0.5, 0.5),
    coordsA=ax_inset_2035.transAxes,
    xyB=(x_2035, ring_2035_value),
    coordsB=ax.transData,
    color='black',
    linewidth=2,
    linestyle='--'
)
ax.add_artist(con2)



#scenario = 'MT'  # 'AC' or 'MT'
#sheet = scenario+'Grid_2024MidCase'  # main sheet
#selected_material = 'copper'

plt.savefig(f'figures/supply_demand_{selected_material}_{scenario}.png', bbox_inches='tight')


plt.show()
