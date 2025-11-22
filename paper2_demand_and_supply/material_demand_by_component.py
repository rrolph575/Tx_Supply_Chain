import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from itertools import cycle


def read_material_data(material, excel_file='Grid_RING_inputs_v2.xlsx'):
    df_mt = pd.read_excel(excel_file, sheet_name=f'MTGrid_2024MidCase_{material}', skiprows=6)
    df_ac = pd.read_excel(excel_file, sheet_name=f'ACGrid_2024MidCase_{material}', skiprows=6)

    years = [str(y) for y in range(2024, 2036)]

    # Group by 'Tech' and sum across years
    tech_mt = df_mt.groupby('Tech')[years].sum()
    tech_ac = df_ac.groupby('Tech')[years].sum()

    # Union of all technologies
    all_techs = tech_mt.index.union(tech_ac.index)

    # Ensure both DataFrames have all techs (fill missing with zeros)
    tech_mt = tech_mt.reindex(all_techs, fill_value=0)
    tech_ac = tech_ac.reindex(all_techs, fill_value=0)

    return tech_mt, tech_ac, all_techs


def plot_material_scenarios(material):
    """
    Plot a stacked bar chart of material required by year and technology,
    comparing two scenarios: MT and AC.
    MT = solid bars, AC = hatched bars. Colors indicate technology.
    Legend shows technologies by color, scenario by hatched/solid white boxes.
    """

    years = [str(y) for y in range(2024, 2036)]

    # Read MT and AC data
    tech_mt, tech_ac, all_techs = read_material_data(material)

    # Convert to millions of metric tons
    tech_mt = tech_mt / 1e6
    tech_ac = tech_ac / 1e6
    ylabel = f'{material} required \n(Millions of metric tons)'

    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 12,
    })
    plt.figure(figsize=(12, 6))

    # Stacked bar plot with solid MT, hatched AC
    x = np.arange(len(years))
    width = 0.4
    colors = plt.cm.tab20.colors  # up to 20 distinct colors

    # MT bars (solid)
    bottom_mt = np.zeros(len(years))
    for i, tech in enumerate(all_techs):
        plt.bar(x - width/2, tech_mt.loc[tech], width=width, bottom=bottom_mt,
                color=colors[i % len(colors)], label=tech)
        bottom_mt += tech_mt.loc[tech]

    # AC bars (hatched)
    bottom_ac = np.zeros(len(years))
    for i, tech in enumerate(all_techs):
        plt.bar(x + width/2, tech_ac.loc[tech], width=width, bottom=bottom_ac,
                color=colors[i % len(colors)], hatch='//')
        bottom_ac += tech_ac.loc[tech]

    plt.xticks(x, years, rotation=90)
    plt.ylabel(ylabel)
    
    if material == 'Steel':
        plt.ylim(0, 0.7)

    # Create custom legend
    handles_tech = [Patch(facecolor=colors[i % len(colors)], label=tech) for i, tech in enumerate(all_techs)]
    handles_scenario = [Patch(facecolor='white', edgecolor='black', hatch='', label='MT'),
                        Patch(facecolor='white', edgecolor='black', hatch='//', label='AC')]
    plt.legend(handles=handles_tech + handles_scenario, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


# Procedure
materials = ['Al', 'Cu', 'Steel']  # GOES removed
for material in materials:
    plot_material_scenarios(material)



width = 0.35  # width of each "half" bar (MT or AC)
x = np.arange(len(materials))

# Gather all technologies across materials ---
all_techs_total = set()
for material in materials:
    _, _, all_techs = read_material_data(material)
    all_techs_total.update(all_techs)
all_techs_total = sorted(all_techs_total)

# Assign consistent colors for all technologies ---
tech_colors = {tech: plt.cm.tab20(i / len(all_techs_total)) for i, tech in enumerate(all_techs_total)}
tech_colors.update({"Transmission Lines": "#0072B2", "Transformers": "#E69F00", "Circuit breakers": "#009E73", "Towers": "#56B4E9", "Batteries": "lightslategrey"})

# Create figure ---
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

# Plot stacked totals ---
for i, material in enumerate(materials):
    tech_mt, tech_ac, all_techs = read_material_data(material)

    # Sum across years (convert to million metric tons)
    mt_sum = tech_mt.sum(axis=1) / 1e6
    ac_sum = tech_ac.sum(axis=1) / 1e6

    mt_bottom = 0
    ac_bottom = 0

    for tech in all_techs_total:
        mt_value = mt_sum.get(tech, 0)
        ac_value = ac_sum.get(tech, 0)

        # MT solid
        ax.bar(i - width/2, mt_value, width, bottom=mt_bottom, color=tech_colors[tech])
        mt_bottom += mt_value

        # AC hatched
        ax.bar(i + width/2, ac_value, width, bottom=ac_bottom, color=tech_colors[tech], hatch='//')
        ac_bottom += ac_value

# Formatting ---
ax.set_xticks(x)
ax.set_xticklabels(materials)
ax.set_xlim(-0.5, len(materials) - 0.5)
ax.set_ylabel('Total material required \n (Millions of metric tons)')

# Legends ---
# --- Tech legend: top left inside plot, 3 columns ---
tech_handles = [Patch(facecolor=tech_colors[tech], label=tech) for tech in all_techs_total]
tech_legend = ax.legend(
    handles=tech_handles,
    loc='upper left',
    bbox_to_anchor=(0.01, 1.0),
    frameon=True,
    fontsize=12,
    ncol=2  # 3 columns for techs
)
ax.add_artist(tech_legend)
# --- Scenario legend: top right inside plot, 1 column ---
mt_patch = Patch(facecolor='white', edgecolor='black', label='MT')
ac_patch = Patch(facecolor='white', edgecolor='black', hatch='//', label='AC')
ax.legend(
    handles=[mt_patch, ac_patch],
    loc='upper right',
    bbox_to_anchor=(0.99, 1.0),
    frameon=True,
    fontsize=12,
    ncol=1  # single column
)

plt.show()