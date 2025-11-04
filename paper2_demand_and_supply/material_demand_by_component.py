import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def plot_material_scenarios(material, excel_file='Grid_RING_inputs_v2.xlsx'):
    """
    Plot a stacked bar chart of material required by year and technology,
    comparing two scenarios: MT and AC.
    MT = solid bars, AC = hatched bars. Colors indicate technology.
    Legend shows technologies by color, scenario by hatched/solid white boxes.
    """

    years = [str(y) for y in range(2024, 2036)]

    # Read MT and AC data
    df_mt = pd.read_excel(excel_file, sheet_name=f'MTGrid_2024MidCase_{material}', skiprows=6)
    df_ac = pd.read_excel(excel_file, sheet_name=f'ACGrid_2024MidCase_{material}', skiprows=6)

    tech_mt = df_mt.groupby('Tech')[years].sum()
    tech_ac = df_ac.groupby('Tech')[years].sum()

    # Union of all technologies
    all_techs = tech_mt.index.union(tech_ac.index)

    # Ensure both DataFrames have all techs (fill missing with zeros)
    tech_mt = tech_mt.reindex(all_techs, fill_value=0)
    tech_ac = tech_ac.reindex(all_techs, fill_value=0)

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
