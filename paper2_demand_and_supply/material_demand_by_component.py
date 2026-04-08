import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from itertools import cycle


def read_material_data(material, excel_file='Grid_RING_inputs_v2_updated_with_MCS2025_to_push.xlsx'):
    df_mt = pd.read_excel(excel_file, sheet_name=f'MTGrid_2024MidCase_{material}', skiprows=6)
    df_ac = pd.read_excel(excel_file, sheet_name=f'ACGrid_2024MidCase_{material}', skiprows=6)

    years = [str(y) for y in range(2026, 2036)]

    # Group by 'Tech' and sum across years
    tech_mt = df_mt.groupby('Tech')[years].sum()
    tech_ac = df_ac.groupby('Tech')[years].sum()

    # Rename 'Cabling for PV, LBW, OSW' to 'Cabling'
    if 'Cabling for PV, LBW, OSW' in tech_mt.index:
        tech_mt = tech_mt.rename(index={'Cabling for PV, LBW, OSW': 'Cabling'})
        tech_ac = tech_ac.rename(index={'Cabling for PV, LBW, OSW': 'Cabling'})
    
    # Combine the OSW and LBW
    if 'OSW' in tech_mt.index and 'LBW' in tech_mt.index:
        tech_mt.loc["Wind"] = tech_mt.loc['LBW'] + tech_mt.loc['OSW']
        tech_ac.loc["Wind"] = tech_ac.loc['LBW'] + tech_ac.loc['OSW']
        # drop OWS and LBW
        tech_mt = tech_mt.drop(['LBW', 'OSW'], errors='ignore')
        tech_ac = tech_ac.drop(['LBW', 'OSW'], errors='ignore') 

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

    years = [str(y) for y in range(2026, 2036)]

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
materials = ['Al', 'Cu', 'Steel' ,'GOES']
for material in materials:
    plot_material_scenarios(material)



width = 0.35  # width of each "half" bar (MT or AC)
x = np.arange(len(materials))

transmission_techs = ['Transmission Lines', 'Towers', 'Transformers', 'Circuit Breakers']

# Gather all technologies across materials ---
all_techs_total = set()
for material in materials:
    _, _, all_techs = read_material_data(material)
    all_techs_total.update(all_techs)
#all_techs_total = sorted(all_techs_total)

# Reorder: priority techs first, then the rest
all_techs_total = [tech for tech in transmission_techs if tech in all_techs_total] + \
                  [tech for tech in sorted(all_techs_total) if tech not in transmission_techs]


# Assign consistent colors for all technologies ---
tech_colors = {tech: plt.cm.tab20(i / len(all_techs_total)) for i, tech in enumerate(all_techs_total)}
tech_colors.update({"Transmission Lines": "#0072B2", "Transformers": "#E69F00", "Circuit breakers": "#009E73", "Towers": "#56B4E9", "Batteries": "lightslategrey"})

# Create figure ---
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

# Plot stacked totals ---
for i, material in enumerate(materials):
    tech_mt, tech_ac, all_techs = read_material_data(material)

    # Sum across years (convert to million metric tons)
    mt_sum = tech_mt.sum(axis=1) / 1e6 / len(tech_mt.keys()) # per year
    ac_sum = tech_ac.sum(axis=1) / 1e6 / len(tech_mt.keys())

    mt_bottom = 0
    ac_bottom = 0

    for tech in all_techs_total:
        mt_value = mt_sum.get(tech, 0)
        ac_value = ac_sum.get(tech, 0)

        gap = 0.02
        # Plot AC bar first (left), then MT bar (right)
        ax.bar(i - width/2 - gap, ac_value, width, bottom=ac_bottom, color=tech_colors[tech])
        ac_bottom += ac_value

        ax.bar(i + width/2 + gap, mt_value, width, bottom=mt_bottom, color=tech_colors[tech])
        mt_bottom += mt_value

    # Place 'AC' label under the AC bar
    ax.text(i - width/2, -0.07, 'AC', ha='center', va='top', fontsize=14, transform=ax.get_xaxis_transform())

    # Place 'MT' label under the MT bar
    ax.text(i + width/2, -0.07, 'MT', ha='center', va='top', fontsize=14, transform=ax.get_xaxis_transform())

# Formatting ---
ax.set_xticks(x)
ax.set_xticklabels(materials)
ax.set_xlim(-0.5, len(materials) - 0.5)
ax.set_ylabel('Total material required \n (Millions of metric tons per year)')

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
plt.savefig('figures/material_per_year_by_scenario_and_tech.png', bbox_inches='tight')
plt.show()



## Calculations of relative percents of material for transmsission, other calcs..abs
material = 'Cu'
tech_mt, tech_ac, all_techs = read_material_data(material)

# transmission_techs
for label, tech in [('MT', tech_mt), ('AC', tech_ac)]:
    total = tech.sum(axis=1)
    percent = total[total.index.isin(transmission_techs)].sum() / total.sum() * 100
    print(f'Percent material {material} {label} scenario:  {percent}')

# percent of steel used for towers
material = 'GOES'
tech_mt, tech_ac, all_techs = read_material_data(material)
for label, tech in [('MT', tech_mt), ('AC', tech_ac)]:
    #print(label)
    #print(tech)
    total = tech.sum(axis=1)
    percent = total[total.index.isin(['Towers'])].sum() / total.sum() * 100
    print(f'Percent material {material} {label} scenario:  {percent}')