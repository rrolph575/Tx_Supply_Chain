import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_demand_and_supply_curve(selected_material):
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
        year_cols = [col for col in df.columns if str(col).isdigit()]

        for term, row in df_filtered.groupby('Term'):
            y_values = row[year_cols].values[0] / 1e6

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

    if selected_material in ['copper', 'aluminum']:
        plt.ylabel(f"{selected_material.capitalize()} (millions of metric tons)")

    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave room for right-side legend
    #plt.title(f"{selected_material.capitalize()} Demand and Supply Curves")

    # Create MT/AC custom legend
    line_mt = Line2D([0], [0], color='black', linewidth=2, linestyle='-', label='MT')
    line_ac = Line2D([0], [0], color='black', linewidth=3, linestyle='--', label='AC')

    # Main legend (terms)
    term_legend = plt.legend(loc='upper left')

    # Add MT/AC legend to the right
    plt.gca().add_artist(term_legend)  # keep term legend visible
    # --- Position second legend directly to the right of first legend ---
    fig.canvas.draw()  # ensure positions are updated
    bbox = term_legend.get_window_extent()
    inv = fig.transFigure.inverted()
    bbox_fig = inv.transform(bbox)

    x_right = bbox_fig[1, 0] + 0.015  # small horizontal gap
    y_top = bbox_fig[1, 1] 

    ax.legend(
        handles=[line_mt, line_ac],
        loc='upper left',
        bbox_to_anchor=(x_right, y_top +0.02),
        frameon=True
    )

    plt.show()


### Run for both materials
materials = ['copper', 'aluminum']
for material in materials:
    plot_demand_and_supply_curve(material)
