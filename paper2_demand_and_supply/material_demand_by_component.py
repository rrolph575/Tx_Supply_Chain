width = 0.35  # width of each "half" bar (MT or AC)
x = np.arange(len(materials))

fig, ax = plt.subplots(figsize=(10, 6))  # slightly smaller width

for i, material in enumerate(materials):
    tech_mt = read_material_data(material, 'MT', excel_file)
    tech_ac = read_material_data(material, 'AC', excel_file)

    mt_sum = tech_mt.sum(axis=1) / 1e6
    ac_sum = tech_ac.sum(axis=1) / 1e6

    mt_bottom = 0
    ac_bottom = 0

    for tech in all_techs:
        mt_value = mt_sum.get(tech, 0)
        ac_value = ac_sum.get(tech, 0)

        # MT solid
        ax.bar(i - width/2, mt_value, width, bottom=mt_bottom, color=tech_colors[tech])
        mt_bottom += mt_value

        # AC hatched
        ax.bar(i + width/2, ac_value, width, bottom=ac_bottom, color=tech_colors[tech], hatch='//')
        ac_bottom += ac_value

# Reduce x-axis spacing by adjusting limits
ax.set_xticks(x)
ax.set_xticklabels(materials)
ax.set_xlim(-0.5, len(materials)-0.5)  # tighter limits

# Legends same as before
tech_handles = [mpatches.Patch(facecolor=tech_colors[tech], label=tech) for tech in all_techs]
tech_legend = ax.legend(handles=tech_handles, bbox_to_anchor=(1, 1), loc='upper left', frameon=True)
mt_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='MT')
ac_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='AC')
scenario_legend = ax.legend(handles=[mt_patch, ac_patch], bbox_to_anchor=(1, 0.4), loc='upper left')
ax.add_artist(tech_legend)

plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.show()
