

import matplotlib.pyplot as plt
import matplotlib.image as mpimg




figpath = 'figures/'

al_AC, al_MT = [f"{figpath}supply_demand_aluminum_{s}.png" for s in ['AC', 'MT']]
cu_AC, cu_MT = [f"{figpath}supply_demand_copper_{s}.png" for s in ['AC', 'MT']]

#%% Make combined figure

fig, axes = plt.subplots(2, 2, figsize=(16,12))
plt.subplots_adjust(wspace=0.05, hspace=0.15)

images = [al_AC, al_MT, cu_AC, cu_MT]
labels = ['Al (AC)', 'Al (MT)', 'Cu (AC)', 'Cu (MT)']
panel_labels = ['(a)', '(b)', '(c)', '(d)']

for idx, ax in enumerate(axes.flat):
    img = mpimg.imread(images[idx])
    ax.imshow(img)
    ax.axis('off')
    # Combined label (bottom-center)
    ax.text(0.5, -0.08, f"{panel_labels[idx]} {labels[idx]}", transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='center', color='black')