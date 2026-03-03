
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as transforms



df = pd.read_csv('supply_and_demand_component_number_summary.csv')


component_labels = ['Transmission lines',  'Transformers', 'Circuit Breakers', 'Towers']


## Build grouped values

# Transmission lines
trans = df[df.component == "transmission_line"][["supply", "demand_ac", "demand_mt"]].iloc[0]

# Transformers → average of high + low (domestic + imported)
transf = df[df.component.str.contains("transformer")]
# sum high and low
low_vals  = transf[transf.component.str.contains("low")][["supply", "demand_ac", "demand_mt"]].sum()
high_vals = transf[transf.component.str.contains("high")][["supply", "demand_ac", "demand_mt"]].sum()
# mean values (bar height)
transf_mean = (low_vals + high_vals) / 2
# error bars = distance from mean to summed values
transf_err = np.abs(high_vals - low_vals) / 2

# Circuit breakers
cb = df[df.component == "circuit_breaker"][["supply", "demand_ac", "demand_mt"]].iloc[0]

# Towers
tower = df[df.component == "tower"][["supply", "demand_ac", "demand_mt"]].iloc[0]

# Combine into plotting arrays
supply_vals = [trans.supply, transf_mean.supply, cb.supply, tower.supply]
ac_vals     = [trans.demand_ac, transf_mean.demand_ac, cb.demand_ac, tower.demand_ac]
mt_vals     = [trans.demand_mt, transf_mean.demand_mt, cb.demand_mt, tower.demand_mt]

# Error bars only for transformer group (index 1)
supply_err = [0, transf_err.supply, 0, 0]
ac_err     = [0, transf_err.demand_ac, 0, 0]
mt_err     = [0, transf_err.demand_mt, 0, 0]

# --- Plot ---
x = np.arange(len(component_labels))
width = 0.25

fig, ax1 = plt.subplots(figsize=(9,5))
ax2 = ax1.twinx()  # Create a second y-axis
left_groups = [0,2]  # indices from components, transmission lines and circuit breakers
right_groups = [1,3] # indices from components, transformers and towers

def plot_group(ax, indices, values, errors, offset, label=None):
    pos = x[indices] + offset
    vals = np.array(values)[indices]
    errs = np.array(errors)[indices]
    ax.bar(pos, vals, width, yerr=errs, capsize=4, label=label)
    ax.ticklabel_format(axis='y', style='sci',scilimits=(0,0))  

# Supply
plot_group(ax1, left_groups, supply_vals, supply_err, -width, "Supply")
plot_group(ax2, right_groups, supply_vals, supply_err, -width)

# Demand AC
plot_group(ax1, left_groups, ac_vals, ac_err, 0, "Demand AC")
plot_group(ax2, right_groups, ac_vals, ac_err, 0)

# Demand MT
plot_group(ax1, left_groups, mt_vals, mt_err, width, "Demand MT")
plot_group(ax2, right_groups, mt_vals, mt_err, width)

ax1.set_xticks(x)
ax1.set_xticklabels(component_labels)

ax1.set_ylabel("Transmission lines (miles) & Circuit Breakers (Quantity)")
ax2.set_ylabel("Transformers (Quantity) & Towers (Quantity)")
## Add a text box saying towers are unconstrained
# index of Towers group
tower_idx = component_labels.index("Towers")
# x position (center of group)
x_pos = x[tower_idx]+0.5*width
trans = transforms.blended_transform_factory(ax2.transData, ax2.transAxes)
tower_vals = [supply_vals[tower_idx], ac_vals[tower_idx], mt_vals[tower_idx]]

ax2.text(
    x_pos,
    0.3,
    "Unconstrained",
    rotation=45,
    ha="center",
    va="bottom",
    transform=trans,
    fontweight="bold"
)

ax1.legend(loc="upper left")

plt.tight_layout()
plt.show()



print('Add in percent of imports')






#%% Make a figure that has units of gap

# These are the arrays in the order of component_labels
# supply_vals 
# ac_vals
# mt_vals

supply_vals = np.array(supply_vals)
gap_in_ac = supply_vals - ac_vals
gap_in_mt = supply_vals - mt_vals
# Put np.nan for last index, towers
gap_in_ac[-1] = np.nan
gap_in_mt[-1] = np.nan


# create dict
gap_dict = {
    'component_labels': component_labels,
    'gap_in_ac': gap_in_ac,
    'gap_in_mt': gap_in_mt
}

# Move transformers to the end
gap_df = pd.DataFrame(gap_dict)
# Move second row to last
rows = gap_df.values.tolist()
rows = rows[:1] + rows[2:] + [rows[1]]
gap_df = pd.DataFrame(rows, columns=gap_df.columns).reset_index(drop=True)


# Rearrange component labels to match new order
component_labels = list(gap_df['component_labels'])

# bar plot
x = np.arange(len(component_labels))
width = 0.35

# colors
color_ac_gap = ['orange']*(len(component_labels))
color_mt_gap = ['green']*(len(component_labels))

# hatch the second group of bars with blue hatching and color the second y-axis with blue to indicate that transformers should be read from that axis.

fig, ax1 = plt.subplots(figsize=(9,5))  
# Plot gap_in_ac bars
bars_ac = ax1.bar(x - width/2, gap_df['gap_in_ac']/1e3, width, color=color_ac_gap, label='AC Gap')
# Plot gap_in_mt bars
bars_mt = ax1.bar(x + width/2, gap_df['gap_in_mt']/1e3, width, color=color_mt_gap, label='MT Gap')
ax1.set_xticks(x)
ax1.set_xticklabels(component_labels)
ax1.set_ylabel('Gap in Supply \n(Thousands of components or thousands of miles)')
ax1.legend(loc='upper left')

ax1.text(
    x[2],
    0.3,
    "Unconstrained",
    rotation=45,
    ha="center",
    va="bottom",
    transform=trans,
    fontweight="bold"
)

ax2 = ax1.twinx()  # Create a second y-axis
ax2.set_ylabel('Gap in Supply (Number of Transformers)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Highlight transformer gap on secondary axis
ax2.bar(x[-1] - width/2, gap_df['gap_in_ac'].iloc[-1], width, color=color_ac_gap[-1], hatch='//', edgecolor='blue')
ax2.bar(x[-1] + width/2, gap_df['gap_in_mt'].iloc[-1], width, color=color_mt_gap[-1], hatch='//',edgecolor='blue')

plt.tight_layout()
plt.show()




# %%
