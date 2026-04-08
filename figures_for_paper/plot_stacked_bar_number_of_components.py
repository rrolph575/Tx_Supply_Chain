import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Set global font sizes
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12
})

# Define a color map for stacked bars (in the same order as df_pivot.columns)
colors = {
    "New LPT": "#1f77b4",  # default Matplotlib blue
    "LPT from NTP": "#ff7f0e",  # orange (will be used for Towers now)
    "Converter-transformers": "#2ca02c",  # green
    "Generator Step-up transformers": "#d62728",  # red
    "Circuit breakers - HVAC": "#5DADE2",  # light blue
    "Circuit breakers - MT HVDC": "#9467bd"  # purple
}


# Data
data = {
    "Component": [
        "Conductors (miles)",
        "New LPT",
        "LPT from NTP",
        "Converter-transformers",
        "Generator Step-up transformers",
        "Circuit breakers - HVAC",
        "Circuit breakers - MT HVDC",
        "Towers"
    ],
    "AC": [50350, 1700, 1914, 0, 11480, 13900, 0, 248935],
    "MT": [68440, 1700, 474, 960, 10579, 9270, 3000, 362740]
}

df = pd.DataFrame(data)

# ----------------------
# Figure 1: Components (excluding Conductors)
# ----------------------
df_towers = df[df["Component"] == "Towers"].set_index("Component")
df_others = df[(df["Component"] != "Towers") & (df["Component"] != "Conductors (miles)")]

# Melt for stacked bar
df_melted = df_others.melt(id_vars="Component", var_name="Type", value_name="Value")
df_pivot = df_melted.pivot(index="Type", columns="Component", values="Value")

# Plot stacked bars
fig1, ax1 = plt.subplots(figsize=(12, 6))
df_pivot.plot(kind="bar", stacked=True, ax=ax1, width=0.4, position=0, color=[colors[col] for col in df_pivot.columns])
# After plotting df_pivot
for container in ax1.containers:
    for bar in container:
        height = bar.get_height()
        if height > 0:
            ax1.text(
                bar.get_x() + bar.get_width()/2,  # center of the bar
                bar.get_y() + height/2,           # middle of the stacked segment
                f'{int(height)}',
                ha='center', va='center',
                color='white',
                fontsize=12
            )


# Towers on secondary axis
ax2 = ax1.twinx()
df_towers.T.plot(kind="bar", ax=ax2, width=0.4, position=1, color=["#4B0082"], legend=False)
for container in ax2.containers:
    for bar in container:
        height = bar.get_height()
        bar.set_hatch('//')
        if height > 0:
            ax2.text(
                bar.get_x() + bar.get_width()/2,  # center of the bar
                bar.get_y() + height/2,           # middle of the bar
                f'{int(height)}',
                ha='center', va='center',
                color='white', fontsize=12
            )

# Titles and labels
#ax1.set_title("AC vs MT Components (Excluding Conductors)")
ax1.set_ylabel("Count")
ax2.set_ylabel("Towers", labelpad=20)
ax2.tick_params(axis='y', colors="#4B0082")   # y-axis ticks
ax2.yaxis.label.set_color("#4B0082")          # y-axis label
ax2.spines['right'].set_color("#4B0082")      # right spine
ax1.set_xlabel("")
ax1.set_xticklabels(df_pivot.index, rotation=0)
ax1.set_xlim(-0.5, len(df_pivot.index)-0.5)

# Combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, bbox_to_anchor=(1.2, 1), loc="upper left")

# Scientific notation
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

# Swap y-axes
ax1.yaxis.set_label_position("right")
ax1.yaxis.tick_right()
ax1.spines['right'].set_position(('outward', 0))
ax1.spines['left'].set_visible(False)

ax2.yaxis.set_label_position("left")
ax2.yaxis.tick_left()
ax2.spines['left'].set_position(('outward', 0))
ax2.spines['right'].set_visible(False)



plt.tight_layout()

# ----------------------
# Figure 2: Conductors only
# ----------------------
import pandas as pd
import matplotlib.pyplot as plt

# Data for conductors
data_conductors = {
    "Type": ["AC", "MT"],
    "AC Conductor": [50350, 49550],
    "HVDC Conductor": [0, 18890]  # AC has no HVDC
}

df_conductors = pd.DataFrame(data_conductors)
df_conductors.set_index("Type", inplace=True)

# Convert to thousands of miles
df_conductors = df_conductors / 1000

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
df_conductors.plot(kind="bar", stacked=True, ax=ax, color=["#FF7F0E", "#5DADE2"])  # AC orange, HVDC blue

# Titles and labels
#ax.set_title("AC vs MT Conductors (AC and HVDC)")
ax.set_ylabel("Thousands of Miles")
ax.set_xlabel("")
ax.set_xticklabels(df_conductors.index, rotation=0)

# Remove scientific notation
ax.ticklabel_format(style='plain', axis='y')

for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_y() + height/2,
                f'{height:.1f}',
                ha='center', va='center',
                color='white',
                fontsize=12
            )


plt.tight_layout()
plt.show()
