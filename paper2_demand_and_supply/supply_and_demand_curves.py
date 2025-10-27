
import pandas as pd
import matplotlib.pyplot as plt


# Read in data
df = pd.read_excel('Grid_RING_inputs_v2.xlsx', sheet_name='MTGrid_2024MidCase',skiprows=6)



# Filter
selected_material = 'copper' # copper, aluminum
df_filtered = df[df['Refined Material'] == selected_material]



## Plot material requirements over time
# Extract only the year columns (numeric column names)
year_cols = [col for col in df.columns if str(col).isdigit()]
# Set up the figure
plt.rcParams.update({
    'font.size': 18,           # base font size
    'axes.titlesize': 18,      # title font size
    'axes.labelsize': 18,      # x and y label font size
    'xtick.labelsize': 18,     # x-axis tick font size
    'ytick.labelsize': 18,     # y-axis tick font size
    'legend.fontsize': 15,     # legend font size
    'legend.title_fontsize': 18,
})
plt.figure(figsize=(12, 8))

# Loop through each Term and plot its row values
for term, row in df_filtered.groupby('Term'):
    # Take the first row for this Term (if multiple, you can handle differently)
    if selected_material=='copper':
        y_values = row[year_cols].values[0]/1e6 # convert to millions
    if selected_material=='aluminum':
        y_values = row[year_cols].values[0]/1e6 # convert to millions
    plt.plot(year_cols, y_values, marker='o', label=term)

# Labels and title
#plt.xlabel('Year')
if selected_material=='copper':
    plt.ylabel('Copper Required (millions of metric tons)')
if selected_material=='aluminum':
    plt.ylabel('Aluminum Required (millions of metric tons)')

plt.xticks(rotation=90)
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
