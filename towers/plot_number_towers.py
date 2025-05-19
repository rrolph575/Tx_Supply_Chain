import pandas as pd
import matplotlib.pyplot as plt

# Numbers from number_and_materials_towers.py. Can make this automated later. 



## AC Scenario
AC_lines_AC_scenario_data = {
    'Type': ['Tangent', 'Running Angle', 'Non Angled Deadend', 'Angled Deadend'],
    'Count_AC_lines_AC_Scenario': [175.071, 49.104, 12.380, 12.380]
}
df_AC_lines_AC_scenario = pd.DataFrame(AC_lines_AC_scenario_data)


## DC Scenario
# AC
AC_lines_DC_scenario_data = {    
    'Type': ['Tangent', 'Running Angle', 'Non Angled Deadend', 'Angled Deadend'],
    'Count_AC_lines_DC_Scenario': [204.530, 48.710, 12.248, 12.248] 
}
df_AC_lines_DC_scenario = pd.DataFrame(AC_lines_DC_scenario_data)

# DC
DC_lines_DC_scenario_data = {    
    'Type': ['Tangent', 'Running Angle', 'Non Angled Deadend', 'Angled Deadend'],
    'Count_DC_lines_DC_Scenario': []
}

df_DC_lines_DC_scenario = pd.DataFrame(DC_lines_DC_scenario_data)



# Merge all scenarios on the 'Type' column
df_total_towers = df_AC_lines_AC_scenario.merge(
    df_AC_lines_DC_scenario, on='Type').merge(df_DC_lines_DC_scenario, on='Type')

print(df_total_towers)




# Reshape the DataFrame for plotting
df_total_towers_melted = df_total_towers.melt(
    id_vars='Type', 
    var_name='Scenario', 
    value_name='Count'
)

# Create a stacked bar plot
df_pivot = df_total_towers_melted.pivot(index='Type', columns='Scenario', values='Count')
df_pivot.plot(kind='bar', stacked=True, figsize=(10, 6))

# Add labels and title
plt.xlabel('Tower Type', fontsize=14)
plt.ylabel('Count ($\\times 10^3$)', fontsize=14)
plt.title('Total Towers by Scenario', fontsize=16)

# Adjust legend
plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 12})

# Show the plot
plt.tight_layout()
plt.show()