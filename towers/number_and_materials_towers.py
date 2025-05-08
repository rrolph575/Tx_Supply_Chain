"""
Unless otherwise specified, costs are USD2024/mile
"""

#%% Imports
import os
import pandas as pd
import matplotlib.pyplot as plt


projpath = os.path.dirname(os.getcwd())
datapath  = os.path.join(projpath,'outputs/')
figpath = os.path.join(projpath,'plots/')

#%% Constants
KVS = [69, 115, 138, 161, 230, 345, 500, 765]

#%% Import conductor lengths 
#use_avg = False # Don't use average number of cables per bundle.   These files have been generated from find_cable_lengths...py
#hvac_data = pd.read_pickle(datapath+'HVAC_Scenario_HVAC_lines' + str(use_avg) + '_avg_num_bundles.pkl')
#mthvdc_dc_data = pd.read_pickle(datapath+'MTHVDC_Scenario_HVDC_lines' + str(use_avg) + '_avg_num_bundles.pkl')
#mthvdc_ac_data = pd.read_pickle(datapath+'MTHVDC_Scenario_HVAC_lines' + str(use_avg) + '_avg_num_bundles.pkl')



#%% Functions

def get_number_structures(
        kv=500,
        AC_or_DC_conductor='AC',
        AC_or_DC_scenario='AC',
        percent_Tangent = 0.7,
        percent_RunningAngle = 0.1,
        percent_NonAngledDeadend = 0.1,
        percent_AngledDeadend = 0.1
    ):

    circuit_number = [
        'single',
        'double',
    ]
    dictin_structure = {
        bus: pd.read_csv(
            os.path.join(
                datapath,
                f'tower_{AC_or_DC_conductor}_conductor_{circuit}_circuit.csv'
            ),
            index_col=0,
        )
        for circuit in circuit_number
    }
    struct_specs = pd.concat(dictin_structure, axis=0)[str(kv)].unstack(0)

    ## Read in how many miles of each type of conductor
    miles_of_conductor = pd.read_csv(
        os.path.join(datapath, f'miles_conductor_{AC_or_DC_scenario}_scenario_{AC_or_DC_conductor}_lines.csv')
    ).squeeze()

    structure_types = ['Tangent', 'RunningAngle', 'NonAngledDeadend', 'AngledDeadend']
    df_number_of_structures = pd.DataFrame(columns=structure_types)
    
    for struct in structure_types:
        number_of_structures[struct] = struct_specs.loc[f'{struct}_structures_per_mile'] * miles_of_conductor[miles_of_conductor['kV'] == kv]['miles'] * f'percent_{struct}'


    return df_number_of_structures


### !! Can add in another def here to calculate the total weight of al and stl using the df_number_of_structures above

#%%### Procedure
# if __name__ == '__main__':  ## commented because of vscode settings. change later.
#%% Imports
import matplotlib.pyplot as plt
import site

# %%
### Call functions
# List of kv values to loop through
kv_values = [69,115,138,161,230,345,500,765] 


# Constants for other parameters
substation_option = 'new'  # 'upgrade' for 1 or 2 positions, 'new', for 4 or 6 positions
num_positions = 4
landtype = 'light_veg' # light_veg, forest, wetland
# 'ring', 'breaker_and_half', 'double_breaker'  ... note the code produces data for all 3
# bus types given the constants above... can be reworked to be more efficient if desired
bus_type_for_plot = 'breaker_and_half'
fig_filename = os.path.join(
    figpath, f"{substation_option}_{bus_type_for_plot}_{num_positions}positions_{landtype}.png"
)


# List to collect plot data for all kv values
all_plot_data = []

# Loop through each kv value
for kv in kv_values:
    # Access road and terrain costs
    df_access_and_terrain = get_land_terrain_cost(
        kv=kv,
        substation_option=substation_option,
        num_positions=num_positions,
        landtype=landtype
    )

    # Cable component costs
    cable_costs_df = get_cable_costs(
        kv=kv,
        substation_option=substation_option,
        num_positions=num_positions
    )

    # Other component costs
    components = [
        ('circuit_breaker_unit_costs.csv', 'circuit_breaker'),
        ('disconnect_switch_unit_costs.csv', 'disconnect_switches'),
        ('bus_unit_costs.csv', 'bus_support'),
        ('voltage_transformer_unit_costs.csv', 'voltage_transformers'),
        ('control_enclosure_unit_costs.csv', 'control_enclosure'),
        ('relay_panel_costs.csv', 'relay_panel'),
        ('deadend_angled_structure_costs.csv', 'deadend_struct')
    ]

    # Dictionary to store the costs
    component_costs_dict = {}

    # Loop through each component to calculate costs
    for csv_file, component_name in components:
        component_costs_dict[component_name] = component_costs(
            kv=kv,
            substation_option=substation_option,
            num_positions=num_positions,
            csv=csv_file,
            component_name=component_name
        )

    # Transforming the data into a DataFrame
    rows = []
    for cost_type, series in component_costs_dict.items():
        for bus_type, cost in series.items():
            rows.append({'bus_type': bus_type, 'cost': cost, 'cost_type': cost_type})

    component_costs_df = pd.DataFrame(rows)

    ### Combine dataframes
    df = pd.concat([df_access_and_terrain, cable_costs_df, component_costs_df])

    ### Add common costs
    pivot_df = df.pivot_table(
        index='bus_type', columns='cost_type', values='cost', aggfunc='sum').fillna(0)
    total_cost_by_bustype = pivot_df.copy()
    total_cost_by_bustype = total_cost_by_bustype.sum(axis=1)
    total_mult = get_common_cost_mult()
    pivot_df['softcost'] = total_cost_by_bustype * (total_mult - 1)

    ### Prepare for plotting
    plot_df = pivot_df.copy()
    plot_df = plot_df / 1e6  # Convert to millions
    plot_df['kv'] = kv  # Add kv to the plot DataFrame

    # Append the current plot data to the list
    all_plot_data.append(plot_df)



## Get validation costs for each kv
validation_costs = []
for kv in kv_values:
    MISO_validation_costs = get_validation_cost(
        kv=kv,
        substation_option=substation_option,
        num_positions=num_positions,
        bus_type_for_plot=bus_type_for_plot
    )
    validation_costs.append(MISO_validation_costs)

validation_to_plot = pd.DataFrame(validation_costs)


# Concatenate all plot data into a single DataFrame
final_plot_df = pd.concat(all_plot_data)

# Run and save plot
stacked_bar(
    final_plot_df=final_plot_df,
    MISO_validation_costs=validation_to_plot[bus_type_for_plot],
    fig_filename=fig_filename,
)


#### !!! NOTES BELOW TO BE DISCUSSED !!! #####

#  Table 2.3 -6 (current transformer)
# !! Verify with MISO these are not included in costs

#  Table 2.3 -8 (Grid supporting devices unit costs)
# !! Verify with MISO these are not included in validation costs

#  Table 2.3 -7 (Power transformer)  # !! Not all substations need power transformers
#     If we use Table 3.1 - 5 to get number of MVAs,
#  .. then the costs are much higher than the validation values in Table 4.2-1.
#  .. For example, Table 3.1 -5  gives 2598 MVA for 500kV/500kV.
#  .. when multiplied by 13784 $/MVA from Table 2.3 -7,
#  .. that is already 35 $M, order of magnitude higher than validation value in Table 4.1-1.
# %%
