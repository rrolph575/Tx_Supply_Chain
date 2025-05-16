"""
Unless otherwise specified, costs are USD2024/mile
"""

#%% Imports
import os
import pandas as pd
import matplotlib.pyplot as plt


projpath = os.path.dirname(os.getcwd())
datapath  = os.path.join(projpath,'towers','data_conductor_lengths')
figpath = os.path.join(projpath,'plots')

#%% Constants
kv_values = [230, 345, 500, 765]
circuit = 'double'

#%% Import conductor lengths 
#use_avg = False # Don't use average number of cables per bundle.   These files have been generated from find_cable_lengths...py
#hvac_data = pd.read_pickle(datapath+'HVAC_Scenario_HVAC_lines' + str(use_avg) + '_avg_num_bundles.pkl')
#mthvdc_dc_data = pd.read_pickle(datapath+'MTHVDC_Scenario_HVDC_lines' + str(use_avg) + '_avg_num_bundles.pkl')
#mthvdc_ac_data = pd.read_pickle(datapath+'MTHVDC_Scenario_HVAC_lines' + str(use_avg) + '_avg_num_bundles.pkl')
## Might replace above lines with Ram's numbers


#%% Functions

# Inputs include assumptions about percentage of each type of structure (e.g. percent_Tangent = 0.7 assumes 70% of all structures are tangent structures)
def get_number_structures(
        AC_or_DC_conductor='AC',
        AC_or_DC_scenario='AC',
        percent_Tangent = 0.7,
        percent_RunningAngle = 0.1,
        percent_NonAngledDeadend = 0.1,
        percent_AngledDeadend = 0.1
    ):

    percent_values = {
        'percent_Tangent': percent_Tangent,
        'percent_RunningAngle': percent_RunningAngle,
        'percent_NonAngledDeadend': percent_NonAngledDeadend,
        'percent_AngledDeadend': percent_AngledDeadend
    }

    print(AC_or_DC_conductor)

    print(f'tower_{AC_or_DC_conductor}_conductor_{circuit}_circuit.csv')


    df_structs = pd.read_csv(os.path.join(datapath,f'tower_{AC_or_DC_conductor}_conductor_{circuit}_circuit.csv'), header=None)
    df_structs.columns = ['kv'] + df_structs.iloc[0, 1:].astype(float).astype(int).tolist()
    df_structs = df_structs.iloc[1:,:]
    df_structs = df_structs.set_index('kv').T

    ## Read in how many miles of each type of conductor
    miles_of_conductor = pd.read_csv(
        os.path.join(datapath, f'miles_conductor_{AC_or_DC_scenario}_scenario_{AC_or_DC_conductor}_lines.csv')
    ).squeeze()
    print(datapath, f'miles_conductor_{AC_or_DC_scenario}_scenario_{AC_or_DC_conductor}_lines.csv')

    structure_types = ['Tangent', 'RunningAngle', 'NonAngledDeadend', 'AngledDeadend']
    df_number_of_structures = pd.DataFrame(index=miles_of_conductor['kV'],columns=structure_types)

    # Find number of structures by type
    for struct in structure_types:
        for kv in df_structs.index:
            df_number_of_structures.loc[kv,struct] = df_structs.loc[kv, f'{struct}_structures_per_mile'] * miles_of_conductor[miles_of_conductor['kV'] == kv]['miles'].iloc[0] * percent_values[f'percent_{struct}']

    # !!!!!!! Can add here similar to the for loop above, something like..
    #for struct in structure_types:
    #    for kv in df_structs.index:
    #        df_material_for_structures.loc[kv,struct] = df_materials.loc[struct, f'{struct}_kg_al_per_struct'] * df_number_of_structures.loc[kv,struct]
    
    return df_number_of_structures


### !! Can add in another def here to calculate the total weight of al and stl using the df_number_of_structures above






#%%### Procedure
# if __name__ == '__main__':  ## commented because of vscode settings. change later.

# Constants for other parameters
AC_or_DC_conductor='DC'
AC_or_DC_scenario='DC'
percent_Tangent = 0.7
percent_RunningAngle = 0.1
percent_NonAngledDeadend = 0.1
percent_AngledDeadend = 0.1


df_number_struct = get_number_structures(
    AC_or_DC_conductor=AC_or_DC_conductor,
    AC_or_DC_scenario=AC_or_DC_scenario,
    percent_Tangent = percent_Tangent,
    percent_RunningAngle = percent_RunningAngle,
    percent_NonAngledDeadend = percent_NonAngledDeadend,
    percent_AngledDeadend = percent_AngledDeadend
)



#%% Create a stacked bar plot
df_number_thousands_of_structures = df_number_struct/1e3
ax = df_number_thousands_of_structures.plot(kind='bar', stacked=True, figsize=(10, 6))

# Add labels and title
#plt.rcParams.update({'font.size': 18})
plt.xlabel('kV',fontsize=24)
plt.ylabel('Number of Structures (x1000)',fontsize=18)
plt.title(f'{AC_or_DC_scenario} scenario, {AC_or_DC_conductor} conductor, {circuit} circuit',fontsize=18)
ax.set_ylim([0,100])
plt.legend(title='Structure Type', bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 14})

# Show the plot
plt.tight_layout()
plt.show()

# %%  Calculate the amount of steel and aluminum required for each voltage



