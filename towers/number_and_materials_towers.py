"""
Unless otherwise specified, costs are USD2024/mile
"""

#%% Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import math


projpath = os.path.dirname(os.getcwd())
datapath  = os.path.join(projpath,'towers','data')
figpath = os.path.join(projpath,'plots')

#%% Constants
kv_values = [230, 345, 500, 765]
#circuit = 'double'

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
        percent_Tangent = 1,
        percent_RunningAngle = 1,
        percent_NonAngledDeadend = 1,
        percent_AngledDeadend = 1
    ):

    percent_values = {
        'percent_Tangent': percent_Tangent,
        'percent_RunningAngle': percent_RunningAngle,
        'percent_NonAngledDeadend': percent_NonAngledDeadend,
        'percent_AngledDeadend': percent_AngledDeadend
    }

    print(AC_or_DC_conductor)

    #print(f'tower_{AC_or_DC_conductor}_conductor_{circuit}_circuit.csv')

    circuit = 'double' # overwritten if AC line is 500 kV and higher
    if AC_or_DC_conductor == 'DC':
        circuit = 'single'
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
            # Use single circuit data if AC line is 500 kV and higher. 
            if (AC_or_DC_conductor == 'AC' and kv >= 500) or AC_or_DC_conductor == 'DC':
                circuit = 'single'
                df_structs = pd.read_csv(os.path.join(datapath,f'tower_{AC_or_DC_conductor}_conductor_{circuit}_circuit.csv'), header=None)
                df_structs.columns = ['kv'] + df_structs.iloc[0, 1:].astype(float).astype(int).tolist()
                df_structs = df_structs.iloc[1:,:]
                df_structs = df_structs.set_index('kv').T
            df_number_of_structures.loc[kv,struct] = df_structs.loc[kv, f'{struct}_structures_per_mile'] * miles_of_conductor[miles_of_conductor['kV'] == kv]['miles'].iloc[0] * percent_values[f'percent_{struct}']

    return df_number_of_structures


def calc_steel_weights(df_number_struct, AC_or_DC_conductor, AC_or_DC_scenario):
    
    tower_types = ['Tangent', 'RunningAngle', 'NonAngledDeadend', 'AngledDeadend']
    df_steel = pd.DataFrame(index=df_number_struct.index, columns=tower_types)

    # Calculate steel weights for each tower type
    for struct in tower_types:
        if AC_or_DC_conductor == 'AC':
            tp = 'steelpole'
            circuit_type = 'double' # overwritten if AC line is 500 kV and higher
        else:
            tp = 'steeltower'
            circuit_type = 'single'
        filename = os.path.join(datapath, f'structure_{AC_or_DC_conductor.lower()}_{tp}_{circuit_type}circuit_{struct.lower()}.csv')
        df_materials = pd.read_csv(filename, header=None)
        df_materials.columns = ['kv'] + df_materials.iloc[0, 1:].astype(float).astype(int).tolist()
        df_materials = df_materials.iloc[1:,:]
        df_materials = df_materials.set_index('kv').T
        for kv in df_number_struct.index:
            # Assume single circuit if AC line is 500 kV and higher
            if (AC_or_DC_conductor == 'AC' and kv >= 500) or AC_or_DC_conductor == 'DC':
                circuit_type = 'single'
                filename = os.path.join(datapath, f'structure_{AC_or_DC_conductor.lower()}_{tp}_{circuit_type}circuit_{struct.lower()}.csv')
                df_materials = pd.read_csv(filename, header=None)
                df_materials.columns = ['kv'] + df_materials.iloc[0, 1:].astype(float).astype(int).tolist()
                df_materials = df_materials.iloc[1:,:]
                df_materials = df_materials.set_index('kv').T
            # Assuming df_materials contains steel weight per structure for each type
            if AC_or_DC_conductor == 'DC' and kv == 500:
                df_steel.loc[kv,struct] = (
                df_materials.loc[kv,'steelweight_lbs'] * math.ceil(df_number_struct.loc[kv, struct])
            )
            else:
                df_steel.loc[kv, struct] = 0

    return df_steel


#%%### Procedure
# if __name__ == '__main__':  ## commented because of vscode settings. 


# Constants for other parameters
AC_or_DC_conductor='DC'
AC_or_DC_scenario='DC'
percent_Tangent = 1
percent_RunningAngle = 1
percent_NonAngledDeadend = 1
percent_AngledDeadend = 1


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
plt.ylabel('Number of Structures ($\\times 10^3$)',fontsize=18)
plt.title(f'{AC_or_DC_scenario} scenario, {AC_or_DC_conductor} conductor', fontsize=18)
ax.set_ylim([0,180])
plt.legend(title='Structure Type', title_fontsize = 16, bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 14})
ax.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
plt.show()



# %%  Calculate the amount of steel and aluminum required for each voltage
df_steel = calc_steel_weights(df_number_struct, AC_or_DC_conductor, AC_or_DC_scenario)
df_steel = df_steel/2000/1e6 # convert to millions of US tons
# Plot 
ax = df_steel.plot(kind='bar', stacked=True, figsize=(10, 6))

# Add labels and title
#plt.rcParams.update({'font.size': 18})
plt.xlabel('kV',fontsize=18)
plt.ylabel('Amt of steel (Millions of US tons)',fontsize=18)
plt.title(f'{AC_or_DC_scenario} scenario, {AC_or_DC_conductor} conductor, steel weights',fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=20)
#ax.set_ylim([0,180])
plt.legend(title='Structure Type', title_fontsize =16, bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 14})
plt.tight_layout()
plt.show()



