import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



basepath = "C:/Users/rrolph/OneDrive - NREL/Projects/FY25/Transmission_Supply_Chain/"
datapath = basepath + "Data/R02_Transmission_Expansion/"
plotpath = basepath + "plots_new/"
scenario_name = 'S03'  # S01 is only HVAC and S03 has both HVDC and HVAC.
# Assumption on MVA capacity per transformer
#MVA_per_transformer = 400 # MVA per HVAC transformer. But right now we are using Hico america assumptions below.
# Assumption on # converter-transformers per HVDC line
number_converter_transformers_per_line = 24 
# Switch for calculating HVAC transformers within the MTHVDC scenario
calc_AC_transformers_in_MTHVDC_scenario = True
# This switch is here to highight that values are hardcoded at the bottom after running each scenario.  This can be improved later if necessary. 
plot_only = True


if scenario_name == 'S01':
    filenames = ['R02_S01_Transmission_Expansion_EI-1' , 'R02_S01_Transmission_Expansion_ERCOT-1', 'R02_S01_Transmission_Expansion_WECC-1']

if scenario_name == 'S03':
    filenames = ['R02_S03_Transmission_Expansion_CONUS-2']


dfs = []
# Combine df from the S01 scenarios
for file in filenames:
    df = pd.read_csv(datapath + file + '.csv')
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)

def get_number_transformers(df, scenario_name):       
    # Filter the data to transformers only (where the voltage steps up or down)
    df['Vn [kV]'] = df['Vn [kV]'].astype(str)
    df = df[(df['Add [Y/N]'] == 'Y')]  # !!! Added this line already to include only 'Y' column !!!!
    if scenario_name == 'S01':
        filtered_df = np.nan # number of lines with converter-transformers
        filtered_df_HVAC_transformers = df[df['Type'] == '2TF'] # 6 integers indicates a step up/down
        # Calculate how many transformers needed to match capacity from data
        #filtered_df_HVAC_transformers['Number_transformers_needed'] = np.ceil(filtered_df_HVAC_transformers['Rate1[MVA]']/MVA_per_transformer)
    if scenario_name == 'S03': 
        filtered_df = df[(df['Add [Y/N]'] == 'Y') & (df['Type'] == 'HVDC')] # number of lines with converter-transformers
        #filtered_df_HVAC_transformers = df[df['Vn [kV]'].str.match(r'^\d{6}$')] # 6 integers indicates a step up/down
        filtered_df_HVAC_transformers = df[df['Type'] == '2TF']
        #filtered_df_HVAC_transformers['Number_transformers_needed'] = np.ceil(filtered_df_HVAC_transformers['Rate1[MVA]']/MVA_per_transformer)
    return filtered_df, filtered_df_HVAC_transformers



#filtered_df, filtered_df_HVAC_transformers = get_number_transformers(df_all, MVA_per_transformer, scenario_name)
filtered_df, filtered_df_HVAC_transformers = get_number_transformers(df_all, scenario_name)


### Create a dataframe with the following assumptions from HICOamerica, since 2000/2400 MVA transformers do not exist
# Data
data = {
    "Voltage": ["345/138", "345/230", "500/230*", "500/345**", "765/500"],
    "Phase": [3, 3, 1, 1, 1],
    "Capacity (MVA)": [700, 700, 667, 667, 667],
    "Required units to achieve 2000 MVA": [3, 3, 9, 9, 9]
}
# Create DataFrame
df_transformer_assumptions = pd.DataFrame(data)
# Standardizing voltage names
df_transformer_assumptions['Voltage_cleaned'] = df_transformer_assumptions['Voltage'].replace({
    "500/345**": "500-345", 
    "500/230*": "500-230", 
    "345/138": "345-138", 
    "345/230": "345-230", 
    "765/500": "765-500"
})



#def plot_number_transformers(filtered_df, MVA_per_transformer, scenario_name):

# Plot 
if scenario_name == 'S01' or calc_AC_transformers_in_MTHVDC_scenario == True:

    ### Plot the number of transformers for each unique kV value

    # Count the occurrences of each unique 'Vn [kV]' value
    vn_counts = filtered_df_HVAC_transformers['Vn [kV]'].value_counts()
    vn_counts.index = vn_counts.index.str.replace('/', '', regex=False)
    vn_counts = vn_counts.groupby(vn_counts.index).sum()
    vn_counts.index = vn_counts.index.str[:3] + '-' + vn_counts.index.str[3:]
    
    """# Number of rows for HVAC transformers
    fig, ax = plt.subplots()
    ax.bar(vn_counts.index, vn_counts.values, color='skyblue', edgecolor='black')
    ax.set_title('# LPTs for ' + scenario_name)
    ax.set_xlabel('Voltage')
    ax.set_ylabel('Number of rows in datafile that correspond to transformer locations (not number of transformers)')
    #ax.set_ylim([0,80])
    plt.tight_layout()
    plt.savefig(plotpath + scenario_name + '.png', bbox_inches='tight')
    plt.show() """


    ### Plot the number of transformers needed for each MVA rating, assuming a certain MVA per transformer.
    # Find unique MVA values
    # Group by 'Rate1[MVA]' and sum 'Number_transformers_needed'
    #grouped = filtered_df_HVAC_transformers.groupby('Rate1[MVA]')['Number_transformers_needed'].sum()

    # Plotting
    """ plt.figure(figsize=(8, 6))
    grouped.plot(kind='bar', color='skyblue')
    plt.title('Total Number of Transformers Needed for HVAC assuming ' + str(MVA_per_transformer) + 'MVA per transformer')
    plt.xlabel('Rate1 [MVA]')
    plt.ylabel('Total Number of Transformers')
    plt.xticks(rotation=0)
    plt.savefig(plotpath + scenario_name + '_LPTs_needed_assuming' + str(MVA_per_transformer) + 'MVA_per_transformer.png', bbox_inches='tight')
    plt.show() """


    ## Multiply the number of transformers per voltage by the number of transformers required for each voltage according to MVA capacity assumptions of HICOamerica
    # Merge the DataFrames on the cleaned voltage column
    df_merged = pd.merge(df_transformer_assumptions, vn_counts, left_on='Voltage_cleaned', right_index=True)

    # Create the new column by multiplying required units by the counts
    df_merged['Total required units'] = df_merged['Required units to achieve 2000 MVA'] * df_merged['count']

    # Display the result
    df_merged[['Voltage', 'Required units to achieve 2000 MVA', 'count', 'Total required units']]

    # Plotting the 'Total required units' for each voltage
    plt.figure(figsize=(10, 6))
    plt.bar(df_merged['Voltage'], df_merged['Total required units'], color='skyblue')

    # Adding labels and title
    plt.xlabel('Voltage (kV)', fontsize=12)
    plt.ylabel('Total Required Units', fontsize=12)
    plt.title('Total Required HVAC Transformer Units for Each Voltage \n using HicoAmerica assumptions on capacity', fontsize=14)

    # Rotating x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Display the plot
    plt.tight_layout()
    plt.show()

    print('The total number of HVAC transformers is ' + str(df_merged['Total required units'].sum()))
    


if scenario_name == 'S03':
    # Separate the data based on Rate1[MVA] 
    df_4000 = filtered_df[filtered_df['Rate1[MVA]'] == 4000]
    df_2000 = filtered_df[filtered_df['Rate1[MVA]'] == 2000]
    
    # Count occurrences of each voltage for each group
    vn_counts_4000 = df_4000['Vn [kV]'].value_counts()
    vn_counts_2000 = df_2000['Vn [kV]'].value_counts()
    
    # Shorten the voltage labels to first 3 characters (e.g., "500", "345")
    vn_counts_4000.index = vn_counts_4000.index.str[:3]
    vn_counts_2000.index = vn_counts_2000.index.str[:3]
    
    ######## Plot the number of lines ###########
    fig, ax = plt.subplots()

    # Plot the bars for Rate1[MVA] = 10000 and Rate1[MVA] = 5000
    #ax.bar(vn_counts_4000.index, vn_counts_4000.values, color='skyblue', edgecolor='black', label='Rate1 = 4000 MVA')
    #ax.bar(vn_counts_2000.index, vn_counts_2000.values, color='orange', edgecolor='black', label='Rate1 = 2000 MVA')

    # Combine the datasets because we decided we dont need to plot them separately
    vn_counts_2000 = pd.Series(vn_counts_2000)
    vn_counts_4000 = pd.Series(vn_counts_4000)
    combined_counts = vn_counts_2000.add(vn_counts_4000, fill_value=0)  

    # Plot combined counts
    ax.bar(combined_counts.index, combined_counts.values, color='skyblue')

    # Set plot title and labels
    ax.set_title('# HVDC lines for ' + scenario_name)
    ax.set_xlabel('Voltage (kV)')
    ax.set_ylabel('Number of HVDC lines')

    # Set the y-axis to have integer ticks
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Add a legend
    #ax.legend(title="Rate1[MVA]")

    # Optional: Set y-axis limit if needed (uncomment line below)
    # ax.set_ylim([0, 80])

    # Adjust layout and save plot as image
    plt.tight_layout()
    #plt.savefig(plotpath + 'HVDC_lines.png', bbox_inches='tight')
    
    # Show the plot
    plt.show()



    ####### Plot the number of converter-transformers ########
    # Create the bar plot
    fig, ax = plt.subplots()

    # Plot the bars for Rate1
    #ax.bar(vn_counts_4000.index, vn_counts_4000.values*number_converter_transformers_per_line, color='skyblue', edgecolor='black', label='Rate1 = 4000 MVA')
    #ax.bar(vn_counts_2000.index, vn_counts_2000.values*number_converter_transformers_per_line, color='orange', edgecolor='black', label='Rate1 = 2000 MVA')

    number_converter_transformers = combined_counts.values*number_converter_transformers_per_line

    ax.bar(combined_counts.index, number_converter_transformers, color='#FF8C00', edgecolor=None)

    # Set plot title and labels
    #ax.set_title('# Converter-Transformers assuming ' + str(number_converter_transformers_per_line) + ' per line') #, ' + scenario_name)
    ax.set_xlabel('Voltage (kV) on HVAC side', labelpad=10, fontsize=14)
    ax.set_ylabel('Number of converter-transformers', fontsize=14)

    # Set the y-axis to have integer ticks
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(rotation=90)
    # Add a legend
    #ax.legend(title="Rate1[MVA]")

    # Optional: Set y-axis limit if needed (uncomment line below)
    # ax.set_ylim([0, 80])

    # Adjust layout and save plot as image
    plt.tight_layout()
    #plt.savefig(plotpath + 'HVDC_converter_transformers.png', bbox_inches='tight')

    # Print total number converter transformers
    print('Total number of converter-transformers: ' + str(sum(number_converter_transformers)))






### This is a plot of data taken from above runs, can make this automated later if necessary..
if plot_only == 'True': 
    # HVAC scenario
    number_hvac_transformers_for_scenario1 = 1914 # df_merged['Total required units'].sum() 
    # MTHVDC scenario
    number_converter_transformers_for_scenario2 =  960 # number_converter_transformers
    number_hvac_transformers_for_scenario2 = 474 # df_merged['Total required units'].sum() 
    labels = ['HVAC Scenario', 'MTHVDC Scenario']

    # Plotting the data
    fig, ax = plt.subplots()

    ax.bar(labels, [number_hvac_transformers_for_scenario1, number_hvac_transformers_for_scenario2], label='New Transformers', color='orange')
    ax.bar(labels, [0, number_converter_transformers_for_scenario2], bottom=[number_hvac_transformers_for_scenario1, number_hvac_transformers_for_scenario2], label='Converter-Transformers', color='blue')

    # Adding labels and title
    ax.set_ylabel('Number of Transformers')
    ax.set_title('Transformers for HVAC and MTHVDC')
    ax.set_ylim([0,2000])
    ax.legend()

    # Show plot
    plt.show()
