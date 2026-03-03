


# Plots the domstically produced and imported LPT in the US from the US Dept of Commerce data
# Reference: Figs. VIII-36 and VIII-41 from  https://media.bis.gov/media/documents/redacted-goes-report-updated-10-26-21.pdf


import pandas as pd
import matplotlib.pyplot as plt


years = list(range(2015, 2020))
#years.append("2019 YTD (Jun)") # Commented out because we don't have data for YTD of domestically produced and if they are on the same plot, it looks like 0
#years.append("2020 YTD (Jun)")
domestic_LPTs = [152, 139, 122, 130, 137]
imports_LPTs = [681, 576, 520, 505, 617]


dict_total_LPTs = {
    year: {'domestic': domestic_LPTs, 'imports': imports_LPTs}
    for year, domestic_LPTs, imports_LPTs in zip(years, domestic_LPTs, imports_LPTs)
}


x = range(len(years))
width = 0.35

plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(10,6))
plt.bar([i - width/2 for i in x], domestic_LPTs, width, label='Domestic')
plt.bar([i + width/2 for i in x], imports_LPTs, width, label='Imports')

plt.xticks(x, years)
plt.ylabel('Number of LPTs')
#plt.title('Domestic vs Imported LPTs in US (2015-2019)')
plt.legend()
plt.tight_layout()
plt.show()