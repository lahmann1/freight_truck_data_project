import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get the whole database
df = pd.read_csv('data/FAF5.5.1_State.csv')
print(df.columns)

# Domestic trucks only
df = df[(df['dms_mode'] == 1) & df['trade_type'] == 1]
miles = 1000 * df['tmiles_2017'] / df['tons_2017']
print(np.nansum(miles))
print(np.nansum(1e6 * df['tmiles_2017']))

# Originate CA only
df = df[df['dms_origst'] == 6]

miles = 1000 * df['tmiles_2017'] / df['tons_2017']
print(np.nansum(miles))
print(np.nansum(1e6 * df['tmiles_2017']))

plt.hist(1000 * df['tmiles_2017'] / df['tons_2017'], bins=100, edgecolor='k')
plt.xlabel('Distance (miles)')
plt.ylabel('Trips')
plt.show()

plt.hist(1000 * df['tons_2017'], bins=np.linspace(0, 32, 101), edgecolor='k')
plt.xlabel('Shipment Weight (tons)')
plt.ylabel('Trips')
plt.yscale('log')
plt.show()

plt.hist(df['sctg2'], weights=df['tmiles_2017'], bins=np.arange(0.5, 43.5), edgecolor='k')
plt.xticks(np.arange(1, 43),
           [
               'Live animals/fish',
               'Cereal grains',
               'Other ag prods.',
               'Animal feed',
               'Meat/seafood',
               'Milled grain prods.',
               'Other foodstuffs',
               'Alcoholic beverages',
               'Tobacco prods.',
               'Building stone',
               'Natural sands',
               'Gravel',
               'Nonmetallic minerals',
               'Metallic ores',
               'Coal',
               'Crude petroleum',
               'Gasoline',
               'Fuel oils',
               'Natural gas and other fossil products',
               'Basic chemicals',
               'Pharmaceuticals',
               'Fertilizers',
               'Chemical prods.',
               'Plastics/rubber',
               'Logs',
               'Wood prods.',
               'Newsprint/paper',
               'Paper articles',
               'Printed prods.',
               'Textiles/leather',
               'Nonmetal min. prods.',
               'Base metals',
               'Articles-base metal',
               'Machinery',
               'Electronics',
               'Motorized vehicles',
               'Transport equip.',
               'Precision instruments',
               'Furniture',
               'Misc. mfg. prods.',
               'Waste/scrap',
               'Mixed freight'
           ],
           rotation=90
           )
plt.tight_layout()
plt.ylabel('Million Ton-miles')
plt.show()

plt.hist(1e6 * df['value_2017'], bins=np.linspace(0.0, 10000, 101), density=True)
plt.yscale('log')
plt.ylabel('Number of Shipments')
plt.xlabel('Value (USD)')
plt.show()
