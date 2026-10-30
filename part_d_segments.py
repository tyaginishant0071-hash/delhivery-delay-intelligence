import pandas as pd
import os

os.makedirs('outputs', exist_ok=True)

# Loading and Reading the input CSV file "data/modelling_table.csv" into a pandas DataFrame:
modelling_table_df= pd.read_csv('data/modelling_table.csv')

# Creating a "hub_profile" for grouping "hub_id" and computing aggregations:
hub_profile= modelling_table_df.groupby('hub_id').agg(
    shipments= ('shipment_id', 'count'),
    mean_delivery_hours= ('delivery_hours', 'mean'),
    delay_rate= ('is_delayed', 'mean'),
    mean_distance_km= ('distance_km', 'mean'),
    mean_n_stops= ('n_stops', 'mean')
)

# Sorting the "hub_profile" and rounding the "hub_profile_df":
hub_profile_df= hub_profile.sort_values(by= 'hub_id', ascending= True).reset_index(drop= True)
hub_profile_df= hub_profile_df.round({
    'shipments': 0,
    'mean_delivery_hours': 4,
    'delay_rate': 4,
    'mean_distance_km': 4,
    'mean_n_stops': 4
})


# Save the file:
hub_profile_df.to_csv('outputs/hub_profile.csv', index=False)
print("Hub profile saved to 'outputs/hub_profile.csv'!!\n")