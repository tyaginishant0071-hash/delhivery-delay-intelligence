import pandas as pd

from src.shipment_prep import clean_shipments, build_features

raw= pd.read_csv("data/raw_shipments.csv")
hubs_df= pd.read_csv("data/hubs.csv")

# "clean_shipments" function is called from "src.shipment_prep" to clean the raw data:
clean_data, computed_medians = clean_shipments(raw)

# "build_features" function is called from "src.shipment_prep" to build new feature columns:
clean_data= build_features(clean_data, hubs_df)

# Adding a target column "is_delayed". It is 1 when delivery_hours is strictly greater than promised_hours, otherwise 0:
clean_data['is_delayed']= (clean_data['delivery_hours'] > clean_data['promised_hours']).astype(int)

# Using print function to print both "clean data" and "computed medians":
print("Clean Data:\n",clean_data)
print("Medians:", computed_medians)
print("Sum of is_delayed:", clean_data["is_delayed"].sum())
print("Mean of is_delayed:", clean_data["is_delayed"].mean())
print("Values of is_delayed column:", clean_data["is_delayed"].unique())

# Save the final modelling table:
clean_data.to_csv('data/modelling_table.csv', index= False)
