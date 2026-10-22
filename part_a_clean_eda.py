import pandas as pd
import os

from src.shipment_prep import clean_shipments, build_features, FEATURES

raw= pd.read_csv("data/raw_shipments.csv")
hubs_df= pd.read_csv("data/hubs.csv")

# Making a copy of "raw" data and removing the duplicates for 'checks':
dedup_raw= raw.drop_duplicates().copy()

checks= {
    'raw_rows': raw.shape[0],
    'duplicate_rows_removed': raw.duplicated().sum(),
    'rows_after_dedup': dedup_raw.shape[0],
    'missing_distance_km_before': dedup_raw['distance_km'].isnull().sum(),
    'missing_parcel_weight_before': dedup_raw['parcel_weight_kg'].isnull().sum(),
    'sentinel_weight_rows': dedup_raw['parcel_weight_kg'].gt(100).sum(),
    'missing_declared_value_before': dedup_raw['declared_value'].isnull().sum(),
    'missing_payment_mode_before': dedup_raw['payment_mode'].isnull().sum(),
    'service_type_spellings_before': dedup_raw['service_type'].nunique(),
}

print("Metric checks before cleaning data:\n",checks)



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
print("File Saved Successfully!")

print("\n")
feature_target_cols = FEATURES + ["delivery_hours", "is_delayed"]


Q1 = clean_data['delivery_hours'].quantile(0.25)
Q3 = clean_data['delivery_hours'].quantile(0.75)
iqr = Q3 - Q1
lower_bound = Q1 - 1.5 * iqr
upper_bound = Q3 + 1.5 * iqr

checks_2= {
    'service_type_spellings_after': clean_data['service_type'].nunique(),
    'median_distance_km': clean_data['distance_km'].median(),
    'median_parcel_weight_kg': clean_data['parcel_weight_kg'].median(),
    'median_declared_value_inr': clean_data['declared_value_inr'].median(),
    'rows_after_hub_merge': clean_data.shape[0],
    'nulls_in_features_and_target': clean_data[feature_target_cols].isna().sum().sum(),
    'delayed_count': clean_data['is_delayed'].sum(),
    'delay_rate': clean_data['is_delayed'].mean(), 
    'declared_value_mean_inr': clean_data['declared_value_inr'].mean(),
    'declared_value_median_inr': clean_data['declared_value_inr'].median(),
    'log_declared_value_mean': clean_data['log_declared_value'].mean(),
    'log_declared_value_median': clean_data['log_declared_value'].median(),
    'hours_q1': clean_data['delivery_hours'].quantile(0.25),
    'hours_q3': clean_data['delivery_hours'].quantile(0.75),
    'hours_lower_bound': lower_bound,
    'hours_upper_bound': upper_bound,   
    'hours_outlier_count': ((clean_data['delivery_hours'] < lower_bound) | (clean_data['delivery_hours'] > upper_bound)).sum()
}

print("Metric checks after cleaning:\n",checks_2)

# Combine both dictionaries and convert to DataFrame in one go:
checks_df= pd.DataFrame(list({**checks, **checks_2}.items()), columns= ['metric', 'value'])

# Round any floating-point numbers automatically:
checks_df['value']= checks_df['value'].round(4)



# Save the checks DataFrame to a CSV file:
os.makedirs('outputs', exist_ok= True)
checks_df.to_csv('outputs/part_a_checks.csv', index=False)
print("Saved successfully!")