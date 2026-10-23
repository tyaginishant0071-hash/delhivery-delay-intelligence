import pandas as pd
import os

os.makedirs('outputs', exist_ok= True)


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

# Combine dictionaries:
all_checks = {**checks, **checks_2}

# Round only the actual floating-point numbers
rounded_checks= {}
for name, value in all_checks.items():
    if isinstance(value, float):
        rounded_checks[name]= round(value, 4)
    else:
        rounded_checks[name]= int(value)

# Converting dictionary into dataframe:
checks_df= pd.DataFrame(list(rounded_checks.items()), columns= ['metric', 'value'], dtype= object)

# Save to CSV:
checks_df.to_csv('outputs/part_a_checks.csv', index=False)
print("'outputs/part_a_checks.csv' !!Saved successfully!!")


# Adding new column to clean_data "distance_band":
clean_data['distance_band']= pd.cut(
    clean_data['distance_km'],
    bins= [0, 200, 400, 600, 2000],
    labels= ['0-200', '201-400', '401-600', '601+'],
    right= True
)

# Grouping "service_type", "hub_tier", "distance_band" and check some metric:
group_cols = ['service_type', 'hub_tier', 'distance_band']
summaries_list = []

for cols in group_cols:
    summary= clean_data.groupby(cols, observed=False).agg(
        shipments=(cols, 'count'),
        delayed=('is_delayed', 'sum'),
        delay_rate= ('is_delayed', 'mean'),
        mean_hours= ('delivery_hours', 'mean'),
    ).reset_index().rename(columns= {cols: 'group'})

    # Applying astype(str) to the DataFrame column:
    summary['group'] = summary['group'].astype(str)
    
    # Adding the dimension column:
    summary['dimension'] = cols

    summaries_list.append(summary)

# Combine all summaries into one master table:
summaries= pd.concat(summaries_list, ignore_index= True)

# Rounding to (4 decimal place) the columns with float value:
summaries[['delay_rate', 'mean_hours']] = round(summaries[['delay_rate', 'mean_hours']], 4)

# Changing the column order as per the brief:
col_order = ['dimension', 'group', 'shipments', 'delayed', 'delay_rate', 'mean_hours']
master_summary = summaries[col_order]

master_summary.to_csv('outputs/part_a_delay_by_group.csv', index=False)
print("'outputs/part_a_delay_by_group.csv' !!Saved successfully!!")

print("\n")
print(master_summary)