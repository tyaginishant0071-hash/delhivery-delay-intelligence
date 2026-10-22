import pandas as pd
import numpy as np


PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]



# Function created "clean_shipments":
def clean_shipments(raw, medians= None):

    # Removing the duplicated row by using ".drop_duplicates()" function and ".copy()" function to copy the data and allocate new memory which prevents the "SettingWithCopyWarning".
    clean_df= raw.drop_duplicates().copy()

    # Standardizing column "service_type" into exactly clean categories (Express, Standard, Economy).
    clean_df['service_type']= clean_df['service_type'].str.strip().str.title()

    # Replacing a missing value with the placeholder "Unknown" in "payment_mode" column.
    clean_df['payment_mode']= clean_df['payment_mode'].fillna("Unknown")

    # Removing the rupee sign and the thousands commas, convert to a number, store it in a new column declared_value_inr, and drop the text column:
    clean_df['declared_value_inr']= clean_df['declared_value'].str.replace('[₹,]', '', regex= True).astype(float)
    clean_df.drop(columns= ['declared_value'], inplace= True)

    # "dispatch_time" Converting to a datetime:
    clean_df['dispatch_time']= pd.to_datetime(clean_df['dispatch_time'], errors= 'coerce')

    # parcel_weight_kg: Any value above 100 is a scanner-fault sentinel (the network accepts parcels up to 40 kg) — treat it as missing:
    clean_df['parcel_weight_kg']= clean_df['parcel_weight_kg'].mask(clean_df['parcel_weight_kg'] > 100)      # Replace values > 100 with NaN

    # Fill median in missing data points in columns with there column median: ("distance_km, parcel_weight_kg and declared_value_inr"):
    cols = ['distance_km', 'parcel_weight_kg', 'declared_value_inr']

    # If medians is None, compute them from the current data
    if medians is None:
        medians = {
            'distance_km': float(clean_df['distance_km'].median()),
            'parcel_weight_kg': float(clean_df['parcel_weight_kg'].median()),
            'declared_value_inr': float(clean_df['declared_value_inr'].median())
        }
    
    # Filling missing values using the dictionary
    clean_df[cols]= clean_df[cols].fillna(medians)

    return clean_df, medians     

# Function created "build_features":
def build_features(clean, hubs):

    # # Merge clean_data with hub_df (merge_how= 'left'):
    clean= pd.merge(left= clean, right= hubs, on= 'hub_id',how= 'left')

    # Add promised_hours to the returned DataFrame. For each shipment, use its service_type to look up the promise time from the existing PROMISED_HOURS dictionary:
    clean['promised_hours'] = clean['service_type'].map(PROMISED_HOURS)

    # Add is_cod to the returned DataFrame. For each shipment, use its payment_mode(1= COD, 0= Prepaid and Unknown):
    clean['is_cod'] = (clean['payment_mode'] == 'COD').astype(int)

    # Add is_weekend to the returned DataFrame. For each shipment, use its dispatch_time (Saturday or Sunday= 1 and Monday through Friday= 0):
    clean['is_weekend'] = clean['dispatch_time'].dt.dayofweek.isin([5, 6]).astype(int)

    # Add dispatch_hour by taking the hour from each shipment’s dispatch_time and ignoring the minutes:
    clean['dispatch_hour']= clean['dispatch_time'].dt.hour

    # Add "log_declared_value" by taking the natural logarithm of "declared_value_inr" for each shipment:
    clean['log_declared_value']= np.log(clean['declared_value_inr'])

    return clean