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

    return (clean_df, medians)     
 