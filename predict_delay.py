import pandas as pd
import numpy as np
import joblib

from src.shipment_prep import clean_shipments, build_features


# Loading and reading the CSV datafiles:
new_parcels_df= pd.read_csv("data/new_shipments.csv")                  # From CSV datafile: "data/new_shipments.csv"
hubs_df= pd.read_csv("data/hubs.csv")                                  # From CSV datafile: "data/hubs.csv"
segments_df= pd.read_csv("outputs/hub_segments.csv")                   # From CSV datafile: "outputs/hub_segments.csv"
model= joblib.load("models/delay_model.pkl")                           # From PKL datafile: "models/delay_model.pkl"
threshold_df= pd.read_csv("models/delay_threshold.csv")                # From CSV datafile: "models/delay_threshold.csv"
column_order_df= pd.read_csv("models/feature_order.csv")               # From CSV datafile: "models/feature_order.csv"
medians_df= pd.read_csv("models/impute_values.csv")                    # From CSV datafile: "models/impute_values.csv"

# Turning the medians_df to dict (distance_km, parcel_weight_kg, and declared_value_inr):
medians= medians_df.set_index('column')['median'].to_dict()

# Cleaning and preparing new parcels with help of (clean_shipments func):
clean_new_parcels, returned_medians= clean_shipments(new_parcels_df, medians)

# Passing the "clean_new_parcels" and "hubs_df" to (build_features func):
new_features= build_features(clean_new_parcels, hubs_df)

# Selecting the model input containing columns exactly the order list in "feature_order.csv":
feature_list = column_order_df['feature'].tolist()
new_features_df = new_features[feature_list]

#Reading the threshold into a single float variable:
threshold = float(threshold_df['threshold'].iloc[0])

# Predicting and deciding the action:
new_features['delay_probability'] = model.predict_proba(new_features_df)[:, 1].round(4)

# Set up a "flagged" (1 if probability meets or exceeds threshold; otherwise 0):
new_features['flagged'] = np.where(new_features['delay_probability'] >= threshold, 1, 0)

# Joining the "segments_df" to each parcel using "hub_id":
new_fea_seg_df = pd.merge(new_features, segments_df[['hub_id', 'segment']], on='hub_id', how='left')

# Assigning the action (Flagged and segment Escalate → Escalate to hub manager; Flagged with any other segment → Proactive customer call; Not flagged → No action):
conditions= [
    (new_fea_seg_df['flagged'] == 1) & (new_fea_seg_df['segment'] == 'Escalate'),
    (new_fea_seg_df['flagged'] == 1) & (new_fea_seg_df['segment'] != 'Escalate'),
    (new_fea_seg_df['flagged'] == 0)
]

choices= ['Escalate to hub manager', 'Proactive customer call', 'No action']
new_fea_seg_df['action']= np.select(conditions, choices, default= 'Unknown')

# Confirming that no row resulted in 'Unknown' action:
if (new_fea_seg_df['action'] == 'Unknown').any():
    raise ValueError("Error: Some shipments resulted in 'Unknown' action due to unhandled conditions.")

# Creating a new dataframe with "new_shipment_action":
new_shipment_action= new_fea_seg_df[['shipment_id', 'hub_id', 'segment', 'delay_probability', 'flagged', 'action']]

# Saving the CSV file:
new_shipment_action.to_csv('outputs/new_shipment_actions.csv', index= False)
print("'outputs/new_shipment_actions.csv' saved successfully!\n")