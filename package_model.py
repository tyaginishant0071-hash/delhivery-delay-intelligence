import pandas as pd
import numpy as np
import os
import joblib

from src.shipment_prep import clean_shipments, FEATURES
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


os.makedirs('models', exist_ok= True)

# Loading and reading the data from "outputs/part_c_final_config.csv", "data/modelling_table.csv" and "data/raw_shipments.csv":
final_config_df= pd.read_csv("outputs/part_c_final_config.csv")
modelling_table_df= pd.read_csv("data/modelling_table.csv")
shipments_df= pd.read_csv("data/raw_shipments.csv")

# Preparing the model inputs features(X) and target(y):
X_fea= modelling_table_df[FEATURES]
y_tar= modelling_table_df['is_delayed']

# Using the same splitting procedure used in "part_c":
X_train, X_temp, y_train, y_temp= train_test_split(X_fea, y_tar, test_size= 0.3, stratify= y_tar, random_state= 42)
X_val, X_test, y_val, y_test= train_test_split(X_temp, y_temp, test_size= 0.5, stratify= y_temp, random_state= 42)

# Checking the split size and labels count:
print(f"X_train shape: {X_train.shape[0]}  |  y_train 'is_delayed' count: {y_train.sum()}")
print(f"X_val shape: {X_val.shape[0]}  |  y_val 'is_delayed' count: {y_val.sum()}")
print(f"X_test shape: {X_test.shape[0]}  |  y_test 'is_delayed' count: {y_test.sum()}\n")

# Calling the "clean_shipments" with "shipments_df" to return two things: the clean shipment table and dict of medians containing (distance_km, parcel_weight_kg, 'declared_value_inr'):
clean_shipments_df, medians= clean_shipments(shipments_df)

# Creating a "medians" table:
medians_df= pd.DataFrame(medians.items(), columns= ['column', 'median'])

# Rounding the "medians_df":
medians_df= medians_df.round(4)

# Saving the file to csv:
medians_df.to_csv("models/impute_values.csv", index= False)
print("'models/impute_values.csv' saved successfully!\n")

# Reading the settings from the "final_config":
selector_k= final_config_df['selector_k'][0]
n_estimators= final_config_df['n_estimators'][0]
max_depth= final_config_df['max_depth'][0]
min_samples_leaf= final_config_df['min_samples_leaf'][0]
threshold= final_config_df['threshold'][0]
chosen_from= final_config_df['chosen_from'][0]

# Handling "max_depth":
max_depth= int(max_depth) if max_depth != 'unlimited' else None

# Building the pipeline:
pipeline= Pipeline([
    ('selector', SelectKBest(score_func= f_classif, k= selector_k)),
    ('model', RandomForestClassifier(n_estimators= n_estimators, max_depth= max_depth, min_samples_leaf= min_samples_leaf, random_state= 42))
])

# Fitting the training split:
pipeline.fit(X_train, y_train)

# Saving the fitted "pipeline" to "models/delay_model.pkl" using joblib:
joblib.dump(pipeline, 'models/delay_model.pkl')
print("'models/delay_model.pkl' saved successfully!\n")

# Saving the "threshold" to "models/delay_threshold.csv":
threshold_df= final_config_df[['threshold']]
threshold_df.to_csv('models/delay_threshold.csv', index= False)
print("'models/delay_threshold.csv' saved successfully!\n")

# Saving the "FEATURES" to "models/feature_order.csv" naming column 'feature':
features= pd.DataFrame({'feature': FEATURES})
features.to_csv('models/feature_order.csv', index= False)
print("'models/feature_order.csv' saved successfully!\n")

# Loading "models/delay_model.pkl" using joblib.load:
loaded_model= joblib.load('models/delay_model.pkl')

# Using the same test rows for both predictions:
y_load_test_pred_proba= loaded_model.predict_proba(X_test)[:, 1]
y_pipe_test_pred_proba= pipeline.predict_proba(X_test)[:, 1]

# Calculating the largest difference by finding the absolute diff between original and loaded probability:
reload_max_abs_probability_diff= np.max(np.abs(y_load_test_pred_proba - y_pipe_test_pred_proba))
print(f"Largest difference between predictions: {reload_max_abs_probability_diff}\n")

#Creating a "reload-check table" which consist of two columns (metric, value):
reload_check_table= pd.DataFrame([
    {'metric': 'test_rows_scored', 'value': X_test.shape[0]},
    {'metric': 'reload_max_abs_probability_diff', 'value': float(np.round(reload_max_abs_probability_diff, 4))},
    {'metric': 'threshold', 'value': float(np.round(threshold_df['threshold'][0], 4))}
], dtype= object)

# Saving the file to "outputs/part_d_reload_check.csv":
reload_check_table.to_csv('outputs/part_d_reload_check.csv', index= False)
print("'outputs/part_d_reload_check.csv' saved successfully!\n")

# Matrix comparison of "pipeline" and "loaded_model":
pipe_pred= (y_pipe_test_pred_proba >= threshold_df['threshold'][0]).astype(int)
cm_pipe= confusion_matrix(y_test, pipe_pred)
TN, FP, FN, TP= cm_pipe.ravel()
print(f"Pipeline Confusion Matrix= TN: {TN} | FP: {FP} | FN: {FN} | TP: {TP}\n")

load_pred= (y_load_test_pred_proba >= threshold_df['threshold'][0]).astype(int)
cm_load= confusion_matrix(y_test, load_pred)
TN, FP, FN, TP= cm_load.ravel()
print(f"Loaded Model Confusion Matrix= TN: {TN} | FP: {FP} | FN: {FN} | TP: {TP}\n")