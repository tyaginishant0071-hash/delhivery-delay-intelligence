import pandas as pd

from src.shipment_prep import FEATURES
from sklearn.model_selection import train_test_split

# Reading the data from data folder "modelling_table.csv":
model_table_df= pd.read_csv("data/modelling_table.csv")
print("Rows and columns of model_table_df:", model_table_df.shape, "\n")                         # Print the rows (1200) and columns (27) of the model_table_df.

# Splitting the data into features(X) and target(y):
X= model_table_df[FEATURES]                         # Using "FEATURES": there are only 17 columns.
y= model_table_df['is_delayed']

# Using "train_test_split(test_size= 0.3, stratify, random_state= 42)" with stratify on the target in both calls:
X_train, X_temp, y_train, y_temp= train_test_split(X, y, test_size= 0.3, stratify= y, random_state= 42)      # First call (Stratify on target(y)).
X_val, X_test, y_val, y_test= train_test_split(X_temp, y_temp, test_size= 0.5, stratify= y_temp, random_state= 42)   # Second call (Stratify on target(y_temp)).

# Printing the row size of: (X_train, X_val and X_test) and count of "is_delayed":
print(f"X_train rows: {X_train.shape[0]}  |  y_train 'is_delayed' count: {y_train.sum()}")
print(f"X_val rows: {X_val.shape[0]}  |  y_val 'is_delayed' count: {y_val.sum()}")
print(f"X_test rows: {X_test.shape[0]}  |  y_test 'is_delayed' count: {y_test.sum()}\n")

