import pandas as pd

from src.shipment_prep import FEATURES
from sklearn.model_selection import train_test_split


# Loading data from "modelling_table.csv":
modelling_df= pd.read_csv("data/modelling_table.csv")

# Spliting the data into two parts from (X) Features and (y) target:
X= modelling_df[FEATURES]
y= modelling_df['delivery_hours']


# Train-Validate-Test Split:
# First call: spliting X and y with test_size=0.3 and random_state=42. And get training and a temporary 30%.
X_train, X_temp, y_train, y_temp= train_test_split(X, y, test_size= 0.3, random_state= 42)
# Second call: splitting the temporary part with test_size=0.5 and random_state=42. And get validation and test.
X_val, X_test, y_val, y_test= train_test_split(X_temp, y_temp, test_size= 0.5, random_state= 42)

# Each row in X_train, X_val, X_test:

print(f"\nX_train rows: {X_train.shape[0]}  |  X_val rows: {X_val.shape[0]}  |  X_test rows: {X_test.shape[0]}\n")         # \n is used to print the output in nextline.
# Each row in Y_train, Y_val, Y_test:
print(f"\ny_train rows: {y_train.shape[0]}  |  y_val rows: {y_val.shape[0]}  |  y_test rows: {y_test.shape[0]}\n")