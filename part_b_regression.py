import pandas as pd
import numpy as np
import os

os.makedirs('outputs', exist_ok= True)

from src.shipment_prep import FEATURES
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



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
print(f"y_train rows: {y_train.shape[0]}  |  y_val rows: {y_val.shape[0]}  |  y_test rows: {y_test.shape[0]}\n")

# Scaling the train data:
st_sc= StandardScaler()
X_train_scaled= st_sc.fit_transform(X_train)
# Transforming X_test with the same fitted scaler.
X_test_scaled= st_sc.transform(X_test)

# Columns metric and value:
checks= {
    'train_rows': X_train.shape[0],
    'val_rows': X_val.shape[0],
    'test_rows': X_test.shape[0],
    'train_scaled_max_abs_mean': np.abs(X_train_scaled.mean(axis= 0)).max(),
    'test_scaled_max_abs_mean': np.abs(X_test_scaled.mean(axis= 0)).max()
}

# Create DataFrame with object dtype to preserve mixed types and rounding the values(4):
rounded_dict= {name: round(value, 4) for name, value in checks.items()}
checks_df = pd.DataFrame(rounded_dict.items(), columns= ['metric', 'value'], dtype= object)

# Save the file:
checks_df.to_csv("outputs/part_b_checks.csv", index= False)
print("'outputs/part_b_checks.csv' !! Saved Successfully !!\n")

# Predicting the mean of "y_train" for every validation row and record its metrics:
# Computing the mean of y_train:
y_train_mean= y_train.mean() 
print("Mean of y_train:\n", y_train_mean)

# Making a prediction for every validation row, which is that one number repeated 180 times using "np.full":
y_val_pred= np.full(shape= y_val.shape, fill_value=y_train_mean)

# Store the mean baseline results and score it against y_val with four numbers using "sklearn.metrics":
baseline_metrics = {
    'model': 'mean_baseline',
    'split': 'val',
    'MAE': round(mean_absolute_error(y_val, y_val_pred), 4),
    'MSE': round(mean_squared_error(y_val, y_val_pred), 4),
    'RMSE': round(np.sqrt(mean_squared_error(y_val, y_val_pred)), 4),
    'R2': round(r2_score(y_val, y_val_pred), 4)
}

# Initialize a list to collect results across all models:
all_model_metrics = [baseline_metrics]

print("Baseline metrics collected successfully!\n")

