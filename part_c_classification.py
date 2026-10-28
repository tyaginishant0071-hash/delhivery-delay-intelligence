import pandas as pd
import numpy as np
import warnings

from src.shipment_prep import FEATURES
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import OptimizeWarning

# Suppress optimization warnings right at the start:
warnings.simplefilter("ignore", category=OptimizeWarning)

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


all_models = []                      # Used to store the model and metrics of all models.

# 1. Blind baseline model:
# Predicting on "zero's" same length of "y_val":
zero_pred_array = np.zeros(len(y_val))

# Evaluating metrics on "y_val and zero_pred_array":
cm = confusion_matrix(y_val, zero_pred_array)
TN, FP, FN, TP= cm.ravel()              # Used for unpacking the matrix.

all_models.append({
    'model': 'blind_all_on_time',
    'TN': cm[0, 0],
    'FP': cm[0, 1],
    'FN': cm[1, 0],
    'TP': cm[1, 1],
    'accuracy': round(accuracy_score(y_val, zero_pred_array), 4),
    'precision': round(precision_score(y_val, zero_pred_array, zero_division=0), 4),
    'recall': round(recall_score(y_val, zero_pred_array, zero_division=0), 4),
    'f1': round(f1_score(y_val, zero_pred_array, zero_division=0), 4)
})


# 2. Logistic Regression Pipeline:
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model_LR', LogisticRegression(random_state=42))
])

# Fitting on "X_train and y_train" and prediction on "X_val":
pipeline.fit(X_train, y_train)
y_val_pipe_pred = pipeline.predict(X_val)

# Evaluating metrics on "y_val and y_val_pipe_pred":
cm = confusion_matrix(y_val, y_val_pipe_pred)
TN, FP, FN, TP= cm.ravel()              # Used for unpacking the matrix.

all_models.append({
    'model': 'logistic_regression',
    'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
    'accuracy': round(accuracy_score(y_val, y_val_pipe_pred), 4),
    'precision': round(precision_score(y_val, y_val_pipe_pred, zero_division=0), 4),
    'recall': round(recall_score(y_val, y_val_pipe_pred, zero_division=0), 4),
    'f1': round(f1_score(y_val, y_val_pipe_pred, zero_division=0), 4)
})


# 3. Decision Tree Classifier Tuning:
max_depths = [2, 3, 4, 5, 6, 8, 10]
highest_f1_score = -1.0
best_depth = None

# Looping through max_depth:
for d in max_depths:
    model_DTC = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    model_DTC.fit(X_train, y_train)
    y_val_DTC_pred = model_DTC.predict(X_val)
    
    score = f1_score(y_val, y_val_DTC_pred, zero_division=0)

    # Picking highest F1; on tie, taking the smaller d:
    if score > highest_f1_score or (score == highest_f1_score and (best_depth is None or d < best_depth)):
        highest_f1_score = score
        best_depth = d

# Fitting on "X_train and y_train" and prediction on "X_val" with best_DTC(max_depth):
best_DTC = DecisionTreeClassifier(criterion="gini", max_depth=best_depth, random_state=42)
best_DTC.fit(X_train, y_train)
best_DTC_preds = best_DTC.predict(X_val)

# Evaluation metrics:
cm = confusion_matrix(y_val, best_DTC_preds)
TN, FP, FN, TP= cm.ravel()              # Used for unpacking the matrix.

all_models.append({
    'model': f'decision_tree_depth_{best_depth}',
    'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
    'accuracy': round(accuracy_score(y_val, best_DTC_preds), 4),
    'precision': round(precision_score(y_val, best_DTC_preds, zero_division=0), 4),
    'recall': round(recall_score(y_val, best_DTC_preds, zero_division=0), 4),
    'f1': round(f1_score(y_val, best_DTC_preds, zero_division=0), 4)
})


# 4. Random Forest Classifier:
model_RFC= RandomForestClassifier(n_estimators=100, random_state=42)

# Fitting on "X_train and y_train" and prediction on "X_val":
model_RFC.fit(X_train, y_train)
y_val_RFC_pred = model_RFC.predict(X_val)

# Evaluation metrics:
cm = confusion_matrix(y_val, y_val_RFC_pred)
TN, FP, FN, TP= cm.ravel()              # Used for unpacking the matrix.

all_models.append({
    'model': 'random_forest_default',
    'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
    'accuracy': round(accuracy_score(y_val, y_val_RFC_pred), 4),
    'precision': round(precision_score(y_val, y_val_RFC_pred, zero_division=0), 4),
    'recall': round(recall_score(y_val, y_val_RFC_pred, zero_division=0), 4),
    'f1': round(f1_score(y_val, y_val_RFC_pred, zero_division=0), 4)
})

print("All models successfully trained and evaluated!\n")

# Creating a dataframe:
models_df= pd.DataFrame(all_models)
models_df= models_df[['model', 'TN', 'FP', 'FN', 'TP', 'accuracy', 'precision', 'recall', 'f1']]

# Save the file:
models_df.to_csv('outputs/part_c_models.csv', index= False)
print("'outputs/part_c_models.csv' !! Saved Successfully !!\n")
