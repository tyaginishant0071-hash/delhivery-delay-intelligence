import pandas as pd
import numpy as np
import os

os.makedirs('outputs', exist_ok= True)

from src.shipment_prep import FEATURES
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score





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

# Loop over depths "1, 2, 3, 4, 6, 8, 12 and None". That’s 8 trees, each DecisionTreeRegressor(max_depth=d, random_state=42):
depths= [1, 2, 3, 4, 6, 8, 12, None]
tree_result= []

for d in depths:
    model_DTR= DecisionTreeRegressor(max_depth= d, random_state= 42)     # "model_DTR" refers to model DecisionTreeRegressor:

    # Fitting each on X_train:
    model_DTR.fit(X_train, y_train)

    # For each tree, computing train_rmse on the training rows and val_rmse on the validation rows.
    y_train_pred= model_DTR.predict(X_train)
    y_val_tree= model_DTR.predict(X_val)

    # Computing metrics
    train_rmse= np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse= np.sqrt(mean_squared_error(y_val, y_val_tree))
    
    # Using dict to add the train_rmse and val_rmse and also, rounding the values by (4):
    tree_result.append({
        'max_depth': str(d),
        'train_rmse': round(train_rmse, 4),
        'val_rmse': round(val_rmse, 4)
    })

# Building a dataframe:
tree_depth_df= pd.DataFrame(tree_result)
tree_depth_df['max_depth']= tree_depth_df['max_depth'].str.replace('None', 'unlimited')

# Save the file:
tree_depth_df.to_csv("outputs/part_b_tree_depth.csv", index= False)
print("'outputs/part_b_tree_depth.csv' !! Saved Successfully !!\n")

# Pick the best depth:
best_depth= tree_depth_df.loc[tree_depth_df['val_rmse'].idxmin()]['max_depth']
# Changing str to int:
if 'unlimited' in best_depth:
  best_depth= None
else:
  best_depth= int(best_depth)

print(f"Best depth: {best_depth}\n")

# Building a scikit-learn Pipeline consisting of a "StandardScaler" and a "LinearRegression":
pipeline= Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Fitting the pipeline on "X_train" and "y_train":
pipeline.fit(X_train, y_train)

# Prediction on X_val, compute MAE, MSE, RMSE, and R2. Appending the dictionary with model: 'linear_regression' and split: 'val':
y_val_pipe = pipeline.predict(X_val)
all_model_metrics.append({
    'model': 'linear_regression',
    'split': 'val',
    'MAE': round(mean_absolute_error(y_val, y_val_pipe), 4),
    'MSE': round(mean_squared_error(y_val, y_val_pipe), 4),
    'RMSE': round(root_mean_squared_error(y_val, y_val_pipe), 4),
    'R2': round(r2_score(y_val, y_val_pipe), 4)
})

# Initializing a "DecisionTreeRegressor" using the optimal "max_depth= best_depth" (along with random_state=42):
model_DT= DecisionTreeRegressor(max_depth= best_depth, random_state= 42)

# Fitting the "DecisionTreeRegressor" on "X_train" and "y_train":
model_DT.fit(X_train, y_train)

# Prediction on X_val, compute MAE, MSE, RMSE, and R2. Appending the dictionary with model: 'decision_tree' and split: 'val':
y_val_DT= model_DT.predict(X_val)
all_model_metrics.append({
    'model': 'decision_tree', 
    'split': 'val',
    'MAE': round(mean_absolute_error(y_val, y_val_DT), 4),
    'MSE': round(mean_squared_error(y_val, y_val_DT), 4),
    'RMSE': round(root_mean_squared_error(y_val, y_val_DT), 4),
    'R2': round(r2_score(y_val, y_val_DT), 4)
})

# Initializing a RandomForestRegressor(n_estimators=100, random_state=42):
rfr= RandomForestRegressor(n_estimators= 100, random_state= 42)

# Fitting the "RandomForestRegressor" on "X_train" and "y_train":
rfr.fit(X_train, y_train)

# Prediction on X_val, compute MAE, MSE, RMSE, and R2. Appending the dictionary with model: 'random_forest' and split: 'val':
y_val_rfr= rfr.predict(X_val)
all_model_metrics.append({
    'model': 'random_forest', 
    'split': 'val',
    'MAE': round(mean_absolute_error(y_val, y_val_rfr),4),
    'MSE': round(mean_squared_error(y_val, y_val_rfr), 4),
    'RMSE': round(root_mean_squared_error(y_val, y_val_rfr), 4),
    'R2': round(r2_score(y_val, y_val_rfr), 4)
})

# Initializing a GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42):
gbr= GradientBoostingRegressor(n_estimators= 100, learning_rate= 0.1, random_state= 42)

# Fitting the "GradientBoostingRegressor" on "X_train" and "y_train":
gbr.fit(X_train, y_train)

# Prediction on X_val, compute MAE, MSE, RMSE, and R2. Appending the dictionary with model: 'gradient_boosting' and split: 'val:
y_val_gbr= gbr.predict(X_val)
all_model_metrics.append({
    'model': 'gradient_boosting', 
    'split': 'val',
    'MAE': round(mean_absolute_error(y_val, y_val_gbr), 4),
    'MSE': round(mean_squared_error(y_val, y_val_gbr),4),
    'RMSE': round(root_mean_squared_error(y_val, y_val_gbr), 4),
    'R2': round(r2_score(y_val, y_val_gbr),4)
})

# Creating a DataFrame:
models_metrics_df= pd.DataFrame(all_model_metrics)
models_metrics_df= models_metrics_df[['model', 'split', 'MAE', 'MSE', 'RMSE', 'R2']]

# Creating a list of different learning rate:
learning_rates= [0.01, 0.1, 1.0]

# Initializing a empty python list "gb_results" to collect the performance metrics:
gb_results= []

# Looping through each learing rates:
for lr in learning_rates:

  # Instantiate a "GradientBoostingRegressor" with (n_estimators=100, the current learning_rate, and random_state=42):
  model_GBR= GradientBoostingRegressor(n_estimators= 100, learning_rate= lr, random_state= 42)

  # Fitting the model using "X_train" and "y_train":
  model_GBR.fit(X_train, y_train)

  # Generate predictions on your validation features (X_val):
  y_val_gb= model_GBR.predict(X_val)

  # Calculating the Validation RMSE and Validation R2 score by comparing predictions against (y_val):
  # Appending a dictionary containing learning_rate, val_rmse, and val_r2 to the results list:
  gb_results.append({
      'learning_rate': lr,
      'val_rmse': round(root_mean_squared_error(y_val, y_val_gb), 4),
      'val_r2': round(r2_score(y_val, y_val_gb), 4)
  })
  
# Converting the list into a pandas DataFrame:
gb_results_df= pd.DataFrame(gb_results)

# Save the file:
gb_results_df.to_csv("outputs/part_b_gb_learning_rate.csv", index= False)
print("'outputs/part_b_gb_learning_rate.csv' !! Saved Successfully !!\n")

# Ranking Candidates by Validation RMSE and removing the "mean_baseline" model:
sorted_ranking = models_metrics_df[models_metrics_df['model'] != 'mean_baseline'].sort_values(by='RMSE', ascending= True).reset_index(drop= True)

# Identifying the top 2 rank candidate:
top_2_rank= sorted_ranking.head(2)['model'].tolist()
print(f"Top 2 rank candidate: {top_2_rank}\n")

best_model_name= top_2_rank[0]
print(f"Best model name: {best_model_name}\n")

# Combine Training and Validation Data (X_train with X_val, and y_train with y_val):
X_train_val= pd.concat([X_train, X_val])
y_train_val= pd.concat([y_train, y_val])

# Dictionary mapping model names to their instantiated objects
model_objects = {
    'linear_regression': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
    'decision_tree': DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Creating a single KFold(n_splits=5, shuffle=True, random_state=42)
kf= KFold(n_splits= 5, shuffle= True, random_state= 42)

cv_results= []
# Looping through each model:
# Running cross-validation for each selected model with the same splitter and default R² scoring:
for model_name in top_2_rank:
  model= model_objects[model_name]

  # Initialize:
  r2_cv_score= cross_val_score(estimator= model, X= X_train_val, y= y_train_val, cv= kf, scoring= 'r2')
  cv_results.append({
      'model': model_name,
      'r2_mean': round(r2_cv_score.mean(), 4),
      'r2_std': round(r2_cv_score.std(), 4)
  })

# Creating a DataFrame and Save the file:
cv_results_df= pd.DataFrame(cv_results)
cv_results_df.to_csv("outputs/part_b_cv.csv", index= False)
print("'outputs/part_b_cv.csv' !!Saved Successfully!!\n")

# Using the pipeline already fitted on X_train, y_train
# Passing unscaled X_test directly (since StandardScaler is built into the pipeline)
y_test_pred = pipeline.predict(X_test)

# Compute MAE, MSE, RMSE, and R2. Adding to the metrics_df: 'linear_regression' and split: 'test':
test_row= {
    'model': 'linear_regression',
    'split': 'test',
    'MAE': round(mean_absolute_error(y_test, y_test_pred), 4),
    'MSE': round(mean_squared_error(y_test, y_test_pred),4),
    'RMSE': round(root_mean_squared_error(y_test, y_test_pred), 4),
    'R2': round(r2_score(y_test, y_test_pred),4)
}

models_metrics_df= pd.concat([models_metrics_df, pd.DataFrame([test_row])], ignore_index= True)

# Save the file:
models_metrics_df.to_csv('outputs/part_b_metrics.csv', index=False)
print("'outputs/part_b_metrics.csv' !! Saved Successfully !\n")