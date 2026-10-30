import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from itertools import combinations

os.makedirs('outputs', exist_ok=True)

# Loading and Reading the input CSV file "data/modelling_table.csv" into a pandas DataFrame:
modelling_table_df= pd.read_csv('data/modelling_table.csv')

# Creating a "hub_profile" for grouping "hub_id" and computing aggregations:
hub_profile= modelling_table_df.groupby('hub_id').agg(
    shipments= ('shipment_id', 'count'),
    mean_delivery_hours= ('delivery_hours', 'mean'),
    delay_rate= ('is_delayed', 'mean'),
    mean_distance_km= ('distance_km', 'mean'),
    mean_n_stops= ('n_stops', 'mean')
).reset_index()

# Sorting the "hub_profile" and rounding the "hub_profile_df":
hub_profile= hub_profile.sort_values(by= 'hub_id', ascending= True).reset_index(drop= True)
hub_profile= hub_profile.round({
    'shipments': 0,
    'mean_delivery_hours': 4,
    'delay_rate': 4,
    'mean_distance_km': 4,
    'mean_n_stops': 4
})

# Save the file:
hub_profile.to_csv('outputs/hub_profile.csv', index=False)
print("'outputs/hub_profile.csv' saved successfully!\n")

# Ensuring correct sorting and feature selection on "hub_profile_df":
hub_profile_df= hub_profile.sort_values(by= 'hub_id', ascending= True).reset_index(drop= True)

# Defining four features to cluster:
features_clust= hub_profile_df[['mean_delivery_hours', 'delay_rate', 'mean_distance_km', 'mean_n_stops']]

# Using "StandardScaler" on "features_clust" to ensures that features with larger numerical ranges don't disproportionately dominate the distance calculations in K-Means:
scaler= StandardScaler()
scaled_features_clust= scaler.fit_transform(features_clust)

# Initializing the "KMeans" model with (K=3, random_state= 42, n_init=10):
kmeans= KMeans(n_clusters= 3, n_init= 10, random_state= 42)

# Fitting and predicting the "kmeans" model:
kmeans.fit(scaled_features_clust)
hub_profile_df['cluster']= kmeans.predict(scaled_features_clust)

# Counts of cluster:
count= hub_profile_df['cluster'].value_counts()
print(f"Counts: {count}\n")

# Using "euclidean_distances" on matrix ("scaled_features_clust"):
dist_matrix= euclidean_distances(scaled_features_clust)

# Generating all unique unordered index pairs (276 pairs for 24 hubs):
pair_indices= list(combinations(range(len(hub_profile_df)), 2))

results = []                           # Created an empty results container to store result.
k_values = [2, 3, 4, 5]                # List of k used for different cluster.

# Looping through each k:
for k in k_values:
  # Initializing "KMeans" model with (n_cluster= k, n_init= 10, random_state= 42):
  kmeans_model= KMeans(n_clusters= k, n_init= 10, random_state= 42)
  # Fitting and predicting:
  labels= kmeans_model.fit_predict(scaled_features_clust)

  # Store labels temporarily in the dataframe for easy lookup:
  hub_profile_df[f'cluster_k{k}']= labels

  # Created two empty list:
  same_cluster_dists= []
  diff_cluster_dists= []

  # Evaluating pair based on current k's cluster assignment:
  for i, j in pair_indices:
    d= dist_matrix[i, j]
    if labels[i] == labels[j]:
      same_cluster_dists.append(d)
    else:
      diff_cluster_dists.append(d)

  # Computing the (mean) distance "diff_cluster_dists", and divide it by the average distance "same_cluster_dists" to get the across_to_within ratio:
  mean_across= np.mean(diff_cluster_dists) if diff_cluster_dists else 0
  mean_within= np.mean(same_cluster_dists) if same_cluster_dists else 0
  ratio= mean_across / mean_within if mean_within > 0 else np.nan  

  # Counting the hubs belong to each cluster using np.unique(return_counts= True):
  unique_clusters, counts= np.unique(labels, return_counts= True)
  
  # Sorting the counts in descending order (e.g., 9, 8, 7) and join them together into a single string separated by hyphens using '-'.join(map(str, sorted_counts)) (resulting in "9-8-7"):
  sorted_counts= sorted(counts, reverse= True)
  cluster_size_str= '-'.join(map(str, sorted_counts))

  # Appending into results:
  results.append({
    'K': k,
    'across_over_within': round(ratio, 4),
    'cluster_sizes': cluster_size_str
    })

# Creating and saving a dataframe into csv:
separation_df= pd.DataFrame(results)
separation_df.to_csv('outputs/part_d_separation.csv', index=False)
print("'outputs/part_d_separation.csv' saved successfully!\n")
