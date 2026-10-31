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

# Cluster ordering and segment naming:
# Grouping 'delay_rate' with cluster labels 'k=3' on dataframe "hub_profile_df":
cluster_col= 'cluster_k3' if 'cluster_k3' in hub_profile_df.columns else 'cluster'           # Asserting the cluster column name based on the existence of 'cluster_k3' in the dataframe.

# Computing mean delay_rate, sorting:
cluster_delay_means= hub_profile_df.groupby(cluster_col)['delay_rate'].mean()
cluster_delay_means= cluster_delay_means.sort_values()

# Mapping the "cluster_delay_means" with name("Standard", "Watch", "Escalate"):
cluster_names= {}
sorted_cluster_ids= cluster_delay_means.index.tolist()
if len (sorted_cluster_ids) >= 3:
  cluster_names[sorted_cluster_ids[0]]= 'Standard'
  cluster_names[sorted_cluster_ids[1]]= 'Watch'
  cluster_names[sorted_cluster_ids[2]]= 'Escalate'

# Assigning names to the hub:
hub_profile_df['segment']= hub_profile_df[cluster_col].map(cluster_names)

# Filtering DataFrame to keep only four specific columns: "hub_id", "segment", "hub_city", and "hub_tier":
if 'hub_city' not in hub_profile_df.columns or 'hub_tier' not in hub_profile_df.columns:
  meta_data= modelling_table_df[['hub_id', 'hub_city', 'hub_tier']].drop_duplicates()
  hub_profile_df= pd.merge(hub_profile_df, meta_data, on= 'hub_id', how= 'left')

segment_df= hub_profile_df[['hub_id', 'segment', 'hub_city', 'hub_tier']]

# Save the file:
segment_df.to_csv('outputs/hub_segments.csv', index=False)
print("'outputs/hub_segments.csv' saved successfully!\n")

# Grouping the data by "segment" and computing metrics:
segment_profile= hub_profile_df.groupby('segment').agg(
    hubs= ('hub_id', 'count'),
    mean_delivery_hours= ('mean_delivery_hours', 'mean'),
    delay_rate= ('delay_rate', 'mean'),
    mean_distance_km= ('mean_distance_km', 'mean'),
    mean_n_stops= ('mean_n_stops', 'mean')
).reset_index()

# Sorting and Rounding the data:
segment_profile= segment_profile.sort_values(by= 'hubs', ascending= True)
segment_profile= segment_profile.round({
    'hubs': 0,
    'mean_delivery_hours': 4,
    'delay_rate': 4,
    'mean_distance_km': 4,
    'mean_n_stops': 4
})

# Creating a dataframe and save the file:
segment_profile_df= pd.DataFrame(segment_profile)
segment_profile_df.to_csv('outputs/segment_profile.csv', index=False)
print("'outputs/segment_profile.csv' saved successfully!\n")

# Creating a mapping logic for tiers (Standard= tier 1, Watch= tier 2 and Escalate= tier 3):
# Creating a copy of a "segment_df":
segment_df_copy= segment_df.copy()

# Mapping the tiers:
tier_mapping= {
    'Standard': 1,
    'Watch': 2,
    'Escalate': 3
}
segment_df_copy['expected_segment']= segment_df_copy['segment'].map(tier_mapping)

# Filtering for "tier" mismatch in "segment_df_copy":
mismatch_df= segment_df_copy[hub_profile_df['hub_tier'] != segment_df_copy['expected_segment']]
mismatch_df= mismatch_df[['hub_id', 'segment', 'hub_city', 'hub_tier',]]

# Save the file:
mismatch_df.to_csv('outputs/tier_vs_segment_mismatch.csv', index=False)
print("'outputs/tier_vs_segment_mismatch.csv' saved successfully!\n")