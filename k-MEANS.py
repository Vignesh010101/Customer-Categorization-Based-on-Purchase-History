# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4th 7:40:39 2024

@author: Vignesh
"""

import pandas as pd 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt

# Read the dataset
data = pd.read_csv('Mall_Customers.csv')

# Extracting relevant features for clustering (Annual Income and Spending Score)
X = data.iloc[:, [3, 4]].values

# Initialize an empty list to store the Within-Cluster-Sum-of-Squares (WCSS) values
wcss = []

# Calculate WCSS for different values of K (number of clusters)
for i in range(1, 11):
    # Initialize KMeans with current value of K
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    # Fit KMeans to the data
    kmeans.fit(X)
    # Append the inertia_ (WCSS) to the list
    wcss.append(kmeans.inertia_)

# Plot the Within-Cluster-Sum-of-Squares (WCSS) for different values of K
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the elbow method, select the number of clusters (K=5 in this case)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
# Predict the cluster labels for each data point
y_pred = kmeans.fit_predict(X)

# Plot the clusters and centroids
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
