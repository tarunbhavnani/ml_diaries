# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:14:05 2023

@author: tarun
"""

#hierarchical clustering- agglomorative clustering

from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_blobs(n_samples=100, centers=4, random_state=42)

# Perform agglomerative clustering with single linkage
agg_clustering = AgglomerativeClustering(n_clusters=None, linkage='single', distance_threshold=0)
y_agg = agg_clustering.fit_predict(X)

# Plot the results

plt.scatter(X[:, 0], X[:, 1], c=y_agg, cmap='viridis')
plt.show()



agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='single')
y_agg = agg_clustering.fit_predict(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=y_agg, cmap='viridis')
plt.show()




#centroid based- Kmeans

from sklearn.cluster import KMeans

# Perform K-Means clustering with four clusters
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
plt.show()

#elbow
# Calculate within-cluster sum of squares for different number of clusters
wss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wss.append(kmeans.inertia_)

# Plot the results
plt.plot(range(1, 11), wss)
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of squares')
plt.show()




