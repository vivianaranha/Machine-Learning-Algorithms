# Import necessary libraries
from sklearn.cluster import KMeans
import numpy as np

# Sample data (e.g., points in 2D space)
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

# Initialize and fit the model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Get the cluster centers and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print("Cluster Centers:\n", centroids)
print("Labels:", labels)


