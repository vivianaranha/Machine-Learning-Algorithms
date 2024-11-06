# Import necessary libraries
from sklearn.decomposition import PCA
import numpy as np

# Sample data (e.g., points in 3D space)
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7], [5, 7, 8]])

# Initialize and fit the model
pca = PCA(n_components=2)  # Reducing to 2 dimensions
X_reduced = pca.fit_transform(X)

print("Reduced Data:\n", X_reduced)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

	