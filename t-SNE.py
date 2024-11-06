# Import necessary libraries
from sklearn.manifold import TSNE
import numpy as np

# Sample data (e.g., points in high-dimensional space)
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7], [5, 7, 8], [8, 9, 10], [9, 10, 11]])

# Initialize and fit the model
tsne = TSNE(n_components=2, perplexity=5, random_state=42)  # Reducing to 2D for visualization
X_reduced = tsne.fit_transform(X)

print("Reduced Data:\n", X_reduced)
