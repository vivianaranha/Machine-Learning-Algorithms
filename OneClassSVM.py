# Import necessary libraries
from sklearn.svm import OneClassSVM
import numpy as np

# Sample data (normal data points clustered around 0)
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]  # Create a dataset with points around two clusters

# New test data including some outliers
X_test = np.r_[X + 2, X - 2, np.random.uniform(low=-6, high=6, size=(20, 2))]

# Initialize and train the model
model = OneClassSVM(gamma='auto', nu=0.1)
model.fit(X_train)

# Predict on test data (-1 indicates an anomaly, 1 indicates normal)
predictions = model.predict(X_test)

# Display predictions
print("Predictions:", predictions)
