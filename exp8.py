import numpy as np
import pandas as pd

# Step 1: Read the data from a CSV file
data = pd.read_csv('Dataset.csv')

X = data.iloc[:,:-1]  # Assumes the numerical data is in the CSV file

# Step 2: Subtract the mean from the data
mean = np.mean(X, axis=0)
X_centered = X - mean

# Step 3: Calculate the covariance matrix
cov_matrix = np.cov(X_centered, rowvar=False)

# Step 4: Calculate the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 5: Sort the eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]


# Step 6: Choose the number of components and project the data onto the new subspace
n_components = 2
selected_eigenvalues = eigenvalues[:n_components]
selected_eigenvectors = eigenvectors[:, :n_components]
X_pca = np.dot(X_centered, selected_eigenvectors)
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

# Print the covariance matrix, original data, and the transformed data
print("\nExplained Variance Ratios: ", explained_variance_ratio)
print("\nTotal Variance Explained: ", sum(explained_variance_ratio))
print("\n\nCovariance Matrix: ", cov_matrix)
print("\n\nOriginal Data: ", X)
print("\n\nTransformed Data: ", X_pca)