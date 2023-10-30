import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd

# Read the data from the CSV file
data = pd.read_csv('Dataset.csv')

# Assuming the last column is the target variable and the rest are features
X = data.iloc[:, :-1]  # Features (all columns except the last)
y = data.iloc[:, -1]

# Generate synthetic data
# X, y = datasets.make_classification(n_samples=10, n_features=2, n_informative=2, n_redundant=0, random_state=42)
print(X)

print(f"y:{y}")
# Create an SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# Calculate and print accuracy
y_pred = clf.predict(X)
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# Print the number of support vectors
print(clf.support_vectors_)
num_support_vectors = len(clf.support_vectors_)
print("Number of support vectors:", num_support_vectors)

# Create a meshgrid to visualize the decision boundary
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot the data points
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.coolwarm, s=30)

# Highlight support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')

# Set axis labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary')

# Show the plot
plt.show()
