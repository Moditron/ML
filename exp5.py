import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Load data
data = pd.read_csv('Dataset.csv')
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable
print(X)
print("\n")
print(y)
print("\n")

# Decision Tree model
model = DecisionTreeClassifier()

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# Display accuracy for each fold
for fold, score in enumerate(scores, start=1):
    print(f"Accuracy for Fold {fold}: {score}")

# Mean accuracy across all folds
mean_accuracy = scores.mean()
print(f"\nMean Accuracy: {mean_accuracy}")
