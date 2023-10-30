import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('Dataset.csv')
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bagging with Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
bagging_accuracy = accuracy_score(y_test, rf_pred)
print("\nBagging (Random Forest) Predictions:", rf_pred)
print("\nBagging (Random Forest) Accuracy:", bagging_accuracy)

# Boosting with AdaBoost
base_model = DecisionTreeClassifier(max_depth=1)
adaboost = AdaBoostClassifier(base_model, n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)
adaboost_pred = adaboost.predict(X_test)
boosting_accuracy = accuracy_score(y_test, adaboost_pred)
print("\n\nBoosting Predictions:", adaboost_pred)
print("\nBoosting (AdaBoost) Accuracy:", boosting_accuracy)

# Stacking
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=1)),
]

meta_classifier = LogisticRegression()
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_classifier)
stacking_model.fit(X_train, y_train)
stacking_pred = stacking_model.predict(X_test)
stacking_accuracy = accuracy_score(y_test, stacking_pred)
print("\n\nStacking Predictions:", stacking_pred)
print("\nStacking Ensemble Accuracy:", stacking_accuracy)
