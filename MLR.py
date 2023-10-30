#MULTIPLE LINEAR REGRESSION
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# User-provided data
data = pd.read_csv('Dataset.csv')
X1 = np.array(data['Independent_Variable1_Name'].values)
X2 = np.array(data['Independent_Variable2_Name'].values)
y = np.array(data['Dependent_Variable_Name'].values)

# Concatenate the features and add a column of ones for the intercept
X = np.vstack((X1, X2, np.ones(X1.shape[0]))).T

# Calculate the coefficients using OLS

coefficients = np.linalg.pinv(X).dot(y)
# Extract the coefficients
coef1, coef2, intercept = coefficients[0], coefficients[1], coefficients[2]

print("Coefficient 1:", coef1)
print("Coefficient 2:", coef2)
print("Intercept:", intercept)

# Create prediction function
def predict(X1, X2):
    return coef1 * X1 + coef2 * X2 + intercept

# Calculate Root Mean Square Error (RMSE)
predictions = predict(X1, X2)
rmse = np.sqrt(np.mean((y - predictions) ** 2))

print("Root Mean Square Error (RMSE):", rmse)

new_x1 = int(input("Enter the value of x1 for testing the regression model: "))
new_x2 = int(input("Enter the value of x2 for testing the regression model: "))

print(f'Predicted value of y is: {predict(new_x1, new_x2)}')

# Create a grid for plotting
x1 = np.linspace(min(X1), max(X1), 10)
x2 = np.linspace(min(X2), max(X2), 15)
X1, X2 = np.meshgrid(x1, x2)
Y = predict(X1, X2)

# Plot the 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, y, color='red', marker='o', label='Data Points')
ax.plot_surface(X1, X2, Y, color='blue', alpha=0.5, label='Regression Plane')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.show()
