import matplotlib.pyplot as plt
import math
import pandas as pd

data = pd.read_csv('Dataset.csv')
x = data['Independent_column_name'].values
y = data['Dependent_column_name'].values

x_mean = sum(x) / len(x)
y_mean = sum(y) / len(y)

numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
denominator = sum((xi - x_mean) ** 2 for xi in x)

m = numerator / denominator
b = y_mean - m * x_mean

def predict(x):
    return m * x + b

# Calculate RMSE
rmse = math.sqrt(sum((yi - predict(xi)) ** 2 for xi, yi in zip(x, y)) / len(x))

# Print slope, intercept, and RMSE
print(f'Slope (m): {m}')
print(f'Intercept (b): {b}')
print(f'RMSE: {rmse}')

new_x = int(input("Enter the value of x for testing the regression model: "))

print(f'Predicted value of y is: {predict(new_x)}')

plt.scatter(x, y, label='data points')

regression_line = [predict(val) for val in x]

plt.plot(x, regression_line, color='red', label='regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
