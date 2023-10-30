import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Objective function
def obj(x):
    return x**2

# Gradient of the objective function
def gradient(x):
    return 2 * x

# Load data from a CSV file
data = pd.read_csv('Dataset.csv')  # Replace 'your_file.csv' with the actual file path

initial_x = data.iloc[0, 0]  # Take the initial value from the CSV file
learning_rate = 0.1
itr = 20

x = initial_x
x_history = [x]
loss_history = [obj(x)]

for i in range(itr):
    grad = gradient(x)
    x -= learning_rate * grad

    x_history.append(x)
    loss_history.append(obj(x))
    print('Iteration ', i)
    print('x:', x)
    print('f(x):', obj(x))

x_values = np.linspace(-6, 6, 400)
y_values = obj(x_values)

plt.plot(x_values, y_values, label="Objective Function")
plt.scatter(x_history, loss_history, color='red', label="Gradient Descent Function ")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()



