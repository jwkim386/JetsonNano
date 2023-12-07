import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/Baakchsu/LinearRegression/master/weight-height.csv")

x = (df['Weight'] - df['Weight'].mean()) / df['Weight'].std()
y = (df["Height"] - df['Height'].mean()) / df["Height"].std()
x_train = np.array(x.values)
y_train = np.array(y.values)

# num_var = x_shape[1]
# n_data = x_shape[0]

W = 0.0
b = 0.0

n_data = len(x_train)

epochs = 1000
learning_rate = 0.01

for i in range(epochs):
    hypothesis = (W * x_train) + b
    cost = np.sum((hypothesis - y_train) ** 2) / n_data
    gradient_w = np.sum((hypothesis - y_train) * 2 * x_train) / n_data
    gradient_b = np.sum((hypothesis - y_train) * 2) / n_data

    W -= learning_rate * gradient_w
    b -= learning_rate * gradient_b

    if i % 100 == 0:
        print('Epoch ({:10d}/{:10d}) cost: {:10f}, W: {:10f}, b:{:10f}'.format(i, epochs, cost, W, b))

print('W: {:10f}'.format(W))
print('b: {:10f}'.format(b))
plt.scatter(x_train[:200], y_train[:200])
pred = np.array(x_train[:200]) * W + b
plt.plot(x_train[:200], pred)
plt.show()