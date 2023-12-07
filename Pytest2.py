import numpy as np

x_train = np.array([[1],[2],[3]]);
y_train = np.array([[2],[4],[6]]);

W = 0.0;
b = 0.0;

lr=0.01
n_data = len(x_train)
nb_epochs = 2000;

for epoch in range(nb_epochs+1):

    hypothesis = x_train * W + b;

    cost = np.sum((hypothesis - y_train)**2) / n_data

    gradient_w = np.sum((W * x_train - y_train + b) * 2 * x_train) / n_data
    gradient_b = np.sum((W * x_train - y_train + b) * 2) / n_data

    W -= lr * gradient_w
    b -= lr * gradient_b

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:,.3f}, b: {:.3f} Cost: {:.6f}'.format(epoch, nb_epochs, W, b, cost))