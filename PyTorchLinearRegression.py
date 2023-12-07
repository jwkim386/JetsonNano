import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_train = torch.FloatTensor([1., 2., 3., 4., 5., 6.])
y_train = torch.FloatTensor([9., 16., 23., 30., 37., 44.])

W = torch.zeros(1, requires_grad=True);
b = torch.zeros(1, requires_grad=True);

optimizer = optim.SGD([W, b], lr=0.01)

epochs = 5000

for i in range(epochs):
    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train)**2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if i % 100 == 0:
        print('Epoch ({:10d}/{:10d}) cost: {:10f}, W: {:10f}, b:{:10f}'.format(i, epochs, cost.item(), W.item(), b.item()))

print('W: {:10f}'.format(W.item()))
print('b: {:10f}'.format(b.item()))
print('result : ')
print(x_train * W + b)