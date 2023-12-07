import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

model = nn.Linear(1,1)

print(list(model.parameters()))

# [Parameter containing: tensor([[0.5153]], requires_grad=True),
# Parameter containing: tensor([-0.4414], requires_grad=True)]

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 2000
for epoch in range(nb_epochs+1):

    prediction = model(x_train)

    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()

    cost.backward()

    optimizer.step()

    if epoch %100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

new_var = torch.FloatTensor([[4.0]])

pred_y = model(new_var)

print("Pridiction value after training when imput is 4 : ", pred_y)

print("list of w and b", list(model.parameters()))