import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

torch.manual_seed(1)

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]] 
y_train = [2, 2, 2, 1, 1, 1, 0, 0] 

x_train = torch.FloatTensor(x_train) 
y_train = torch.LongTensor(y_train)

# 모델을 선언 및 초기화. 4개의 특성을 가지고 3개의 클래스로 분류. input_dim=4, output_dim=3. 
model = nn.Linear(4, 3)

# optimizer 설정 
optimizer = optim.SGD(model.parameters(), lr=0.1) 

nb_epochs = 1000 
for epoch in range(nb_epochs + 1): 

    # H(x) 계산 
    prediction = model(x_train) 

    # cost 계산 
    cost = F.cross_entropy(prediction, y_train) #cost(w)

    # cost로 H(x) 개선 
    optimizer.zero_grad() 
    cost.backward() 
    optimizer.step() 

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))


