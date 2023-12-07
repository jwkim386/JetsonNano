import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]] 
y_data = [[0], [0], [0], [1], [1], [1]] 
x_train = torch.FloatTensor(x_data) #data를 tensor로 변환
y_train = torch.FloatTensor(y_data)

model = nn.Sequential(
    nn.Linear(2, 1), # input_dim = 2, output_dim = 1 
    nn.Sigmoid() # 출력은 시그모이드 함수를 거친다 
)

# optimizer 설정 
optimizer = optim.SGD(model.parameters(), lr=1) 

nb_epochs = 1000 
for epoch in range(nb_epochs + 1): 

    # H(x) 계산 
    hypothesis = model(x_train) 
    # cost 계산 
    cost = F.binary_cross_entropy(hypothesis, y_train) #이진 분류할때 사용하는                                                        cost함수, 0 or 1을 return한다.
    # cost로 H(x) 개선 
    optimizer.zero_grad() 
    cost.backward() 
    optimizer.step()

    # 20번마다 로그 출력 
    if epoch % 10 == 0:
        # 예측값이 0.5를 넘으면 True로 간주
        prediction = hypothesis >= torch.FloatTensor([0.5])
        # 실제값과 일치하는 경우만 True로 간주 
        correct_prediction = prediction.float() == y_train
        # 정확도를 계산
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        # 각 에포크마다 정확도를 출력
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))

x_test = torch.FloatTensor([3,4])
y_prid = model(x_test) >= torch.Tensor([0.5])

print(y_prid)