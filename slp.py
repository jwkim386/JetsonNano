import torch
import torch.nn as nn

# device = 'cuda' if torch.cuda.is_available() else 'cpu' 
device = 'cpu'
torch.manual_seed(777) 
if device == 'cuda': 
    torch.cuda.manual_seed_all(777)

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device) 
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

linear = nn.Linear(2, 1, bias=True) #input_dim, output_dim
sigmoid = nn.Sigmoid() 
model = nn.Sequential(linear, sigmoid).to(device)

# 비용 함수와 옵티마이저 정의 
criterion = torch.nn.BCELoss().to(device) #이진 분류에서 사용하는 Cross Entropy함수
optimizer = torch.optim.SGD(model.parameters(), lr=1)

#10,001번의 에포크 수행. 0번 에포크부터 10,000번 에포크까지. 
for step in range(10001): 
    optimizer.zero_grad() 
    hypothesis = model(X) 
    # 비용 함수 
    cost = criterion(hypothesis, Y) 
    cost.backward()
    optimizer.step() 
    if step % 100 == 0: # 100번째 에포크마다 비용 출력 
        print(step, cost.item())

print(list(model.parameters()))

with torch.no_grad(): 
    hypothesis = model(X) 
    predicted = (hypothesis > 0.5).float() 
    accuracy = (predicted == Y).float().mean() 
    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(Y): ', Y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())
