import torch 
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# for reproducibility 
torch.manual_seed(777) #random seed 고정
if device == 'cuda': #gpu연산 setting
    torch.cuda.manual_seed_all(777)

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device) 
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

model = nn.Sequential(
	nn.Linear(2, 4, bias=True), # input_layer = 2, hidden_layer1 = 10 
	nn.Sigmoid(), 
	# nn.Linear(10, 10, bias=True), # hidden_layer1 = 10, hidden_layer2 = 10 
	# nn.Sigmoid(), 
	# nn.Linear(10, 10, bias=True), # hidden_layer2 = 10, hidden_layer3 = 10 
	# nn.Sigmoid(), 
	nn.Linear(4, 1, bias=True), # hidden_layer3 = 10, output_layer = 1 
	nn.Sigmoid() 
	).to(device)

criterion = torch.nn.BCELoss().to(device) #cost함수
optimizer = torch.optim.SGD(model.parameters(), lr=1)

for epoch in range(10001):
    optimizer.zero_grad()
    # forward 연산
    hypothesis = model(X)

    # 비용 함수
    cost = criterion(hypothesis, Y)
    cost.backward() 
    optimizer.step()

    # 100의 배수에 해당되는 에포크마다 비용을 출력
    if epoch % 100 == 0:
        print(epoch, cost.item())

print(list(model.parameters()))

with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(Y): ', Y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())
