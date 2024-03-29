import torch 
import torchvision.datasets as dsets 
import torchvision.transforms as transforms 
import torch.nn.init
import numpy as np
import matplotlib.pyplot as plt
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정 
torch.manual_seed(777) 

# GPU 사용 가능일 경우 랜덤 시드 고정 
if device == 'cuda': 
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001 
training_epochs = 15 
batch_size = 100

# MNIST dataset 
mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=True) 

mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train, 
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        # Conv -> (?, 28, 28, 32)
        # Pool -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        # Conv ->(?, 14, 14, 64)
        # Pool ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)

        # 전결합층을 위해서 Flatten
        out = self.fc(out)

        return out

# CNN 모델 정의 
model = CNN().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device) # 비용 함수에 소프트맥스 포함 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)

# 결과 출력에 필요한 변수들과 함수들 정의 
num_data = len(data_loader.dataset) 
num_batch = np.ceil(num_data / batch_size)

fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()
# accuracy를 계산하는 함수를 lambda로 정의

start_epoch = 0
for epoch in range(start_epoch+1, training_epochs+1):     
    avg_cost = 0 
    total_batch = len(data_loader)
    loss_arr = []
    acc_arr = []

    for batch, (X, Y) in enumerate (data_loader, 1):# 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device) 
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        acc = fn_acc(hypothesis, Y)
        cost.backward()
        optimizer.step()

        loss_arr += [cost.item()]
        acc_arr += [acc.item()]
        avg_cost += cost / total_batch
        if batch % 100 == 0:
           print('Train: EPOCH %04d/%04d BATCH %04d/%04d LOSS: %.4f ACC %.4f' 
           %(epoch, training_epochs, batch, num_batch, np.mean(loss_arr),
           np.mean(acc_arr)))

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

# 학습을 진행하지 않을 것이므로 torch.no_grad() 
with torch.no_grad(): 
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
    
    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28*28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap="Greys", 
               interpolation="nearest")
    plt.show()
