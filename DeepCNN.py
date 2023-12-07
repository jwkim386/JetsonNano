import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
# device = torch.device("cpu")
print("The training device is ", device)

# for reproducibility
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# hyperparameters
learning_rate = 0.001
training_ephochs = 30
batch_size = 50

# MNIST dataset

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform = transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform = transforms.ToTensor(),
                         download=True)

# dataset loader
data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

data_loader1 = DataLoader(dataset=mnist_test,
                         batch_size=batch_size,
                         shuffle=False,
                         drop_last=True)

# DeepCNN Network

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
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)

        # 전결합층을 위해서 Flatten
        x = self.fc(x)

        return x

# MNIST data image of shape 28*28 = 784
model = CNN().to(device)

# Cost function and Optimizer Define
criterion = torch.nn.CrossEntropyLoss().to(device)

# Internally including Softmax Fct.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)

for epoch in range(training_ephochs):
    avg_cost = 0

    for X, Y in data_loader:
        # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch+1, avg_cost))

print('Learning finished')

torch.save(model.state_dict(), 'CNNmodel_state_dict.pt')
torch.save(model, 'CNNmodel.pt')

# 학습을 진행하지 않을 것이므로 torch.no_grad()
with torch.no_grad(): # no perform the gradient
    count = 0
    accuracy = 0.0
    for X_test, Y_test in data_loader1:
        # X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
        # Y_test = mnist_test.test_labels.to(device)
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)

        prediction = model(X_test)
        correct_predition = torch.argmax(prediction, 1) == Y_test
        accuracy += correct_predition.float().mean()
        count +=1
    accuracy /= count
    print('Accuracy: ', accuracy.item())

    # MNIST prediction for one random data
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r+1].view(1, 1, 28, 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r+1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r+1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()