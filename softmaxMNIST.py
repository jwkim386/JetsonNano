import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("The training device is ", device)

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# hyperparameters
training_ephochs = 15
batch_size = 100

# MNIST dataset

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform = transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=True,
                         transform = transforms.ToTensor(),
                         download=True)

# dataset loader
data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

# MNIST data image of shape 28*28 = 784
# linear = nn.Linear(784, 10, bias=True).to(device)
linear = nn.Sequential(
	nn.Linear(784, 256, bias=True), # input_layer = 2, hidden_layer1 = 10 
	# nn.Sigmoid(), 
	nn.Linear(256, 100, bias=True), # hidden_layer1 = 10, hidden_layer2 = 10 
	# nn.Sigmoid(), 
	nn.Linear(100, 10, bias=True), # hidden_layer2 = 10, hidden_layer3 = 10 
	# nn.Sigmoid(), 
	# nn.Linear(4, 1, bias=True), # hidden_layer3 = 10, output_layer = 1 
	# nn.Sigmoid() 
	).to(device)
# Cost function and Optimizer Define
criterion = nn.CrossEntropyLoss().to(device)

# Internally including Softmax Fct.
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

for epoch in range(training_ephochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        X = X.view(-1, 28*28).to(device)
        Y = Y.to(device)
        optimizer.zero_grad()

        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:','%04d' % (epoch+1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')

with torch.no_grad(): # no perform the gradient
    X_test = mnist_test.test_data.view(-1, 28*28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = linear(X_test)
    correct_predition = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_predition.float().mean()
    print('Accuracy: ', accuracy.item())

    # MNIST prediction for one random data
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r+1].view(-1, 28*28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r+1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r+1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()