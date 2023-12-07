import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) #이미지 픽셀값 정규화
]) 

#CIFAR10 dataset 가져오기
trainset = torchvision.datasets.CIFAR10(root = './data',
					    train = True,
					    download = True,
					    transform=transform)
testset = torchvision.datasets.CIFAR10(root = './data',
					    train = False,
					    download = True,
					    transform=transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck')

trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
testloader = DataLoader(testset, batch_size=8, shuffle=False)

dataiter = iter(trainloader)
images, labels = dataiter.next()
