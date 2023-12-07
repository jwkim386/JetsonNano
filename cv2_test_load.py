import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import cv2
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)

if device == 'cuda':
    torch.cuda.manual_seed_all(777)

drawing = False
capture = False
ix, iy = -1, -1

def nothing(x):
    pass

def draw_line(event, x, y, flags, param):
    global ix, iy, drawing, capture, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (ix,iy), (x,y), (255,255,255), thickness=thick)
            ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        frame = cv2.resize(img, dsize=(28,28), interpolation=cv2.INTER_AREA)
        capture = True



PATH = "/home/jwkim/PycharmProjects/TestWorks/test.pickle"
learning_rate = 0.001
training_epochs = 5
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.fc1 = torch.nn.Linear(4*4*128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob)
        )

        self.fc2 = torch.nn.Linear(625,10,bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1) # Flatten output for fc
        out = self.layer4(out)
        out = self.fc2(out)
        return out

model = CNN().to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

img = np.zeros((512,512,1), np.uint8)
frame = np.zeros((28,28,1), np.uint8)
cv2.namedWindow('image')
cv2.createTrackbar('Thickness', 'image', 5, 10, nothing)
cv2.setMouseCallback('image', draw_line)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    thick = cv2.getTrackbarPos('Thickness', 'image')
    if capture == True:
        capture = False
        img = np.zeros((512,512,1), np.uint8)
        test_frame = torch.from_numpy(frame)
        test_frame = test_frame.type(torch.FloatTensor)
        with torch.no_grad():
            X_test = test_frame.view(1,1,28,28).to(device)
            prediction = model(X_test)
            #print(prediction)
            prediction = torch.argmax(prediction)
            prediction = prediction.to(torch.device("cpu"))
            prediction = prediction.numpy()
            print(prediction)

cv2.destroyAllWindows()