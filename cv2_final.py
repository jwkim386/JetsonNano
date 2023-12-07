import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw
import PIL
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)

if device == 'cuda':
    torch.cuda.manual_seed_all(777)

window = Tk()
window.geometry("600x400")
window.title("MNIST Prediction")

wp = 10
filename = "/home/jwkim/PycharmProjects/TestWorks/image.jpg"
capture = False
result_value = StringVar()
moving = False

def save_image():
    global filename, capture, result_value
    image1.save(filename)
    capture = True
    if capture == True:
        capture = False
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_AREA)
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #img = Image.open(filename)
        test_frame = transforms.ToTensor()(img)
        test_frame = test_frame.type(torch.FloatTensor)
        with torch.no_grad():
            X_test = test_frame.view(1, 1, 28, 28).to(device)
            prediction = model(X_test)
            # print(prediction)
            prediction = torch.argmax(prediction)
            prediction = prediction.to(torch.device("cpu"))
            prediction = prediction.numpy()
            print(prediction)
            result_value = prediction
            result_label.configure(text = result_value)

def mouseScroll(event):
    global wp
    wp = scale_bar.get()


def clearCanvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 256, 256), fill=(0))

def draw_t(event):
    global moving
    moving = True

def draw_f(event):
    global moving
    moving = False
    canvas.old_coords = None

def drawing(event):
    if moving:
        x1, y1 = (event.x), (event.y)
        if canvas.old_coords:
            x2, y2 = canvas.old_coords
            canvas.create_line(x1, y1, x2, y2, fill='white', width=wp)
            draw.line([x1, y1, x2, y2], fill=(255), width=wp)
        canvas.old_coords = x1, y1


canvas = Canvas(window, bg='black', width=256, height=256)
# canvas.pack(expand = YES, fill = BOTH)
canvas.pack()
canvas.bind("<Button-1>", draw_t)
canvas.bind("<ButtonRelease-1>",draw_f)
canvas.bind("<Motion>", drawing)
canvas.old_coords = None
# PIL
image1 = PIL.Image.new("L", (256, 256), color=(0))
draw = ImageDraw.Draw(image1)
# Button
btn1 = Button(window, text='save', command=save_image)
btn2 = Button(window, text='clear', command=clearCanvas)
btn1.place(x=400, y=270)
btn2.place(x=460, y=270)
scale_bar = Scale(window, command=mouseScroll, orient='horizontal', resolution=1, from_=10, to=15)
scale_bar.pack(side=TOP)

#Label
basic_label = Label(window, text = "result = ")
result_label = Label(window)
basic_label.place(x=260, y=330)
result_label.place(x=320, y=330)

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
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob)
        )

        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # Flatten output for fc
        out = self.layer4(out)
        out = self.fc2(out)
        return out


model = CNN().to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

window.mainloop()
