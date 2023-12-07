import torch
from tkinter import *
from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt
import numpy as np

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

    # @torch.cuda.amp.autocast()
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)

        # 전결합층을 위해서 Flatten
        x = self.fc(x)

        return x

class main:
    def __init__(self, master):
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if USE_CUDA else "cpu")
        # device = torch.device("cpu")
        print("The training device is ", self.device)

        # for reproducibility
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(777)
        else:
            torch.manual_seed(777)

        print('Read MNIST model parameters from file ...')
        self.model = torch.load('CNNmodel.pt')
        self.model.eval() #preparing for running the MNIST model
        print('Model parameter loading to GPU ...')

        # dummy running the model for removing delay. It's tricky.
        xx = torch.FloatTensor(np.random.randn(28, 28))
        self.model(xx.view(1, 1, 28, 28).float().to(self.device))
        print('Ready for running ...')

        self.master = master
        self.color_fg = 'white'
        self.color_bg = 'black'
        self.old_x = None
        self.old_y = None
        self.penwidth = 30
        self.drawWidgets()
        self.c.bind('<B1-Motion>', self.paint)  # drawing the line
        self.c.bind('<ButtonRelease-1>', self.reset)
        # PIL
        self.image1 = Image.new("L", (560, 560), color=(0))
        self.draw = ImageDraw.Draw(self.image1)

    def paint(self, e):

        if self.old_x and self.old_y:
#            self.c.create_line(self.old_x, self.old_y, e.x, e.y, width=self.penwidth, fill=self.color_fg,
#                               capstyle=ROUND, smooth=True)
            self.c.create_line(self.old_x, self.old_y, e.x, e.y, fill=self.color_fg, width=self.penwidth, tags='currentline')
            self.draw.line([(self.old_x, self.old_y), (e.x, e.y)], fill=255, width=int(self.penwidth))
        self.old_x = e.x
        self.old_y = e.y

    def reset(self, e):  # resetting or cleaning the canvas
        self.old_x = None
        self.old_y = None

    def changeW(self, e):  # change Width of pen through slider
        self.penwidth = e

    def clear(self):
        self.c.delete(ALL)
        self.image1 = Image.new("L", (560, 560), color=(0))
        self.draw = ImageDraw.Draw(self.image1)

    def run(self):
        # 학습을 진행하지 않을 것이므로 torch.no_grad()
        # print('Start prediction')
        with torch.no_grad():  # no perform the gradient
            img = torch.FloatTensor(np.array(self.image1.resize((28, 28), 3)))
            # Use Image.NEAREST (0), Image.LANCZOS (1), Image.BILINEAR (2), Image.BICUBIC (3), Image.BOX (4) or Image.HAMMING (5)
            X_single_data = img.view(1, 1, 28, 28).float().to(self.device)
            prediction = self.model(X_single_data)
            # print('Prediction = ', torch.argmax(prediction, 1).item())
        # plt.imshow(np.array(X_single_data.view(28,28).to('cpu')), cmap='gray')
        # plt.show()
        Label(self.Box1, text='  ['+str(torch.argmax(prediction, 1).item())+']  ', font='arial 50',
              bg='yellowgreen', fg='red').grid(row=0, column=0)

    def drawWidgets(self):
        self.message = Frame(self.master, padx=0, pady=0)
        Label(self.message, text='MNIST Prediction Test', font='arial 50', borderwidth=5, relief=RIDGE,
              width=26, height = 1, bg='yellow', fg='blue').grid(row=0, column=0)
        self.message.place(x=12, y=6, width=1000, height=100)

        self.controls = Frame(self.master, padx=5, pady=5)
        Label(self.controls, text='Pen Width: ', font='arial 18').grid(row=0, column=0)
        self.slider = Scale(self.controls, from_=20, to=40, resolution=2, width=30, length=200,
                            command=self.changeW, orient=HORIZONTAL)
        self.slider.set(self.penwidth)
        self.slider.grid(row=0, column=1, ipadx=10)
        self.controls.place(x=100, y=680, width=560, height=100)

        self.Box1 = Frame(self.master, padx=0, pady=0)
        Label(self.Box1, text='Result', font='arial 33', width = 8, height = 13, borderwidth=5, relief=RIDGE,
              bg='yellowgreen', fg='red').grid(row=0, column=0)
        self.Box1.place(x=566, y=97, width=220, height=660)

        self.c = Canvas(self.master, width=560, height=560, bg=self.color_bg)
        self.c.place(x=0, y=100, width=560, height=560)

        # Button
        self.btn1 = Button(self.master, text='RUN', font='arial 35', command=self.run)
        self.btn2 = Button(self.master, text='CLEAR', font='arial 35', command=self.clear)
        self.btn3 = Button(self.master, text='EXIT', font='arial 35', command=self.master.destroy)
        self.btn1.place(x=780, y=100, width=220, height=220)
        self.btn2.place(x=780, y=320, width=220, height=220)
        self.btn3.place(x=780, y=540, width=220, height=220)

if __name__ == '__main__':

    root = Tk()
    main(root)
    root.geometry("1000x760+510+100")
    root.resizable(False, False)

    root.title("MNIST Prediction")
    root.mainloop()