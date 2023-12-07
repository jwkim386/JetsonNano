from tkinter import *
from tkinter import colorchooser
from PIL import Image, ImageDraw, ImageGrab
import matplotlib.pyplot as plt
import numpy as np

class main:
    def __init__(self, master):
        self.master = master
        self.color_fg = 'white'
        self.color_bg = 'black'
        self.old_x = None
        self.old_y = None
        self.penwidth = 20
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

    def run(self):
        plt.imshow(np.array(self.image1), cmap='gray')
        plt.show()

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
    root.geometry("1000x760+300+150")
    root.resizable(False, False)

    root.title("MNIST Prediction")
    root.mainloop()