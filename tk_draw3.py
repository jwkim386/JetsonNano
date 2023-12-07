from tkinter import *
from PIL import Image, ImageDraw
import PIL

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
#        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_AREA)
#         cv2.imshow('img',img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         #img = Image.open(filename)
#         test_frame = transforms.ToTensor()(img)
#         test_frame = test_frame.type(torch.FloatTensor)
#         with torch.no_grad():
#             X_test = test_frame.view(1, 1, 28, 28).to(device)
#             prediction = model(X_test)
#             # print(prediction)
#             prediction = torch.argmax(prediction)
#             prediction = prediction.to(torch.device("cpu"))
#             prediction = prediction.numpy()
#             print(prediction)
#             result_value = prediction
#             result_label.configure(text = result_value)

def mouseScroll(event):
    global wp
    wp = scale_bar.get()


def clearCanvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill='black')

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

window.mainloop()