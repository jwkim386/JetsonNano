import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("Lenna.png")
# img = np.array(img)
# plt.imshow(img)
# plt.show()
cv2.imshow('Lenna', img)
cv2.waitKey()