import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x) # cos함수

# matplotlib을 이용해 점들 그리기 
# plt.plot(x, y_sin, 'oy')
# plt.plot(x, y_cos, '+g')
# plt.xlabel('x axis label')
# plt.ylabel('y axis label')
# plt.title('Sine and Cosine')
# plt.legend(['Sine', 'Consine'])

plt.subplot(231)
plt.plot(x, y_sin, 'or')
plt.title('Sine')
plt.subplot(236)
plt.plot(x, y_cos, '+g')
plt.title('Cosine')

plt.show()
