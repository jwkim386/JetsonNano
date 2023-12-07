import time
import numpy as np

matnum = 200
a = np.random.random((matnum,matnum))
result = np.zeros_like(a)
# Program to multiply two matrices using nested loops

st = time.perf_counter()
# iterating by row of A
for i in range(matnum):
    # iterating by column of A
    for j in range(matnum):
        # iterating by rows of A
        for k in range(matnum):
            result[i][j] += a[i][k] * a[k][j]

ed = time.perf_counter()
result1 = np.matmul(a,a)
ed1 = time.perf_counter()

print("time for nested loops: ", ed - st)
print("time for numpy matmul: ", ed1 - ed)