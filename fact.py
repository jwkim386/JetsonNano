#!/usr/bin/python3

def fact(x):
    """Returns the factorial of its argument,
    assumed to be a positive integer"""
    if x == 0:
        return 1
    return x * fact(x - 1)

print
print('N fact(N)')
print("---------")

for n in range(10):
    print(n, fact(n))
