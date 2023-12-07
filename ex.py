"""factorial done recursively and iteratively"""

def fact1(n):
    ans = 1
    for i in range(2,n):
        ans = ans * i
    return ans * n

def fact2(n):
    if n < 1:
        return 1
    else:
        return n * fact2(n - 1)
