import random

def sort(a):
    n = len(a)
    for _ in range(0,n*n*10):
        i=random.randint(0, n-2)
        if (a[i] > a[i+1]):
            a[i], a[i+1] = a[i+1], a[i]
    return a

print('sort([3, 8, 2, 1, 5]=', sort([3,8,2,1,5]))
