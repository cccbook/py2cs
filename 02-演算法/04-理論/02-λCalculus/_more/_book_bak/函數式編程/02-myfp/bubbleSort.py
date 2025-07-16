import fp

def swap(a, i, j):
    a[i],a[j] = a[j], a[i]

def bubbleSort(a):
    n = len(a)
    fp.each(range(0,n), lambda i: # for i in range(0,n):
        fp.each(range(0,i), lambda j: # for j in range(0,i):
             swap(a,i,j) if a[j]>a[i] else None
        )
    )
    return a

a = [3,7,2,6,8,4]
bubbleSort(a)
print(a)
