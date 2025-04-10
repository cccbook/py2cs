def sort(a):
    for i in range(len(a)):
        for j in range(i):
            if (a[j] > a[i]):
                a[i], a[j] = a[j], a[i]
    return a

print('sort([3, 8, 2, 1, 5]=', sort([3,8,2,1,5]))
