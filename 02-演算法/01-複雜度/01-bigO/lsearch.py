def find(a, o):
    for i in range(len(a)):
        if a[i] == o:
            return i
    return -1

print(f"find([a, d, x, b, g], x)={find(['a','d','x','b','g'], 'x')}")
