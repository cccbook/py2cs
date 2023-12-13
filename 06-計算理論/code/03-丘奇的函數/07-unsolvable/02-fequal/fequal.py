def fequal(f1, f2, n):
    a = f1(n)
    b = f2(n)
    return a==b

def f1(n):
    return n * n

def f2(n):
    s = 0
    for _ in range(n):
        for _ in range(n):
            for _ in range(n):
                for _ in range(n):
                    s = s+1


print('fequal(f1,f2,3)=', fequal(f1, f2, 3))
print('fequal(f1,f2,1000)=', fequal(f1, f2, 1000))


