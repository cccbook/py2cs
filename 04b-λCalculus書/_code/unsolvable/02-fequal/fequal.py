def fequal(A, B):
    a = A()
    b = B()
    return a==b

def f1():
    return 10 * 10

def f2():
    n = 10000
    s = 0
    for _ in range(n):
        for _ in range(n):
            for _ in range(n):
                for _ in range(n):
                    s = s+1

print('fequal(f1,f2)=', fequal(f1,f2))


