def h(f, input):
    f(input)
    return True

def f1(n):
    return n * n

def f2(n):
    s = 0
    for _ in range(n):
        for _ in range(n):
            for _ in range(n):
                for _ in range(n):
                    s = s+1


print('h(f1,3)=', h(f1, 3))
print('h(f2,10)=', h(f2, 10))
# print('h(f2,1000)=', h(f2, 1000))
