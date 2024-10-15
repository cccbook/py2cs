N = 16 # N = 9 # N = 3
f1 = lambda x: N / x
f2 = lambda x: x - 1 / 4 * (x * x - N)
f3 = lambda x: 1 / 2 * (x + N / x)

x1 = x2 = x3 = 1

for i in range(50):
    x1, x2, x3 = f1(x1), f2(x2), f3(x3)
    print('x1:', x1, 'x2', x2, 'x3', x3)
