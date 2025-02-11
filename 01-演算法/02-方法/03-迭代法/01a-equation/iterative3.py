f1 = lambda x: 3 / x
f2 = lambda x: x - 1 / 4 * (x * x - 3)
f3 = lambda x: 1 / 2 * (x + 3 / x)

x1 = x2 = x3 = 1

for i in range(20):
    x1, x2, x3 = f1(x1), f2(x2), f3(x3)
    print('x1:', x1, 'x2', x2, 'x3', x3)
