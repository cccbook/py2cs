'''
牛頓逼近法求叠代式及應用 -- https://www.itread01.com/articles/1490047226.html

切線方程式 y  = f(x0) + f'(x0)(x-x0)

y = 0 時 0 = f(x0) + f'(x0)(x-x0)

=> -f(x0) = f'(x0)(x-x0)

=> -f(x0)/f'(x0) = x-x0

=> x0-f(x0)/f'(x0) = x
'''

dx = 0.000001
dmin = 0.000000001

def f(x):
    return x*x - 3

def df(f, x):
    return (f(x+dx)-f(x))/dx

def nextx(f, x):
    return x-f(x)/df(f, x)

def iterate(x):
    xLast = x
    while (True):
        print("x=", x)
        x = nextx(f, x)
        if (abs(xLast-x) < dmin): break
        xLast = x
    return x

print('sqrt(3) = ', iterate(1))
