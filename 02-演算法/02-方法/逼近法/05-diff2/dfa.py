from math import sin, pi

def df1(f, x, h=0.00001):
    return (f(x+h)-f(x-h))/(2*h)

def df2(f, x, h=0.00001):
    return (f(x+h)-2*f(x)+f(x-h))/(h**2)

def df3(f, x, h=0.00001):
    return (f(x*2*h)-2*f(x+h)+2*f(x-h)-f(x-2*h))/(2*(h**3))

def df4(f, x, h=0.00001):
    return (f(x+2*h)-4*f(x+h)+6*f(x)-4*f(x-h)+f(x-2*h))/(h**4)
