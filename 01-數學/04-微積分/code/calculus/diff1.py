from math import *

def df(f, x, h=0.00001):
    return (f(x+h)-f(x))/h

print('df(sin, pi/4)=', df(sin, pi/4))
