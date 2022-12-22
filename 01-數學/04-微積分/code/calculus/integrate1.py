# 參考 https://en.wikipedia.org/wiki/Integral
from math import *

def integrate(f, a, b, h=0.001):
    area = 0
    x = a
    while (x < b):
        area += f(x)*h
        x += h
    return area

print('integrate(sin, 0, pi)=', integrate(sin, 0, pi))
