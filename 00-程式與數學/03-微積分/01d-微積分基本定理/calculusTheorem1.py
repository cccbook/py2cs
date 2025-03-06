from diff import diff
from integral import integral
from math import sin, cos

def calculusTheorem1(f, x):
    rx = diff(lambda x:integral(f, 0, x), x)
    fx = f(x)
    print('rx=', rx, 'fx=', fx)

x = 3.72
calculusTheorem1(sin, x)
