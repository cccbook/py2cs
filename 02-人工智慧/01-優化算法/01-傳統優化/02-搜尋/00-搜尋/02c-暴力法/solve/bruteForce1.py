from numpy import arange
import math

def f(x) :
    # return x*x-4*x+1
    return math.sin(x*x+2*x)/x*x*x 

for x in arange(-100, 100, 0.001):
    if abs(f(x)) < 0.001:
        print("x=", x, " f(x)=", f(x)) 
