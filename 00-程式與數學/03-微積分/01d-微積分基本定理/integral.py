from numpy import arange
import math
# 積分 integral calculus
def integral(f, a, b, dx = 0.0001):
    area = 0.0
    for x in arange(a, b, dx):
        area += f(x) * dx
    return area


if __name__=="__main__":
    print('integral(sqrt, 0, 1) =', integral(math.sqrt, 0, 1))
    print('integral(sin,  0, pi) =', integral(math.sin, 0, math.pi))
