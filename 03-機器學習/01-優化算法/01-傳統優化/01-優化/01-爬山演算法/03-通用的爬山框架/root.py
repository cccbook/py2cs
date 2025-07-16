
from hillClimbing import hillClimbing
from random import uniform
from math import sqrt 

def complexNorm(x):
    return sqrt(x.real**2+x.imag**2)

def polynomialEval(a, x):
    n = len(a)
    r = 0
    for i in range(n):
        r += a[i]*(x**i)
    return r

def polyHeight(a, x):
    return -1.0*complexNorm(polynomialEval(a, x))

def complexNeighbor(x, h=0.001):
    dx = uniform(-h, h)*1+uniform(-h, h)*1j
    return x+dx

def polynomialRoot(a, h=0.001):
    return hillClimbing(
        0+0j, 
        lambda x:polyHeight(a,x), 
        complexNeighbor
    )

print(f'polynomial_eval(x*2-2x+1)=', polynomialEval([1,-2,1], 1.0+0j))

print(f'polynomial_root(x*2+0x+1)=', polynomialRoot([1,0,1]))
print(f'polynomial_root(x**4-3x**2-4)=', polynomialRoot([-4, 0, -3, 0, 1]))
