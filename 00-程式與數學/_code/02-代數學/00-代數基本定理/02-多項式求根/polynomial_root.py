from random import uniform
from math import sqrt 

def complex_norm(x):
    return sqrt(x.real**2+x.imag**2)

def polynomial_eval(a, x):
    n = len(a)
    r = 0
    for i in range(n):
        r += a[i]*(x**i)
    return r

def polynomial_root(a, h=0.001):
    x = 0+0j
    fail = 0
    while True:
        fx = polynomial_eval(a, x)
        if complex_norm(fx)<0.0001:
            print('success:x=', x, 'fx=', fx)
            return x
        dx = uniform(-h, h)*1+uniform(-h, h)*1j
        fx1 = polynomial_eval(a, x+dx)
        if complex_norm(fx1)<complex_norm(fx):
            x = x+dx
            fail = 0
        else:
            fail += 1
            if fail > 10000:
                print('fail:x=', x, 'fx=', fx)
                return None

print(f'polynomial_eval(x*2-2x+1)=', polynomial_eval([1,-2,1], 1.0+0j))

print(f'polynomial_root(x*2+0x+1)=', polynomial_root([1,0,1]))
print(f'polynomial_root(x**4-3x**2-4)=', polynomial_root([-4, 0, -3, 0, 1]))
