import numpy as np
from scipy.optimize import minimize
def f(p):
    x, y = p
    return (x-1)**2+(y-2)**2

p=[0,0]
r = minimize(f, p)
print('r=', r)
