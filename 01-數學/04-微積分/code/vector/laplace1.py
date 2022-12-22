from vfield import *
from numpy import linalg as LA

def f(p):
    return 1.0/LA.norm(p)

p = np.array([1.,2.,3.])
print(laplace(f, p))
