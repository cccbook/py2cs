import numpy as np
from cmath import exp
from math import pi

def dft(f):
    N = len(f)
    F = np.array([0+0j]*N)
    for n in range(N):
        for x in range(N):
            F[n] += exp(-1j*2.*pi*x*n/N)*f[x]
    return F

def idft(F):
    N = len(F)
    f = np.array([0+0j]*N)
    for x in range(N):
        for n in range(N):
            f[x] += exp(1j*2*pi*x*n/N)*F[n]
        f[x] /= N
    return f
