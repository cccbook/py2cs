import numpy as np
from cmath import exp
from math import pi

def dft(f):
    N = len(f)
    F = np.array([0+0j]*N)
    for k in range(N):
        for n in range(N):
            F[k] += exp(-1j*2.*pi*n*k/N)*f[n]
    return F

def idft(F):
    N = len(F)
    f = np.array([0+0j]*N)
    for n in range(N):
        for k in range(N):
            f[n] += exp(1j*2*pi*n*k/N)*F[k]
        f[n] /= N
    return f
