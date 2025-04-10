import numpy as np

def DFT(f):
    N = len(f)
    F = [0+0j]*N
    for x in range(N):
        for n in range(N):
            exp = np.exp(1j*(-2.0*np.pi*x)/N*n)
            F[n] += f[x]*exp
    return F


def iDFT(F):
    N = len(F)
    f = [0+0j]*N
    for n in range(N):
        for x in range(N):
            exp = np.exp(1j*(2.0*np.pi*x)/N*n)
            f[x] += F[n]*exp/N
    return f

x = np.arange(0, np.pi, np.pi/8) # np.arange(0, 10*np.pi, np.pi/8)
f = [np.exp(1j*xi) for xi in x]

np.set_printoptions(precision=4, floatmode='fixed')
print('f=', f)
F = DFT(f)
print('F=DFT(f)=', F)
f2 = iDFT(F)
print('f2=iDFT(F)=', f2)
print('allclose(f, f2)=', np.allclose(f, f2))
