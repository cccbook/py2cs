# https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html
import random
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)
pi = np.pi

def fseries(f, a=-pi, b=pi, N=128):
    L = (b-a)/2.0
    t = np.arange(a, b, L/N)
    q = random.randint(0, N)
    print('random:q=', q)
    ft = f(q*t*pi/L)
    Fq = np.fft.fft(ft)/(2*N)
    # print('Fq=', Fq)
    # plt.plot(t,ft,label="f0t", color="green", linewidth=1)
    # plt.plot(t,Fq,label="F0q", color="blue", linewidth=1)
    # plt.show()
    return Fq

def guess(Fq):
    for i in range(len(Fq)):
        if abs(Fq[i]) > 0.01:
            return i
    return -1

Fq = fseries(np.cos, a=2*pi, b=10*pi, N=128)
q = guess(Fq)
print('guess:q=', q)
# fseries(np.cos, L=np.pi, N=128)
