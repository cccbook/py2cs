# https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html
import random
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)
pi = np.pi

def fseries(f, L=pi, N=128.0):
    t = np.arange(-L, L, L/N)
    q0 = 0.0
    q1 = 1.0
    q2 = N/2
    q3 = N
    f0t = f(q0*t*pi/L)
    f1t = f(q1*t*pi/L)
    f2t = f(q2*t*pi/L)
    f3t = f(q3*t*pi/L)
    F0q = np.fft.fft(f0t)/(2*N)
    F1q = np.fft.fft(f1t)/(2*N)
    F2q = np.fft.fft(f2t)/(2*N)
    F3q = np.fft.fft(f3t)/(2*N)
    print('F0q=', F0q)
    print('F1q=', F1q)
    print('F2q=', F2q)
    print('F3q=', F3q)

    plt.figure()

    plt.subplot(221)
    plt.plot(t,f0t,label="f0t", color="green", linewidth=1)
    plt.plot(t,F0q,label="F0q", color="blue", linewidth=1)

    plt.subplot(222)
    plt.plot(t,f1t,label="f1t", color="green", linewidth=1)
    plt.plot(t,F1q,label="F1q", color="blue", linewidth=1)

    plt.subplot(223)
    plt.plot(t,f2t,label="f2t", color="green", linewidth=1)
    plt.plot(t,F2q,label="F2q", color="blue", linewidth=1)

    plt.subplot(224)
    plt.plot(t,f3t,label="f3t", color="green", linewidth=1)
    plt.plot(t,F3q,label="F3q", color="blue", linewidth=1)
    plt.show()

fseries(np.cos, L=10*pi, N=128)
# fseries(np.cos, L=np.pi, N=128)
