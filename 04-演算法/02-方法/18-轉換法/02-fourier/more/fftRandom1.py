# https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html
import random
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)
pi = np.pi
# freq = random.randint(0,10)
t = np.arange(-10*pi, 10*pi, 0.1*pi)
q1 = 9.0
q2 = 3.0 # q2 = 2.0
ft = np.cos((q1*t)/(2*pi))
gt = np.cos((q2*t)/(2*pi))
# fi = np.fft.ifft(ft)
Fq = np.fft.fft(ft)
Gq = np.fft.fft(gt)
ft2 = np.fft.ifft(Fq)
gt2 = np.fft.ifft(Gq)
# print('ft=', ft)
# print('Fq=', Fq)
# print('ft2=', ft2)

plt.plot(t,ft,label="ft = cos(2pi t)", color="red", linewidth=2)
plt.plot(t,gt,label="gt = cos(2pi 7t)", color="purple", linewidth=2)
plt.plot(t,Fq,label="Fq=fft(ft)", color="blue", linewidth=2)
plt.plot(t,Gq,label="Gq=fft(gt)", color="green", linewidth=2)
# plt.plot(t,ft2,label="ft2=ifft(Fq)", color="green", linewidth=2)
plt.show()

'''
\hat:x[k]=\sum_:n=0^:N-1 e^:-i\frac:2\pi:Nnkx[n] \qquad k = 0,1,\ldots,N-1.

fft(k) = \sum e^:-i (2 pi nk)/N x[n]

N = 200

fft(k) = \sum e^:-i (2 pi nk)/N x[n]

e^:-i (2 pi nk)/N = cos((2 pi nk)/N) + i sin(2 pi nk)/N)

令 t = (2 pi n)/N, k = 3

e^:-i (2 pi nk)/N = cos((2 pi nk)/N) + i sin(2 pi nk)/N)
'''
