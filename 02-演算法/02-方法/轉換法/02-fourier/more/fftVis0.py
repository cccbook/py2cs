# https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html
import random
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)
pi = np.pi
N = 100.0
# freq = random.randint(0,10)
t = np.arange(-pi, pi, pi/N)
q1 = 1.0
q2 = N/2
q3 = N
f1t = np.cos(q1*t)
f2t = np.cos(q2*t)
f3t = np.cos(q3*t)
F1q = np.fft.fft(f1t)/(2*N)
F2q = np.fft.fft(f2t)/(2*N)
F3q = np.fft.fft(f3t)/(2*N)
print('F1q=', F1q)
print('F2q=', F2q)
print('F3q=', F3q)
# f1 = np.fft.ifft(F1q)
# f2 = np.fft.ifft(F2q)
# f3 = np.fft.ifft(F3q)

plt.figure()

plt.subplot(311)
plt.plot(t,f1t,label="f1t", color="red", linewidth=1)
plt.plot(t,F1q,label="F1q", color="#FFFF00", linewidth=1)
plt.subplot(312)
plt.plot(t,f2t,label="f2t", color="green", linewidth=1)
plt.plot(t,F2q,label="F2q", color="#FF00FF", linewidth=1)
plt.subplot(313)
plt.plot(t,f3t,label="f3t", color="blue", linewidth=1)
plt.plot(t,F3q,label="F3q", color="#00FFFF", linewidth=1)
# plt.plot(t,ft2,label="ft2=ifft(Fq)", color="green", linewidth=2)
plt.show()
