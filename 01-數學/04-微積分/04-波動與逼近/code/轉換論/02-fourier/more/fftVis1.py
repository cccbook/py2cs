# https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html
import random
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)
pi = np.pi
# freq = random.randint(0,10)
t = np.arange(-pi, pi, 0.01*pi)
q1 = 1.0
q2 = 50.0
q3 = 100.0
f1t = np.cos(q1*t)
f2t = np.cos(q2*t)
f3t = np.cos(q3*t)
F1q = np.fft.fft(f1t)/len(t)
F2q = np.fft.fft(f2t)/len(t)
F3q = np.fft.fft(f3t)/len(t)
# f1 = np.fft.ifft(F1q)
# f2 = np.fft.ifft(F2q)
# f3 = np.fft.ifft(F3q)

plt.plot(t,f1t,label="f1t", color="red", linewidth=1)
plt.plot(t,f2t,label="f2t", color="green", linewidth=1)
plt.plot(t,f3t,label="f3t", color="blue", linewidth=1)
plt.plot(t,F1q,label="F1q", color="#FFFF00", linewidth=1)
plt.plot(t,F2q,label="F2q", color="#FF00FF", linewidth=1)
plt.plot(t,F3q,label="F3q", color="#00FFFF", linewidth=1)
# plt.plot(t,ft2,label="ft2=ifft(Fq)", color="green", linewidth=2)
plt.show()
