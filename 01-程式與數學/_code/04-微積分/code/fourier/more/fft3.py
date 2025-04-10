# https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)
t = np.arange(0, 10*np.pi, 0.1*np.pi)
# ft = 10*np.sin(t)
ft = 10*np.cos(t)+5*np.cos(10*t)
fi = np.fft.ifft(ft)
print('ft=', ft)
print('fi=', fi)
plt.plot(t,ft,label="$10 sin(x)$", color="red", linewidth=2)
plt.plot(t,fi,label="ifft:10sin(x)$", color="blue", linewidth=2)

plt.show()
