# https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html
import random
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)
f = np.random.randint(0,10,2)
t = np.arange(-10*np.pi, 10*np.pi, 0.1*np.pi)
# ft = 10*np.sin(t)
ft = 10*np.cos(f[0]*t) # +5*np.cos(f[1]*t)
fi = np.fft.fft(ft)
print('ft=', ft)
print('fi=', fi)
print('f=', f)
plt.plot(t,ft,label="$y = 10 sin(f x)$", color="red", linewidth=2)
plt.plot(t,fi,label="ifft: y$", color="blue", linewidth=2)
plt.show()
# 要猜出 f[] 只要把 fi 中不是零的數值 q/2pi 就會得到結果了。