# https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)
t = np.arange(0, 10*np.pi, 0.1*np.pi)
# ft = 10*np.sin(t)
ft = 10*np.cos(t)
ff = np.fft.fft(ft)
fi = np.fft.ifft(ff)
print('ft=', ft)
print('fi=', fi)
print('ft==fi', np.allclose(ft, fi))
