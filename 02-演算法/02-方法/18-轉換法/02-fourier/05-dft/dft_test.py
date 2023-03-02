from dft import *
import numpy as np

N = 200
# N = 8
t = np.arange(-pi, pi, pi/N)
q = 1
f = np.cos
ft = f(q*t)
Fq = dft(ft)
ft2 = idft(Fq)
print('ft=', ft)
print('Fq=', Fq)
print('ft2=', ft2)
np.testing.assert_array_almost_equal(ft, ft2)



