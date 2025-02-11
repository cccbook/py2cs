# https://en.wikipedia.org/wiki/Buffon%27s_needle_problem

from numpy.random import uniform
import numpy as np
t=45; L=36; n=50000
x=uniform(0,t/2,n) # n 個 [0,t/2] 上的隨機數
phi = uniform(0, np.pi, n)
h = sum(x<=L*np.sin(phi)/2)
pi = 2*n*L/(t*h); print(pi)
