# https://www.freecodecamp.org/news/calculate-definite-indefinite-integrals-in-python/
from scipy.integrate import quad
import numpy as np

def integrand1(x):
	return x**2

def integrand2(x):
	return(x+1)/x**2    

def integrand3(x):
	return np.log(np.sin(x))

def integrand4(x):
  return np.exp(-x)

print(quad(integrand2, 0, 1))
print(quad(integrand2, 1, 2))
print(quad(integrand3, 0, 2))
print(quad(integrand4, 0, np.inf))
