# https://www.freecodecamp.org/news/calculate-definite-indefinite-integrals-in-python/

from scipy.integrate import tplquad

def integrand(z, y, x):
	return z*(x+y+z)
    
print(tplquad(integrand, 0, 1, 4, 5, 0, 1))