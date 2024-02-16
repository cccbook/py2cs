# https://www.freecodecamp.org/news/calculate-definite-indefinite-integrals-in-python/
from scipy.integrate import dblquad

def integrand1(y, x):
	return x*y**2

print(dblquad(integrand1, 0, 1, 2, 4))

def upper_limit_y(x):
	return x**2
    
def lower_limit_y(x):
	return x
    
def integrand2(y, x):
	return x+y
  
print(dblquad(integrand2, 0, 2, lower_limit_y, upper_limit_y))

