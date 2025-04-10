# https://www.freecodecamp.org/news/calculate-definite-indefinite-integrals-in-python/
from sympy import Symbol, integrate, sin

x = Symbol('x')
print(integrate(x**2, x))
print(integrate(sin(x), x))