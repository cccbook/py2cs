# https://docs.sympy.org/1.5.1/tutorial/calculus.html
from sympy import *
x, y, z = symbols('x y z')
init_printing(use_unicode=True)

print('diff(cos(x), x)=', diff(cos(x), x))
print('diff(exp(x**2), x)=', diff(exp(x**2), x))
