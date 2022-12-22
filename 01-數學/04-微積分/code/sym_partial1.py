# https://docs.sympy.org/1.5.1/tutorial/calculus.html
from sympy import *

init_printing(use_unicode=True)

x, y, z = symbols('x y z')
expr = exp(x*y*z)
print('diff(xyz)=', diff(expr, x))
