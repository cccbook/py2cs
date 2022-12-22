# https://docs.sympy.org/1.5.1/tutorial/calculus.html
from sympy import *
x, y, z = symbols('x y z')
init_printing(use_unicode=True)

print('integrate(cos(x), x)=', integrate(cos(x), x))
print('integrate(exp(-x), (x, 0, oo))=', integrate(exp(-x), (x, 0, oo)))

expr = Integral(log(x)**2, x)
print('Integral(log(x)**2, x)=', expr.doit())
