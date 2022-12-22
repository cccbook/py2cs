from sympy import *
x, y = symbols('x y')

i1 = Integral(x*y*exp(-x), (x, 0, oo))
print('Integral(x*y*exp(-x), (x, 0, oo))=', i1.doit())
