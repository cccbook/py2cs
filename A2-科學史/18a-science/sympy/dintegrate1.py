from sympy import *
x,y = symbols('x y')
expr=sin(x**2)
i_expr=integrate(expr, (x, -oo, oo))
print(i_expr)
