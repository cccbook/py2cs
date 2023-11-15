from sympy import *
x, y, z = symbols('x y z')
expr = sin(x)/x
l_expr=limit(expr, x, 0)
print(l_expr)
