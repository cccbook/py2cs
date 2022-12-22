from sympy import *
x,y = symbols('x y')
expr=(2*x)**3*(-5*x*y**2)
s_expr=simplify(expr)
print(s_expr)
