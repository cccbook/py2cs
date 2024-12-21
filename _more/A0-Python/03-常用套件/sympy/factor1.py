from sympy import *
x,y = symbols('x y')
expr=x**4+x*y+8*x
f_expr=factor(expr)
e_expr=expand(f_expr)
print(f_expr)
print(e_expr)
