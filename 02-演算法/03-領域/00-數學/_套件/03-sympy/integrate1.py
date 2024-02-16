from sympy import *
x,y = symbols('x y')
expr=exp(x)*sin(x) + exp(x)*cos(x)
i_expr=integrate(expr,x)
print(i_expr)
