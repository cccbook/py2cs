from sympy import *
x,y = symbols('x y')
a,b,c=symbols('a b c')
expr=a*x**2 + b*x + c
s_expr=solve( expr, x)
print(s_expr)
