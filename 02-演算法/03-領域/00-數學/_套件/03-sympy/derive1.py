from sympy import *
x,y = symbols('x y')
expr=sin(x)*exp(x)
diff_expr=diff(expr, x)
diff_expr2=diff(expr,x,2)
print(diff_expr)
print(diff_expr2)
