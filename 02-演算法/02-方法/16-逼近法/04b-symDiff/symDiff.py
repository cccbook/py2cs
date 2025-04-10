from sympy import *
x = symbols('x')
print('df1=', diff(x**5, x))
print('df2=', diff(x**5, x, x))
print('df3=', diff(x**5, x, x, x))
print('df4=', diff(x**5, x, x, x, x))
print('df5=', diff(x**5, x, x, x, x, x))


