from sympy import *
x,y = symbols('x y')
print(solve([x + y-10,2*x+y-16],[x,y]))
