# 來源 -- https://docs.sympy.org/latest/modules/plotting.html
from sympy import symbols
from sympy.plotting import plot
x = symbols('x')
p1 = plot(x*x, show=False)
p2 = plot(x, show=False)
p1.append(p2[0])
p1.show()
