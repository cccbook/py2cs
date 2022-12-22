# https://docs.sympy.org/1.5.1/modules/solvers/ode.html
from sympy import Function, dsolve, Eq, Derivative, sin, cos, symbols
from sympy.abc import t
C, R = symbols('C R')
V = Function('V')
sol = dsolve(C*Derivative(V(t), t) + V(t)/R, 0, hint="best")
print('V(t)=', sol.doit())

'''
維基百科的解：V(t) = V0 e^:-t/(RC)

sympy 求得的解： V(t)= Eq(V(t), C1*exp(-t/(C*R)))
令 C1 = V0 就得到相同結果。
'''
