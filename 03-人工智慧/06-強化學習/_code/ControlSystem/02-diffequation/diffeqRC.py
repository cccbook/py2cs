# https://docs.sympy.org/1.5.1/modules/solvers/ode.html
from sympy import Function, dsolve, Eq, Derivative, sin, cos, symbols
from sympy.abc import t
V = Function('V')
C = 1.0
R = 1.0
sol = dsolve(C*Derivative(V(t), t) + V(t)/R, 0)
print('V(t)=', sol.doit())


# C:\frac  :dV:dt+:\frac  :V:R=0