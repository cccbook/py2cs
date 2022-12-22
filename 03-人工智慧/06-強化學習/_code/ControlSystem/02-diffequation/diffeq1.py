# https://docs.sympy.org/1.5.1/modules/solvers/ode.html
from sympy import Function, dsolve, Eq, Derivative, sin, cos, symbols
from sympy.abc import x
f = Function('f')
sol = dsolve(Derivative(f(x), x, x) + 9*f(x), f(x))
print('dsolve(Derivative(f(x), x, x) + 9*f(x), f(x))=', sol.doit())
