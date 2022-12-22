# https://docs.sympy.org/latest/modules/solvers/ode.html

from sympy import Function, dsolve, Eq, Derivative, sin, cos, symbols
from sympy.abc import x
f = Function('f')
r = dsolve(Derivative(f(x), x, x) + 9*f(x), f(x))
print('r=', r) # Eq(f(x), C1*sin(3*x) + C2*cos(3*x))
eq = sin(x)*cos(f(x)) + cos(x)*sin(f(x))*f(x).diff(x)
r = dsolve(eq, hint='1st_exact')
print('r=', r) # [Eq(f(x), -acos(C1/cos(x)) + 2*pi), Eq(f(x), acos(C1/cos(x)))]
r = dsolve(eq, hint='almost_linear')
print('r=', r) # [Eq(f(x), -acos(C1/cos(x)) + 2*pi), Eq(f(x), acos(C1/cos(x)))]
t = symbols('t')
x, y = symbols('x, y', cls=Function)
eq = (Eq(Derivative(x(t),t), 12*t*x(t) + 8*y(t)), Eq(Derivative(y(t),t), 21*x(t) + 7*t*y(t)))
r = dsolve(eq)
print('r=', r) # [Eq(x(t), C1*x0(t) + C2*x0(t)*Integral(8*exp(Integral(7*t, t))*exp(Integral(12*t, t))/x0(t)**2, t)),Eq(y(t), C1*y0(t) + C2*(y0(t)*Integral(8*exp(Integral(7*t, t))*exp(Integral(12*t, t))/x0(t)**2, t) +exp(Integral(7*t, t))*exp(Integral(12*t, t))/x0(t)))]
eq = (Eq(Derivative(x(t),t),x(t)*y(t)*sin(t)), Eq(Derivative(y(t),t),y(t)**2*sin(t)))
dsolve(eq)
print('r=', r) # {Eq(x(t), -exp(C1)/(C2*exp(C1) - cos(t))), Eq(y(t), -1/(C1 - cos(t)))}
