import sympy as sp
A = sp.Matrix(2,2,sp.symbols('a:d'))
B = A.inv(); print(B)