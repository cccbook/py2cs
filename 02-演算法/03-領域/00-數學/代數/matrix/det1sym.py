import sympy as sp
a,b,c = sp.symbols('a b c')
A = sp.Matrix([[a,b,c],[b,c,a],[c,a,b]])
D=sp.det(A); print("D=", D)
