from sympy import symbols, groebner

x, y = symbols('x y')
G = groebner([x**2 + y**2 - 1, x - y], x, y)
for g in G:
    print(g)
