import sympy as sp

x = sp.symbols("x")
exp1 = (x+2)**2-(x+1)**2
print('exp1=', exp1)
exp2 = sp.simplify((x+2)**2-(x+1)**2)
print('exp2=', exp2)
