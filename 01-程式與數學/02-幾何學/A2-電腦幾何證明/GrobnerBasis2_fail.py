from sympy import symbols, groebner, simplify

x, y = symbols('x y')

# 幾何條件：AC^2 = BC^2
eq1 = x**2 + y**2 - ((x - 2)**2 + y**2)

# 結論：x - 1 = 0
eq2 = x - 1

# 建立 Gröbner Basis
G = groebner([eq1], x, y, order='lex')

# 簡化 eq2 看是否能被 G 簡化為 0
reduced_eq2 = simplify(eq2.reduce(G, x, y)[1])

print(f"Grobner Basis: {[g for g in G]}")
print(f"結論 x - 1 對 G 的簡化結果: {reduced_eq2}")
