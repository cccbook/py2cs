import sympy as sp

# 定義符號變數
x = sp.symbols('x')

# 定義多項式 x^2 - 2
polynomial = x**2 - 2

# 解方程 x^2 - 2 = 0
roots = sp.solve(polynomial, x)

# 顯示解（即分裂域中的元素）
print("Roots of x^2 - 2 = 0:", roots)

# 構造分裂域
split_field = sp.FiniteField(roots[0])

print("Split field containing the roots:", split_field)
