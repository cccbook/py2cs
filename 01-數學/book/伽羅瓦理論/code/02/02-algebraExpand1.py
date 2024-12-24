import sympy as sp

# 定義符號變數
x = sp.symbols('x')

# 定義多項式
polynomial = x**2 - 2

# 解多項式方程 x^2 - 2 = 0
roots = sp.solve(polynomial, x)

# 顯示解
print("Roots of x^2 - 2 = 0:", roots)

# 計算代數擴張：在這個例子中我們加入根 sqrt(2)
alpha = roots[0]  # sqrt(2)

# 將代數擴張進行數值計算
print("Value of alpha (sqrt(2)):", alpha.evalf())
