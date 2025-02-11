import sympy as sp

# 定義變數
x = sp.symbols('x')
y = sp.Function('y')

# 定義微分方程 dy/dx = x^2 + 1
ode = sp.Eq(y(x).diff(x), x**2 + 1)

# 求解微分方程
solution = sp.dsolve(ode)

# 顯示結果
print(solution)
