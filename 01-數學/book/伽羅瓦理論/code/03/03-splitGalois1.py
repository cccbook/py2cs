# 導入加羅瓦群計算庫
from sympy import GF

# 定義有限域 F_5（可以用於更簡單的例子）
F5 = sp.FiniteField(5)

# 定義多項式 x^2 - 2 在 F_5 上的分裂
polynomial_f5 = x**2 - 2

# 解多項式在 F_5 上的根
roots_f5 = sp.solve(polynomial_f5, x)

# 顯示根
print("Roots of x^2 - 2 in F_5:", roots_f5)

# 計算分裂域的加羅瓦群
# 這裡我們需要將根視為生成元，並觀察它們的對稱性
# 可以進一步使用群論工具來分析加羅瓦群
