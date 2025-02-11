# https://docs.sympy.org/latest/modules/polys/domainsref.html
from sympy import FiniteField

print('FiniteField=', FiniteField)
# 定義有限域 F_5
F5 = FiniteField(5)

# 在 F_5 上構造代數擴張：x^2 + 1
x = F5.gen()  # 獲取 F_5 的生成元素
polynomial = x**2 + 1

# 解方程 x^2 + 1 = 0 在 F_5 上的根
roots_F5 = sp.solve(polynomial, x)

# 顯示根
print("Roots of x^2 + 1 = 0 in F_5:", roots_F5)
