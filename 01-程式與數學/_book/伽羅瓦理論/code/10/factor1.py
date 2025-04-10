import sympy
from sympy import symbols, Poly, GF

# 定義變數
x = symbols('x')

# 假設我們需要處理的多項式是 x^4 - 1
poly = x**4 - 1

# 在有限域 F_5 中進行多項式的因式分解
F5 = GF(5)  # 定義有限域 F_5
poly_F5 = Poly(poly, x, domain=F5)

# 因式分解
factorization = poly_F5.factor()
print("多項式在 F_5 上的因式分解：", factorization)

# 檢查多項式是否為不可約
is_irreducible = poly_F5.is_irreducible()
print("該多項式是否不可約：", is_irreducible)

# 如果多項式可分解，則顯示其根
if not is_irreducible:
    roots = poly_F5.roots()
    print("多項式的根：", roots)
