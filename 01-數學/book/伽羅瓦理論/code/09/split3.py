import sympy
from sympy import symbols, Eq, solve

# 定義變數
x = symbols('x')

# 假設我們要解決的三等分角問題
# 這是來自於 x^3 - 3x + 1 = 0 的一個例子
# 這裡的解其實與三等分角的問題相關
equation = Eq(x**3 - 3*x + 1, 0)

# 使用 sympy 求解該方程
roots = solve(equation, x)
print("多項式的解：", roots)

# 檢查是否可以通過代數式表示
if len(roots) > 2:
    print("該多項式具有複數根，因此該三等分角問題無法通過簡單的圓規和直尺作圖解決。")
else:
    print("該問題有可能通過圓規和直尺作圖解決。")
