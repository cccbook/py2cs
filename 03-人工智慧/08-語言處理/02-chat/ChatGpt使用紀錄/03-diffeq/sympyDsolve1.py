# 導入 sympy 庫和 dsolve 函數
import sympy
from sympy.abc import y, t
from sympy import dsolve

# 定義一個微分方程，y 為變量，t 為時間
eq = sympy.Eq(y.diff(t), y)

# 設定初始值
ics = {y.subs(t, 0): 1}

# 求解微分方程
solution = dsolve(eq, ics=ics)

# 輸出答案
print(solution)
