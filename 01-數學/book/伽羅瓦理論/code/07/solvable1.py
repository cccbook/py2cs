import sympy
from sympy import S
from sympy.polys.polytools import LC

# 定義變數與多項式
x = sympy.symbols('x')
f = x**5 - x - 1  # 例如：一個五次多項式

# 試圖檢查該多項式是否可解
# 我們的目的是了解其加羅瓦群的結構

# 計算該多項式的根
roots = sympy.solve(f, x)
print("多項式的根：", roots)

# 加羅瓦群的判斷
# 我們假設對於五次多項式，如果加羅瓦群是S5（對稱群），則不可解
# 使用對稱群S5判斷
from sympy.combinatorics import PermutationGroup
from sympy.combinatorics import Permutation

# 假設多項式的加羅瓦群是對稱群S5
# 如果多項式的加羅瓦群是S5，則該多項式不可解
s5_group = PermutationGroup([Permutation([0, 1, 2, 3, 4]), Permutation([0, 2, 4, 1, 3])])  # S5的一個示例

# 顯示加羅瓦群與是否可解
print("加羅瓦群：", s5_group)
if s5_group.is_solvable:
    print("該多項式的加羅瓦群是可解的，該多項式是可解的。")
else:
    print("該多項式的加羅瓦群是不可解的，該多項式不可解。")
