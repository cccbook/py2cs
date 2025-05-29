from sympy import symbols, simplify
from sympy.geometry import Point, Segment

# 定義點
x, y = symbols('x y')
A = Point(0, 0)
B = Point(2, 0)
C = Point(x, y)

# BC 的中點 D
D = Segment(B, C).midpoint

# 建立線段 AD
AD = Segment(A, D)

# 驗證 D 是否在線段 AD 上（方法一）
is_on_segment = AD.contains(D)

# 顯示結果
print(f"中點 D 的座標為: {D}")
print(f"D 是否在線段 AD 上？{is_on_segment}")
