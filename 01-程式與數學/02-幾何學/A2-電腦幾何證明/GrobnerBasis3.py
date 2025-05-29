from sympy import symbols, Eq, groebner, solve

# 定義符號
x, y, a, b = symbols('x y a b')

# 三條中線的直線方程（用兩點式 → 轉成一般式）

# AD：從 A(0,0) 到 D(a,b)
eq1 = Eq(y * a - x * b, 0)  # 交叉乘法消去分母

# BE：從 B(2a,0) 到 E(0,b)
eq2 = Eq((y - 0)*(0 - 2*a) - (x - 2*a)*(b - 0), 0)

# CF：從 C(0,2b) 到 F(a,0)
eq3 = Eq((y - 2*b)*(a - 0) - (x - 0)*(0 - 2*b), 0)

# 將 eq1, eq2 轉為多項式形式（左邊為多項式）
poly1 = eq1.lhs
poly2 = eq2.lhs
poly3 = eq3.lhs

# 建立 Gröbner Basis
G = groebner([poly1, poly2], x, y, order='lex')

# 解出交點 (x, y)
sol = solve(G, x, y)
print(f"中線 AD 與 BE 的交點為: {sol}")

# 驗證此點是否也在第三條中線 CF 上
x_sol, y_sol = sol[x], sol[y]
on_third = poly3.subs({x: x_sol, y: y_sol}).simplify()
print(f"是否也在第三條中線 CF 上？", on_third == 0)
