from sympy.logic.boolalg import Or, And, Not
from sympy.abc import A, B, C

# 定義邏輯表達式
expr = Or(And(A, B), And(Not(A), C))

# 簡化邏輯表達式
simplified_expr = expr.simplify()
print("簡化後的邏輯表達式:", simplified_expr)