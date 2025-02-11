import numpy as np
from scipy.optimize import minimize

# 定義目標函數 f(x) = x^2
def objective(x):
    return x**2

# 定義約束條件 g(x) = x - 1 <= 0
def constraint(x):
    return 1 - x

# 初始猜測
x0 = np.array([0.5])

# 定義約束條件
cons = ({'type': 'ineq', 'fun': constraint})

# 求解最優化問題
result = minimize(objective, x0, constraints=cons)

# 顯示結果
print("最優解 x =", result.x)
print("最小值 f(x) =", result.fun)
