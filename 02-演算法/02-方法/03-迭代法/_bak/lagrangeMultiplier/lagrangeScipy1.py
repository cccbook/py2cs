import numpy as np
from scipy.optimize import minimize

# 定義目標函數
def objective_function(vars):
    x, y = vars
    return x**2 + y**2  # 目標函數 f(x, y) = x^2 + y^2

# 定義約束條件
def constraint(vars):
    x, y = vars
    return x + y - 1  # 約束 g(x, y) = x + y - 1 = 0

# 初始猜測值
initial_guess = [0.5, 0.5]  # 可以隨意選擇一個起始點

# 定義約束條件的字典
constraints = {'type': 'eq', 'fun': constraint}

# 使用最小化函數
result = minimize(objective_function, initial_guess, constraints=constraints)

# 輸出結果
if result.success:
    print("最小化結果：")
    print(f"x: {result.x[0]}, y: {result.x[1]}")
    print(f"目標函數最小值: {result.fun}")
else:
    print("最小化失敗，原因：", result.message)
