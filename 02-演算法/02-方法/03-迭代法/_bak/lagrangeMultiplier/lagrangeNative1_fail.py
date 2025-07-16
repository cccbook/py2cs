import numpy as np

# 定義目標函數
def objective_function(x, y):
    return x**2 + y**2  # 目標函數 f(x, y) = x^2 + y^2

# 定義約束函數
def constraint(x, y):
    return x + y - 1  # 約束 g(x, y) = x + y - 1

# 定義拉格朗日函數
def lagrange_function(x, y, lambda_):
    return objective_function(x, y) + lambda_ * constraint(x, y)

# 計算偏導數
def gradients(x, y, lambda_):
    df_dx = 2 * x  # f對x的偏導數
    df_dy = 2 * y  # f對y的偏導數
    dg_dx = 1      # g對x的偏導數
    dg_dy = 1      # g對y的偏導數
    return df_dx, df_dy, dg_dx, dg_dy

# 使用迭代法求解
def optimize(initial_x, initial_y, initial_lambda, learning_rate=0.01, iterations=100):
    x, y, lambda_ = initial_x, initial_y, initial_lambda

    for _ in range(iterations):
        df_dx, df_dy, dg_dx, dg_dy = gradients(x, y, lambda_)

        # 更新x, y和lambda
        x -= learning_rate * (df_dx + lambda_ * dg_dx)
        y -= learning_rate * (df_dy + lambda_ * dg_dy)
        # lambda_ -= learning_rate * constraint(x, y)
    return x, y, objective_function(x, y)

# 設定初始值
initial_x = 0.9
initial_y = 0.9
initial_lambda = 10

# 求解
optimal_x, optimal_y, min_value = optimize(initial_x, initial_y, initial_lambda)

# 輸出結果
print("最小化結果：")
print(f"x: {optimal_x}, y: {optimal_y}")
print(f"目標函數最小值: {min_value}")
