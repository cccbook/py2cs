import random

def f(x1, x2, x3, x4, x5):
    return x1**2 + x2**2 + x3**2 + x4**2 + x5**2

def monte_carlo_integration_5d(num_samples):
    total = 0

    for _ in range(num_samples):
        # 在每個維度 [0, 1] 內生成隨機點
        x1 = random.uniform(0, 1)
        x2 = random.uniform(0, 1)
        x3 = random.uniform(0, 1)
        x4 = random.uniform(0, 1)
        x5 = random.uniform(0, 1)

        # 計算 f(x) 的值
        y = f(x1, x2, x3, x4, x5)

        # 積分值累加
        total += y

    # 估算積分值
    estimated_integral = (total / num_samples)

    # 由於範圍是 [0, 1] 的五維立方體，積分值需乘以立方體體積
    # 體積 = (範圍上界 - 範圍下界)^5
    volume = (1 - 0)**5
    estimated_integral *= volume

    return estimated_integral

# 設定隨機抽樣的次數
num_samples = 1000000

# 執行蒙特卡洛積分
result = monte_carlo_integration_5d(num_samples)

print(f"估算的五變數函數積分值： {result}")
