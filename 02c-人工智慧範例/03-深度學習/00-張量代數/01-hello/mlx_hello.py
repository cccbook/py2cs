import mlx.core as mx
import mlx.nn as nn

# 創建一個簡單的張量
x = mx.array([1, 2, 3, 4, 5])
print("Original array:", x)

# 進行一些基本運算
y = x * 2
print("Multiplied by 2:", y)

# 計算平均值
mean = mx.mean(x)
print("Mean value:", mean)

# 創建一個 2D 張量
matrix = mx.array([[1, 2, 3], [4, 5, 6]])
print("\n2D array:")
print(matrix)

# 矩陣轉置
transposed = mx.transpose(matrix)
print("\nTransposed matrix:")
print(transposed)

# 使用 neural network 模組創建一個簡單的線性層
linear = nn.Linear(3, 2)  # 輸入維度 3，輸出維度 2
input_data = mx.array([[1.0, 2.0, 3.0]])
output = linear(input_data)
print("\nLinear layer output:")
print(output)
