import numpy as np

# 定義貝氏網路的權重和偏差
weights = np.array([0.1, 0.2, 0.3])
bias = 0.1

# 定義輸入
inputs = np.array([0.5, 0.3, 0.7])

# 計算輸出
output = np.dot(weights, inputs) + bias

# 輸出結果
print(output)
