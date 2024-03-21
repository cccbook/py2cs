import numpy as np

# 定義輸入
inputs = np.array([0.5, 0.3, 0.7])

# 定義輸出
outputs = np.array([0.1, 0.2, 0.3])

# 定義隨機權重和偏差
np.random.seed(0)
weights1 = np.random.randn(3, 5)
bias1 = np.random.randn(5)
weights2 = np.random.randn(5, 4)
bias2 = np.random.randn(4)
weights3 = np.random.randn(4, 3)
bias3 = np.random.randn(3)

# 計算隱藏層1的輸出
hidden1 = np.dot(inputs, weights1) + bias1

# 對隱藏層1的輸出使用激活函數
activated_hidden1 = np.tanh(hidden1)

# 計算隱藏層2的輸出
hidden2 = np.dot(activated_hidden1, weights2) + bias2

# 對隱藏層2的輸出使用激活函數
activated_hidden2 = np.tanh(hidden2)

# 計算輸出層的輸出
output = np.dot(activated_hidden2, weights3) + bias3

# 輸出結果
print(output)
