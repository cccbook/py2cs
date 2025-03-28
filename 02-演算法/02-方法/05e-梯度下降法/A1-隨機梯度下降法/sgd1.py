import numpy as np
import matplotlib.pyplot as plt

# 生成隨機數據
np.random.seed(42)  # 為了可重複性
X = 2 * np.random.rand(100, 1)  # 隨機生成 100 個樣本的特徵
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + 隨機噪聲

# 定義隨機梯度下降法
def stochastic_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m = len(y)
    theta = np.random.randn(2, 1)  # 隨機初始化參數

    for iteration in range(n_iterations):
        for i in range(m):
            random_index = np.random.randint(m)  # 隨機選擇一個樣本
            xi = X[random_index:random_index+1]  # 對應的特徵
            yi = y[random_index:random_index+1]  # 對應的目標值
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)  # 計算梯度
            theta -= learning_rate * gradients  # 更新參數

    return theta

# 增加一列 1，表示截距項
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 在特徵矩陣中加入一列全為 1 的項

# 使用隨機梯度下降法訓練模型
theta_best = stochastic_gradient_descent(X_b, y)

# 輸出參數
print(f"學習到的參數：{theta_best.ravel()}")

# 繪製數據和擬合的直線
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta_best), color='red', label='fit line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('sgd for linear regression')
plt.legend()
plt.show()
