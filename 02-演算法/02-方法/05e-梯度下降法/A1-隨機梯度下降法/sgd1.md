ChatGPT:

以下是一個使用隨機梯度下降法（SGD）來訓練簡單線性回歸模型的 Python 範例。這個範例將生成一些隨機的線性數據，然後使用隨機梯度下降法來學習模型參數。

### 隨機梯度下降法範例：簡單線性回歸

```python
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
plt.plot(X, X_b.dot(theta_best), color='red', label='擬合直線')
plt.xlabel('X')
plt.ylabel('y')
plt.title('隨機梯度下降法：簡單線性回歸')
plt.legend()
plt.show()
```

### 代碼說明

1. **數據生成**：
   - 使用 `numpy` 生成隨機數據，`X` 為特徵，`y` 為目標值，並添加隨機噪聲。

2. **隨機梯度下降法實現**：
   - 定義了 `stochastic_gradient_descent` 函數，該函數接受特徵矩陣 `X`、目標值 `y`、學習率和迭代次數作為參數。
   - 在每次迭代中，隨機選擇一個樣本，計算該樣本的梯度並更新模型參數。

3. **擬合與繪圖**：
   - 使用擬合後的參數繪製擬合的直線，並顯示原始數據點。

### 結果
執行上述代碼後，會顯示一個散點圖，其中包含生成的隨機數據點以及用隨機梯度下降法擬合的直線。這樣可以清楚地看到模型的擬合效果。