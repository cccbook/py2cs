from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# 生成回歸數據
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# 創建線性回歸模型
model = LinearRegression()

# 訓練模型
model.fit(X, y)

# 預測結果
y_pred = model.predict(X)

# 繪製結果
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Linear Fit')
plt.legend()
plt.show()

# 顯示模型係數
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
