from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# 生成回歸數據
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# 創建 Ridge 模型並設置正則化強度
ridge_model = Ridge(alpha=1.0)

# 訓練模型
ridge_model.fit(X, y)

# 預測結果
y_pred = ridge_model.predict(X)

# 繪製結果
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Ridge Fit')
plt.legend()
plt.show()

# 顯示模型係數
print("Ridge Coefficients:", ridge_model.coef_)
