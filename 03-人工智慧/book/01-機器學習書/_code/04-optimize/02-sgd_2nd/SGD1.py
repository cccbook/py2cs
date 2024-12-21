import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# 生成簡單的回歸數據
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# 初始化 SGDRegressor，並設置學習率
sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01)

# 擬合模型
sgd_regressor.fit(X, y)

# 預測結果
y_pred = sgd_regressor.predict(X)

# 繪製結果
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='SGD Fit')
plt.legend()
plt.show()

# 顯示最終的參數
print("Final coefficients:", sgd_regressor.coef_)
print("Final intercept:", sgd_regressor.intercept_)
