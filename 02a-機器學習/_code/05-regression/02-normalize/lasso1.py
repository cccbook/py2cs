from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# 生成回歸數據
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# 創建 Lasso 模型並設置正則化強度
lasso_model = Lasso(alpha=0.1)

# 訓練模型
lasso_model.fit(X, y)

# 預測結果
y_pred = lasso_model.predict(X)

# 繪製結果
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Lasso Fit')
plt.legend()
plt.show()

# 顯示模型係數
print("Lasso Coefficients:", lasso_model.coef_)
