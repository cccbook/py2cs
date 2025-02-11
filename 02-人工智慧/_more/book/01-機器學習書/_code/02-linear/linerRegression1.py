import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 模擬數據
X = np.random.rand(100, 2)  # 100個樣本，2個特徵
y = 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(100)  # 目標變數

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
print("預測結果:", y_pred)
print("模型係數:", model.coef_)
