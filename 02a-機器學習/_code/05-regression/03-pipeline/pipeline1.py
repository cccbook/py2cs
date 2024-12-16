from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# 生成回歸數據
X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 創建 Pipeline，將 StandardScaler 和 LinearRegression 兩個步驟組合
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 數據標準化步驟
    ('regressor', LinearRegression())  # 線性回歸模型
])

# 使用 Pipeline 訓練模型
pipeline.fit(X_train, y_train)

# 用測試數據進行預測
y_pred = pipeline.predict(X_test)

# 計算均方誤差（MSE）
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

from sklearn.model_selection import cross_val_score

# 使用 Pipeline 和交叉驗證進行模型評估
scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')

# 顯示每次交叉驗證的結果
print(f'Cross-validation MSE scores: {-scores}')
print(f'Mean Cross-validation MSE: {-scores.mean()}')
