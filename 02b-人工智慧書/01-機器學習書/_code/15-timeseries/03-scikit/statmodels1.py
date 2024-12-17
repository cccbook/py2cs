from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# 使用 LinearRegression 模型
model = LinearRegression()
X = df.index.values.reshape(-1, 1)  # 時間索引作為特徵
y = df.values

# 使用交叉驗證評估模型
cv_scores = cross_val_score(model, X, y, cv=5)
print(f'Cross-validated MSE: {cv_scores.mean()}')

# 使用 statsmodels 進行回歸分析
X = sm.add_constant(X)  # 添加截距項
ols_model = sm.OLS(y, X)
ols_result = ols_model.fit()
print(ols_result.summary())  # 顯示詳細回歸結果
