import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 創建模擬時間序列數據
dates = pd.date_range('2021-01-01', periods=365, freq='D')
data = 10 + 0.1 * dates.dayofyear + 2 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(scale=1, size=365)
df = pd.Series(data, index=dates)

# 時序分解
result = sm.tsa.seasonal_decompose(df, model='additive', period=365)

# 可視化分解結果
result.plot()
plt.show()

# 提取趨勢與季節性作為特徵
trend = result.trend.dropna()
seasonal = result.seasonal.dropna()

# 準備特徵與目標變量
X = pd.DataFrame({'trend': trend, 'seasonal': seasonal})
y = df.loc[trend.index]  # 目標變量是原始數據

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練隨機森林模型
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
