import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 創建一些模擬數據
dates = pd.date_range('2021-01-01', periods=365, freq='D')
data = pd.Series(10 + 0.1 * dates.dayofyear + 2 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(scale=1, size=365), index=dates)

# 時序分解
result = sm.tsa.seasonal_decompose(data, model='additive', period=365)

# 可視化分解結果
result.plot()
plt.show()
