from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error

# 擬合 ARIMA 模型
model = ARIMA(df, order=(5,1,0))  # 這是 ARIMA(5,1,0) 模型
model_fit = model.fit()

# 預測結果
forecast = model_fit.forecast(steps=30)

# 殘差分析
residuals = model_fit.resid
plot_acf(residuals)
plt.show()

# 計算 MSE
y_pred = model_fit.forecast(steps=30)
mse = mean_squared_error(df[-30:], y_pred)
print(f'Mean Squared Error: {mse}')
