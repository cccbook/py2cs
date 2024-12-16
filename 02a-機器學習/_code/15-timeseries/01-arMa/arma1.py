import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

# 生成一些模擬數據
np.random.seed(42)
n = 100
data = np.random.normal(size=n)

# AR(1) 模型
ar_model = AutoReg(data, lags=1)
ar_model_fitted = ar_model.fit()
print(f"AR(1) 係數: {ar_model_fitted.params}")

# MA(1) 模型
ma_model = ARIMA(data, order=(0, 0, 1))
ma_model_fitted = ma_model.fit()
print(f"MA(1) 係數: {ma_model_fitted.params}")

# ARMA(1, 1) 模型
arma_model = ARIMA(data, order=(1, 0, 1))
arma_model_fitted = arma_model.fit()
print(f"ARMA(1, 1) 係數: {arma_model_fitted.params}")

# 預測
ar_pred = ar_model_fitted.predict(start=90, end=99)
ma_pred = ma_model_fitted.predict(start=90, end=99)
arma_pred = arma_model_fitted.predict(start=90, end=99)

# 顯示結果
plt.figure(figsize=(10, 6))
plt.plot(data, label="Original Data")
plt.plot(np.arange(90, 100), ar_pred, label="AR(1) Prediction")
plt.plot(np.arange(90, 100), ma_pred, label="MA(1) Prediction")
plt.plot(np.arange(90, 100), arma_pred, label="ARMA(1, 1) Prediction")
plt.legend()
plt.show()
