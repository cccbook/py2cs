from scipy.stats import norm
import numpy as np

# 假設均值的先驗分佈是 N(0, 1)
prior_mean = 0
prior_std = 1

# 假設樣本數據來自正態分佈，且我們希望估算均值
data = np.random.normal(loc=5, scale=2, size=1000)

# 似然函數：基於樣本數據的平均值
likelihood_mean = np.mean(data)

# MAP 估計：結合先驗和似然
posterior_mean = (prior_std**2 * likelihood_mean + len(data) * np.var(data) * prior_mean) / (prior_std**2 + len(data) * np.var(data))

print(f"MAP 估算的均值: {posterior_mean}")
