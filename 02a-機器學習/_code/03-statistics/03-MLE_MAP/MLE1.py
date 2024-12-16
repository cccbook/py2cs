import numpy as np
from scipy.stats import norm

# 生成一些數據，假設來自正態分佈
data = np.random.normal(loc=5, scale=2, size=1000)

# 計算最大似然估計（MLE）估算的均值和標準差
mean_MLE = np.mean(data)
std_MLE = np.std(data)

print(f"MLE 估算的均值: {mean_MLE}")
print(f"MLE 估算的標準差: {std_MLE}")
