import numpy as np

# 設定平均值為 100，標準差為 2
mu, sigma = 100, 2

# 使用 numpy 的 random.normal() 函數產生 10 個常態分佈亂數
data = np.random.normal(mu, sigma, 10)

# 打印亂數
print(data)
