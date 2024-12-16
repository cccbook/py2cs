import numpy as np
import matplotlib.pyplot as plt

# 模擬1000個來自正態分佈的隨機數
mu, sigma = 0, 0.1  # 均值與標準差
samples = np.random.normal(mu, sigma, 1000)

# 可視化結果
plt.hist(samples, bins=30, density=True, alpha=0.6, color='b')
plt.title('正態分佈 (Normal Distribution)')
plt.show()
