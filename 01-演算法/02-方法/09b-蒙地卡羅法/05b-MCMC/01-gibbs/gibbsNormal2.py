import numpy as np
import matplotlib.pyplot as plt

# 參數設定
mu_X, mu_Y = 0, 0      # 均值
sigma_X, sigma_Y = 1, 1  # 標準差
rho = 0.8               # 相關係數
num_samples = 10000     # 抽樣數量

# 初始化
x_samples = np.zeros(num_samples)
y_samples = np.zeros(num_samples)

# 初始值
x_samples[0], y_samples[0] = np.random.randn(2)

# Gibbs Sampling
for i in range(1, num_samples):
    # 從 Y 條件分佈中抽樣
    mu_x_given_y = mu_X + rho * (sigma_X / sigma_Y) * (y_samples[i-1] - mu_Y)
    sigma_x_given_y = sigma_X * np.sqrt(1 - rho**2)
    x_samples[i] = np.random.normal(mu_x_given_y, sigma_x_given_y)
    
    # 從 X 條件分佈中抽樣
    mu_y_given_x = mu_Y + rho * (sigma_Y / sigma_X) * (x_samples[i] - mu_X)
    sigma_y_given_x = sigma_Y * np.sqrt(1 - rho**2)
    y_samples[i] = np.random.normal(mu_y_given_x, sigma_y_given_x)

# 繪圖
plt.figure(figsize=(8, 6))
plt.scatter(x_samples, y_samples, alpha=0.3, s=10)
plt.title('Gibbs Sampling from Bivariate Normal Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
