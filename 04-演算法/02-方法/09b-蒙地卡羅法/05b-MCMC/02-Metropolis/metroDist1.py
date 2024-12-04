import numpy as np
import matplotlib.pyplot as plt

# 定義目標分佈的PDF
def target_distribution(x):
    return np.exp(-(x - 2) ** 2) + np.exp(-(x + 2) ** 2)

# Metropolis-Hastings算法實現
def metropolis_hastings(num_samples, sigma):
    samples = []
    # 初始化樣本
    x_current = 0.0

    for _ in range(num_samples):
        # 生成候選樣本
        x_proposed = np.random.normal(x_current, sigma)

        # 計算接受機率
        acceptance_ratio = target_distribution(x_proposed) / target_distribution(x_current)
        acceptance_ratio = min(1, acceptance_ratio)

        # 生成均勻隨機數
        if np.random.rand() < acceptance_ratio:
            x_current = x_proposed
        
        samples.append(x_current)

    return samples

# 參數設定
num_samples = 10000
sigma = 1.0

# 獲取樣本
samples = metropolis_hastings(num_samples, sigma)

# 繪製結果
x = np.linspace(-5, 5, 1000)
plt.hist(samples, bins=50, density=True, alpha=0.6, color='g', label='Sampled Distribution')
plt.plot(x, target_distribution(x) / np.trapz(target_distribution(x), x), label='Target Distribution', color='r')
plt.title('Metropolis-Hastings Sampling')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()
