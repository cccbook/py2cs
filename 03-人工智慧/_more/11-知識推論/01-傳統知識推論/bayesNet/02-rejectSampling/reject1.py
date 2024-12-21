import numpy as np
import matplotlib.pyplot as plt

# 目標分佈：標準正態分佈
def target_distribution(x):
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

# 提議分佈：均勻分佈
def proposal_distribution(size=1):
    return np.random.uniform(-3, 3, size)

# 常數 M：用於縮放提議分佈，這裡選擇 M 大於或等於目標分佈與提議分佈比值的最大值
M = 2.5

# Reject Sampling
def reject_sampling(num_samples):
    samples = []
    while len(samples) < num_samples:
        # 從提議分佈中抽樣
        x = proposal_distribution()
        
        # 計算接受概率
        acceptance_ratio = target_distribution(x) / (M * 1/6)  # 提議分佈均勻分佈 U(-3, 3) 的概率密度是 1/6
        
        # 擲骰子決定是否接受樣本
        if np.random.uniform(0, 1) < acceptance_ratio:
            samples.append(x[0])
    
    return np.array(samples)

# 生成 10000 個樣本
num_samples = 10000
samples = reject_sampling(num_samples)

# 畫出結果
x = np.linspace(-4, 4, 1000)
plt.figure(figsize=(10, 6))

# 標準正態分佈的理論曲線
plt.plot(x, target_distribution(x), label="Target Distribution (Normal)", color='red', lw=2)

# 使用 Reject Sampling 生成的樣本的直方圖
plt.hist(samples, bins=50, density=True, alpha=0.6, color='blue', label="Samples from Reject Sampling")

plt.title("Reject Sampling from Standard Normal Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()
