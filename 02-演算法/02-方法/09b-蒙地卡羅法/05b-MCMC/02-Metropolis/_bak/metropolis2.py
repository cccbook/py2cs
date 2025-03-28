import numpy as np
import matplotlib.pyplot as plt

# 定義目標分布：雙峰分布
def target_distribution(x):
    # 兩個正態分布的加權和
    component1 = 0.3 * np.exp(-(x + 2)**2 / (2 * 0.5**2)) / np.sqrt(2 * np.pi * 0.5**2)
    component2 = 0.7 * np.exp(-(x - 3)**2 / (2 * 1.0**2)) / np.sqrt(2 * np.pi * 1.0**2)
    return component1 + component2

# Metropolis-Hastings 算法
def metropolis_hastings(target, proposal_width, initial_x, iterations):
    x = initial_x
    samples = []
    
    for i in range(iterations):
        # 從提議分布生成候選樣本
        x_proposal = np.random.normal(x, proposal_width)
        
        # 計算接受率 α
        acceptance_ratio = target(x_proposal) / target(x)
        
        # 決定是否接受候選樣本
        if np.random.uniform(0, 1) < acceptance_ratio:
            x = x_proposal  # 接受候選樣本
        
        samples.append(x)  # 儲存樣本
    
    return np.array(samples)

# 參數設置
proposal_width = 1.0   # 提議分布的標準差
initial_x = 0.0        # 初始樣本
iterations = 20000     # 迭代次數

# 執行 Metropolis-Hastings 抽樣
samples = metropolis_hastings(target_distribution, proposal_width, initial_x, iterations)

# 可視化結果
x_values = np.linspace(-5, 7, 1000)
true_distribution = target_distribution(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, true_distribution, 'r', lw=3, label='True Distribution')
plt.hist(samples, bins=50, density=True, alpha=0.6, color='blue', label='MCMC Samples')
plt.legend()
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Metropolis-Hastings Sampling of Bimodal Distribution')
plt.show()
