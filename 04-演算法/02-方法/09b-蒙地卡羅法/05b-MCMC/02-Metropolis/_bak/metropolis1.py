import numpy as np
import matplotlib.pyplot as plt

# 定義目標分布 P(x) 為標準正態分布
def target_distribution(x):
    return np.exp(-x**2 / 2)

# Metropolis-Hastings 算法
def metropolis_hastings(target, proposal_width, initial_x, iterations):
    # 初始化變量
    x = initial_x
    samples = []
    
    for i in range(iterations):
        # 從提議分布（對稱的正態分布）中生成候選樣本
        x_proposal = np.random.normal(x, proposal_width)
        
        # 計算接受率 α
        acceptance_ratio = target(x_proposal) / target(x)
        
        # 決定是否接受
        if np.random.uniform(0, 1) < acceptance_ratio:
            x = x_proposal  # 接受候選樣本
            
        samples.append(x)  # 儲存樣本
    
    return np.array(samples)

# 參數設置
proposal_width = 1.0  # 提議分布的標準差
initial_x = 0.0       # 初始樣本
iterations = 10000    # 迭代次數

# 執行 Metropolis-Hastings 抽樣
samples = metropolis_hastings(target_distribution, proposal_width, initial_x, iterations)

# 可視化結果
x_values = np.linspace(-4, 4, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x_values, (1/np.sqrt(2*np.pi)) * np.exp(-x_values**2 / 2), 'r', lw=3, label='True Distribution')
plt.hist(samples, bins=50, density=True, alpha=0.6, color='blue', label='MCMC Samples')
plt.legend()
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Metropolis-Hastings Sampling of Standard Normal Distribution')
plt.show()
