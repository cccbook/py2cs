import numpy as np
import matplotlib.pyplot as plt

# 定義條件機率表（CPT）
def prior_A():
    return np.random.choice([0, 1], p=[0.4, 0.6])  # P(A)

def likelihood_B_given_A(A):
    if A == 1:
        return np.random.choice([0, 1], p=[0.3, 0.7])  # P(B|A=1)
    else:
        return np.random.choice([0, 1], p=[0.8, 0.2])  # P(B|A=0)

def likelihood_C_given_A(A):
    if A == 1:
        return np.random.choice([0, 1], p=[0.1, 0.9])  # P(C|A=1)
    else:
        return np.random.choice([0, 1], p=[0.7, 0.3])  # P(C|A=0)

# Metropolis-Hastings算法實現
def metropolis_hastings_bayesian_network(num_samples, observed_B, observed_C):
    samples = []
    # 初始化樣本
    A_current = prior_A()

    for _ in range(num_samples):
        # 生成候選樣本
        A_proposed = 1 - A_current  # 簡單的候選方式：切換 A 的狀態

        # 計算似然
        likelihood_current = (likelihood_B_given_A(A_current) == observed_B) * \
                             (likelihood_C_given_A(A_current) == observed_C)

        likelihood_proposed = (likelihood_B_given_A(A_proposed) == observed_B) * \
                              (likelihood_C_given_A(A_proposed) == observed_C)

        # 計算接受機率
        acceptance_ratio = likelihood_proposed / likelihood_current if likelihood_current > 0 else 0
        acceptance_ratio = min(1, acceptance_ratio)

        # 生成均勻隨機數
        if np.random.rand() < acceptance_ratio:
            A_current = A_proposed
        
        samples.append(A_current)

    return samples

# 參數設定
num_samples = 10000
observed_B = 1  # 觀察到 B=1
observed_C = 1  # 觀察到 C=1

# 獲取樣本
samples = metropolis_hastings_bayesian_network(num_samples, observed_B, observed_C)

"""
# 繪製結果
plt.hist(samples, bins=2, density=True, alpha=0.6, color='g', label='Sampled Distribution')
plt.xticks([0, 1], ['A=0', 'A=1'])
plt.title('Metropolis-Hastings Sampling for Bayesian Network')
plt.xlabel('A')
plt.ylabel('Density')
plt.legend()
plt.show()
"""

# 計算後驗概率
posterior_A_1 = np.mean(np.array(samples) == 1)
print(f"Posterior Probability P(A=1 | B=1, C=1): {posterior_A_1:.2f}")
