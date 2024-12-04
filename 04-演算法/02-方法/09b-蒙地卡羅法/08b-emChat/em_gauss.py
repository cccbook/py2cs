import numpy as np
from scipy.stats import norm

# 生成混合高斯模型数据
np.random.seed(42)
data = np.concatenate([np.random.normal(loc=5, scale=1, size=100),
                       np.random.normal(loc=10, scale=2, size=150)])

# 初始化模型参数
mu1, sigma1 = 4, 1
mu2, sigma2 = 8, 1.5
mixing_coefficient = 0.5  # 混合系数，即初始选择第一个高斯分布的概率

# EM算法迭代
max_iterations = 100
tolerance = 1e-6

for iteration in range(max_iterations):
    # E 步
    likelihood1 = norm.pdf(data, loc=mu1, scale=sigma1)
    likelihood2 = norm.pdf(data, loc=mu2, scale=sigma2)
    prob_choose_distribution1 = mixing_coefficient * likelihood1
    prob_choose_distribution2 = (1 - mixing_coefficient) * likelihood2
    normalization = prob_choose_distribution1 + prob_choose_distribution2
    distribution1_given_data = prob_choose_distribution1 / normalization

    # M 步
    new_mixing_coefficient = np.mean(distribution1_given_data)
    new_mu1 = np.sum(distribution1_given_data * data) / np.sum(distribution1_given_data)
    new_sigma1 = np.sqrt(np.sum(distribution1_given_data * (data - new_mu1)**2) / np.sum(distribution1_given_data))
    
    new_mu2 = np.sum((1 - distribution1_given_data) * data) / np.sum(1 - distribution1_given_data)
    new_sigma2 = np.sqrt(np.sum((1 - distribution1_given_data) * (data - new_mu2)**2) / np.sum(1 - distribution1_given_data))

    # 计算参数变化
    delta_mixing_coefficient = np.abs(new_mixing_coefficient - mixing_coefficient)
    delta_mu1 = np.abs(new_mu1 - mu1)
    delta_sigma1 = np.abs(new_sigma1 - sigma1)
    delta_mu2 = np.abs(new_mu2 - mu2)
    delta_sigma2 = np.abs(new_sigma2 - sigma2)

    # 更新模型参数
    mixing_coefficient = new_mixing_coefficient
    mu1, sigma1 = new_mu1, new_sigma1
    mu2, sigma2 = new_mu2, new_sigma2

    # 检查收敛
    if delta_mixing_coefficient + delta_mu1 + delta_sigma1 + delta_mu2 + delta_sigma2 < tolerance:
        break

# 打印结果
print("迭代次数:", iteration + 1)
print("混合系数:", mixing_coefficient)
print("第一个高斯分布的均值和方差:", mu1, sigma1)
print("第二个高斯分布的均值和方差:", mu2, sigma2)
