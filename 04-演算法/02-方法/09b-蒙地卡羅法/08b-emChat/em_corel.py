import numpy as np
from scipy.stats import multivariate_normal

# 生成数据
np.random.seed(42)
mean1 = [2, 3]
cov1 = [[1, 0.5], [0.5, 1]]
mean2 = [6, 8]
cov2 = [[1, -0.5], [-0.5, 1]]

data1 = np.random.multivariate_normal(mean1, cov1, size=100)
data2 = np.random.multivariate_normal(mean2, cov2, size=150)
observations = np.concatenate([data1, data2])

# 初始化模型参数
cov_matrix = np.eye(2)  # 初始协方差矩阵
mixing_coefficient = 0.5  # 混合系数，即初始选择第一个分布的概率

# EM算法迭代
max_iterations = 100
tolerance = 1e-6

for iteration in range(max_iterations):
    # E 步
    likelihood1 = multivariate_normal.pdf(observations, mean=mean1, cov=cov_matrix)
    likelihood2 = multivariate_normal.pdf(observations, mean=mean2, cov=cov_matrix)
    prob_choose_distribution1 = mixing_coefficient * likelihood1
    prob_choose_distribution2 = (1 - mixing_coefficient) * likelihood2
    normalization = prob_choose_distribution1 + prob_choose_distribution2
    distribution1_given_data = prob_choose_distribution1 / normalization

    # M 步
    new_mixing_coefficient = np.mean(distribution1_given_data)
    new_cov_matrix = np.dot((observations - np.mean(observations, axis=0)).T,
                            np.dot(np.diag(distribution1_given_data), (observations - np.mean(observations, axis=0)))) / np.sum(distribution1_given_data)

    # 计算参数变化
    delta_mixing_coefficient = np.abs(new_mixing_coefficient - mixing_coefficient)
    delta_cov_matrix = np.sum(np.abs(new_cov_matrix - cov_matrix))

    # 更新模型参数
    mixing_coefficient = new_mixing_coefficient
    cov_matrix = new_cov_matrix

    # 检查收敛
    if delta_mixing_coefficient + delta_cov_matrix < tolerance:
        break

# 打印结果
print("迭代次数:", iteration + 1)
print("混合系数:", mixing_coefficient)
print("协方差矩阵:", cov_matrix)
