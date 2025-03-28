import numpy as np
from scipy.stats import norm

# 生成数据
np.random.seed(42)
data1 = np.random.normal(loc=5, scale=1, size=100)
data2 = np.random.normal(loc=8, scale=2, size=150)
observations = np.concatenate([data1, data2])

# 初始化模型参数
mean1 = 4
stddev1 = 1
mean2 = 7
stddev2 = 2
mixing_coefficient = 0.5  # 混合系数，即初始选择第一个群体的概率

# EM算法迭代
max_iterations = 100
tolerance = 1e-6

for iteration in range(max_iterations):
    # E 步
    likelihood1 = norm.pdf(observations, loc=mean1, scale=stddev1)
    likelihood2 = norm.pdf(observations, loc=mean2, scale=stddev2)
    prob_choose_group1 = mixing_coefficient * likelihood1
    prob_choose_group2 = (1 - mixing_coefficient) * likelihood2
    normalization = prob_choose_group1 + prob_choose_group2
    group1_given_data = prob_choose_group1 / normalization

    # M 步
    new_mixing_coefficient = np.mean(group1_given_data)
    new_mean1 = np.sum(group1_given_data * observations) / np.sum(group1_given_data)
    new_stddev1 = np.sqrt(np.sum(group1_given_data * (observations - new_mean1)**2) / np.sum(group1_given_data))
    
    new_mean2 = np.sum((1 - group1_given_data) * observations) / np.sum(1 - group1_given_data)
    new_stddev2 = np.sqrt(np.sum((1 - group1_given_data) * (observations - new_mean2)**2) / np.sum(1 - group1_given_data))

    # 计算参数变化
    delta_mixing_coefficient = np.abs(new_mixing_coefficient - mixing_coefficient)
    delta_mean1 = np.abs(new_mean1 - mean1)
    delta_stddev1 = np.abs(new_stddev1 - stddev1)
    delta_mean2 = np.abs(new_mean2 - mean2)
    delta_stddev2 = np.abs(new_stddev2 - stddev2)

    # 更新模型参数
    mixing_coefficient = new_mixing_coefficient
    mean1, stddev1 = new_mean1, new_stddev1
    mean2, stddev2 = new_mean2, new_stddev2

    # 检查收敛
    if delta_mixing_coefficient + delta_mean1 + delta_stddev1 + delta_mean2 + delta_stddev2 < tolerance:
        break

# 打印结果
print("迭代次数:", iteration + 1)
print("混合系数:", mixing_coefficient)
print("第一个群体的均值和标准差:", mean1, stddev1)
print("第二个群体的均值和标准差:", mean2, stddev2)
