import numpy as np
from scipy.stats import norm

# 生成数据
np.random.seed(42)
data1 = np.random.normal(loc=5, scale=1, size=100)
data2 = np.random.normal(loc=10, scale=2, size=150)
observations = np.concatenate([data1, data2])

# 初始化模型参数
mean1 = 4
mean2 = 8
mixing_coefficient = 0.5  # 混合系数，即初始选择第一个成分的概率

# EM算法迭代
max_iterations = 100
tolerance = 1e-6

for iteration in range(max_iterations):
    # E 步
    likelihood1 = norm.pdf(observations, loc=mean1, scale=1)
    likelihood2 = norm.pdf(observations, loc=mean2, scale=1)
    prob_choose_component1 = mixing_coefficient * likelihood1
    prob_choose_component2 = (1 - mixing_coefficient) * likelihood2
    normalization = prob_choose_component1 + prob_choose_component2
    component1_given_data = prob_choose_component1 / normalization

    # M 步
    new_mixing_coefficient = np.mean(component1_given_data)
    new_mean1 = np.sum(component1_given_data * observations) / np.sum(component1_given_data)
    new_mean2 = np.sum((1 - component1_given_data) * observations) / np.sum(1 - component1_given_data)

    # 计算参数变化
    delta_mixing_coefficient = np.abs(new_mixing_coefficient - mixing_coefficient)
    delta_mean1 = np.abs(new_mean1 - mean1)
    delta_mean2 = np.abs(new_mean2 - mean2)

    # 更新模型参数
    mixing_coefficient = new_mixing_coefficient
    mean1 = new_mean1
    mean2 = new_mean2

    # 检查收敛
    if delta_mixing_coefficient + delta_mean1 + delta_mean2 < tolerance:
        break

# 打印结果
print("迭代次数:", iteration + 1)
print("混合系数:", mixing_coefficient)
print("第一个成分的均值:", mean1)
print("第二个成分的均值:", mean2)
