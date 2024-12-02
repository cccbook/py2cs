import numpy as np

# 初始化模型参数
theta1 = 0.6
theta2 = 0.3
coin_choice_prob = 0.5  # 初始选择硬币的概率

# 生成观测数据
np.random.seed(42)
observations = np.random.choice([0, 1], size=100, p=[1 - coin_choice_prob, coin_choice_prob])

# EM算法迭代
max_iterations = 100
tolerance = 1e-6

for iteration in range(max_iterations):
    # E 步
    prob_choose_coin1 = coin_choice_prob * (theta1**observations) * ((1 - theta1)**(1 - observations))
    prob_choose_coin2 = (1 - coin_choice_prob) * (theta2**observations) * ((1 - theta2)**(1 - observations))
    normalization = prob_choose_coin1 + prob_choose_coin2
    coin1_given_observations = prob_choose_coin1 / normalization

    # M 步
    new_coin_choice_prob = np.mean(coin1_given_observations)
    new_theta1 = np.sum(coin1_given_observations * observations) / np.sum(coin1_given_observations)
    new_theta2 = np.sum((1 - coin1_given_observations) * observations) / np.sum(1 - coin1_given_observations)

    # 计算参数变化
    delta_coin_choice_prob = np.abs(new_coin_choice_prob - coin_choice_prob)
    delta_theta1 = np.abs(new_theta1 - theta1)
    delta_theta2 = np.abs(new_theta2 - theta2)

    # 更新模型参数
    coin_choice_prob = new_coin_choice_prob
    theta1 = new_theta1
    theta2 = new_theta2

    # 检查收敛
    if delta_coin_choice_prob + delta_theta1 + delta_theta2 < tolerance:
        break

# 打印结果
print("迭代次数:", iteration + 1)
print("初始选择硬币的概率:", coin_choice_prob)
print("硬币1正面出现的概率:", theta1)
print("硬币2正面出现的概率:", theta2)
