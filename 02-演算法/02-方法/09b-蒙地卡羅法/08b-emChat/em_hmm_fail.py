import numpy as np

# 生成观测数据
np.random.seed(42)

# 状态转移概率（A矩阵）
transition_prob = np.array([[0.7, 0.3], [0.4, 0.6]])

# 观测概率（B矩阵）
observation_prob = np.array([[0.9, 0.1], [0.2, 0.8]])

# 初始状态概率（π向量）
initial_state_prob = np.array([0.5, 0.5])

# 生成隐藏状态和可见状态序列
num_days = 100
hidden_states = [np.random.choice([0, 1], p=initial_state_prob)]
visible_states = [np.random.choice([0, 1], p=observation_prob[hidden_states[0]])]

for _ in range(num_days - 1):
    hidden_states.append(np.random.choice([0, 1], p=transition_prob[hidden_states[-1]]))
    visible_states.append(np.random.choice([0, 1], p=observation_prob[hidden_states[-1]]))

# 初始化模型参数（使用随机初始值）
initial_state_prob_guess = np.random.rand(2)
initial_state_prob_guess /= np.sum(initial_state_prob_guess)

transition_prob_guess = np.random.rand(2, 2)
transition_prob_guess /= np.sum(transition_prob_guess, axis=1)[:, np.newaxis]

observation_prob_guess = np.random.rand(2, 2)
observation_prob_guess /= np.sum(observation_prob_guess, axis=1)[:, np.newaxis]

# EM算法迭代
max_iterations = 100
tolerance = 1e-6

for iteration in range(max_iterations):
    # E 步
    forward = np.zeros((num_days, 2))
    backward = np.zeros((num_days, 2))
    
    # Forward算法
    forward[0] = initial_state_prob_guess * observation_prob_guess[:, visible_states[0]]
    for t in range(1, num_days):
        forward[t] = observation_prob_guess[:, visible_states[t]] * np.dot(forward[t-1], transition_prob_guess)

    # Backward算法
    backward[-1] = 1
    for t in range(num_days-2, -1, -1):
        backward[t] = np.dot(transition_prob_guess, observation_prob_guess[:, visible_states[t+1]] * backward[t+1])

    state_prob_given_observation = forward * backward / np.sum(forward * backward, axis=1)[:, np.newaxis]

    # M 步
    new_initial_state_prob = state_prob_given_observation[0]
    new_transition_prob = np.dot(state_prob_given_observation[:-1].T, transition_prob_guess.T) / np.sum(
        state_prob_given_observation[:-1], axis=0)[:, np.newaxis]
    new_observation_prob = np.dot(state_prob_given_observation.T, np.eye(2)[visible_states]) / np.sum(
        state_prob_given_observation, axis=0)[:, np.newaxis]

    # 计算参数变化
    delta_initial_state_prob = np.abs(new_initial_state_prob - initial_state_prob_guess)
    delta_transition_prob = np.abs(new_transition_prob - transition_prob_guess)
    delta_observation_prob = np.abs(new_observation_prob - observation_prob_guess)

    # 更新模型参数
    initial_state_prob_guess = new_initial_state_prob
    transition_prob_guess = new_transition_prob
    observation_prob_guess = new_observation_prob

    # 检查收敛
    if np.max(delta_initial_state_prob) + np.max(delta_transition_prob) + np.max(delta_observation_prob) < tolerance:
        break

# 打印结果
print("迭代次数:", iteration + 1)
print("初始状态概率的估计:", initial_state_prob_guess)
print("状态转移概率的估计:\n", transition_prob_guess)
print("观测概率的估计:\n", observation_prob_guess)
