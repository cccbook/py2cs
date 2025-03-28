import numpy as np

# 初始化模型参数
initial_state_prob = np.array([0.6, 0.4])
transition_prob = np.array([[0.7, 0.3], [0.4, 0.6]])
observation_prob = np.array([[0.2, 0.8], [0.5, 0.5]])

# 观测序列
observations = [0, 1, 0, 0, 1]  # 0 表示观测到 O，1 表示观测到不是 O

# EM算法迭代
max_iterations = 100
tolerance = 1e-6

for iteration in range(max_iterations):
    # E 步
    forward = np.zeros((len(observations), 2))
    backward = np.zeros((len(observations), 2))

    # 前向算法计算前向概率
    forward[0] = initial_state_prob * observation_prob[:, observations[0]]
    for t in range(1, len(observations)):
        forward[t] = observation_prob[:, observations[t]] * np.dot(transition_prob.T, forward[t - 1])

    # 后向算法计算后向概率
    backward[-1] = 1
    for t in reversed(range(len(observations) - 1)):
        backward[t] = np.dot(transition_prob, observation_prob[:, observations[t + 1]] * backward[t + 1])

    # M 步
    state_prob_given_observation = forward * backward / np.sum(forward * backward, axis=1)[:, np.newaxis]

    # 更新模型参数
    """
    new_initial_state_prob = state_prob_given_observation[0]
    new_transition_prob = np.dot(state_prob_given_observation[:-1].T, transition_prob.T) / np.sum(
        state_prob_given_observation[:-1], axis=0)[:, np.newaxis]
    new_observation_prob = np.dot(state_prob_given_observation.T, np.eye(2)[observations]) / np.sum(
        state_prob_given_observation, axis=0)[:, np.newaxis]
    """
    new_initial_state_prob = state_prob_given_observation[0]
    print('state_prob_given_observation=', state_prob_given_observation)
    print('transition_prob=', transition_prob)
    print("np.sum(state_prob_given_observation, axis=0)[:, np.newaxis]=", np.sum(state_prob_given_observation, axis=0)[:, np.newaxis])
    # new_transition_prob = np.dot(state_prob_given_observation.T, transition_prob.T) / np.sum(
    # state_prob_given_observation, axis=0)[:, np.newaxis]
    
    new_transition_prob = np.dot(state_prob_given_observation, transition_prob.T).T / np.sum(
    state_prob_given_observation, axis=0)[:, np.newaxis]
    new_observation_prob = np.dot(state_prob_given_observation.T, np.eye(2)[observations]) / np.sum(
    state_prob_given_observation, axis=0)[:, np.newaxis]
    print('new_observation_prob=', new_observation_prob)
    print('transition_prob=', transition_prob)
    # 计算参数变化
    delta_initial = np.sum(np.abs(new_initial_state_prob - initial_state_prob))
    delta_transition = np.sum(np.abs(new_transition_prob - transition_prob))
    delta_observation = np.sum(np.abs(new_observation_prob - observation_prob))

    # 更新模型参数
    initial_state_prob = new_initial_state_prob
    transition_prob = new_transition_prob
    observation_prob = new_observation_prob

    # 检查收敛
    if delta_initial + delta_transition + delta_observation < tolerance:
        break

# 打印结果
print("迭代次数:", iteration + 1)
print("初始状态概率:", initial_state_prob)
print("转移概率:", transition_prob)
print("观测概率:", observation_prob)
