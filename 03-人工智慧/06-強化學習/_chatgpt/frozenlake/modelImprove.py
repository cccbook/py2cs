import numpy as np
import gym

# 參數設定
# env = gym.make("FrozenLake-v1", is_slippery=True)
# env = gym.make("FrozenLake-v1") # 預設為 is_slippery=True
env = gym.make("FrozenLake-v1", is_slippery=False)
# env.seed(0)  # 為了結果一致性
np.random.seed(0)

gamma = 0.99  # 折扣因子
theta = 1e-8  # 收斂判定閾值
num_states = env.observation_space.n
num_actions = env.action_space.n

# 初始化策略和價值函數
policy = np.ones([num_states, num_actions]) / num_actions  # 等概率初始策略
V = np.zeros(num_states)

def policy_evaluation(policy, V, gamma, theta):
    """策略評估：根據當前策略計算狀態價值函數V"""
    while True:
        delta = 0
        for s in range(num_states):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + gamma * V[next_state] * (not done))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

def policy_improvement(policy, V, gamma):
    """策略改進：根據當前價值函數更新策略"""
    policy_stable = True
    for s in range(num_states):
        old_action = np.argmax(policy[s])
        action_values = np.zeros(num_actions)
        for a in range(num_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state] * (not done))
        best_action = np.argmax(action_values)
        if old_action != best_action:
            policy_stable = False
        policy[s] = np.eye(num_actions)[best_action]  # 更新策略為貪婪選擇
    return policy, policy_stable

# 策略迭代主循環
def policy_iteration(env, policy, V, gamma, theta):
    while True:
        V = policy_evaluation(policy, V, gamma, theta)
        policy, policy_stable = policy_improvement(policy, V, gamma)
        if policy_stable:
            break
    return policy, V

# 執行策略迭代
optimal_policy, optimal_value_function = policy_iteration(env, policy, V, gamma, theta)

# 顯示最終結果
print("Optimal Policy:")
print(np.argmax(optimal_policy, axis=1).reshape((4, 4)))  # 以 4x4 格式顯示 Frozen Lake 策略
print("\nOptimal Value Function:")
print(optimal_value_function.reshape((4, 4)))
