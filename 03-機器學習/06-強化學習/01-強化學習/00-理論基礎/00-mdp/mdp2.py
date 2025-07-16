import numpy as np

# 定義狀態和動作
states = ['S0', 'S1']
actions = ['A0', 'A1']
rewards = {
    ('S0', 'A0'): 1,
    ('S0', 'A1'): 0,
    ('S1', 'A0'): 2,
    ('S1', 'A1'): 0
}

# 機率轉移模型：每個 (state, action) 對應多個 (next_state, probability)
transitions = {
    ('S0', 'A0'): [('S1', 0.8), ('S0', 0.2)],
    ('S0', 'A1'): [('S0', 1.0)],
    ('S1', 'A0'): [('S0', 1.0)],
    ('S1', 'A1'): [('S1', 0.7), ('S0', 0.3)]
}

# 初始化價值函數
V = np.zeros(len(states))
gamma = 0.9  # 折扣因子

# 貝爾曼迭代
for _ in range(100):  # 設定迭代次數
    V_new = np.copy(V)
    for i, state in enumerate(states):
        action_values = []
        for action in actions:
            reward = rewards[(state, action)]
            expected_value = 0
            for next_state, prob in transitions[(state, action)]:
                next_index = states.index(next_state)
                expected_value += prob * V[next_index]
            action_value = reward + gamma * expected_value
            action_values.append(action_value)
        V_new[i] = max(action_values)
    V = V_new

# 輸出結果
for i, state in enumerate(states):
    print(f"Value of {state}: {V[i]:.2f}")
