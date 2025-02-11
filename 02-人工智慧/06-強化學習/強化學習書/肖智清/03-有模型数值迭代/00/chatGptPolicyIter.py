import numpy as np

# 狀態和行動空間
states = ["s1", "s2"]
actions = ["a1", "a2"]

# 折扣因子
gamma = 0.9

# 轉移概率和回報函數
# 格式: P[s][a] 是字典，包含狀態轉移的概率和回報值
P = {
    "s1": {
        "a1": [("s1", 0.5, 5), ("s2", 0.5, 5)],
        "a2": [("s1", 0.7, 3), ("s2", 0.3, 3)],
    },
    "s2": {
        "a1": [("s1", 0.4, 4), ("s2", 0.6, 4)],
        "a2": [("s1", 0.6, 2), ("s2", 0.4, 2)],
    },
}

# 初始化策略（隨機選擇每個狀態的行動）
policy = {state: np.random.choice(actions) for state in states}

# 初始化狀態價值
V = {state: 0 for state in states}

# 設定收斂條件
epsilon = 1e-4
max_iterations = 1000

def policy_evaluation(policy, V, gamma, epsilon):
    """策略評估：給定策略計算狀態價值函數"""
    while True:
        delta = 0
        for state in states:
            action = policy[state]
            new_value = sum(prob * (reward + gamma * V[next_state])
                            for next_state, prob, reward in P[state][action])
            delta = max(delta, abs(V[state] - new_value))
            V[state] = new_value
        if delta < epsilon:
            break
    return V

def policy_improvement(policy, V, gamma):
    """策略改進：更新策略以獲得更高的狀態價值"""
    policy_stable = True
    for state in states:
        old_action = policy[state]
        # 找到使價值最大的行動
        action_values = {action: sum(prob * (reward + gamma * V[next_state])
                                     for next_state, prob, reward in P[state][action])
                         for action in actions}
        best_action = max(action_values, key=action_values.get)
        # 更新策略
        if best_action != old_action:
            policy_stable = False
        policy[state] = best_action
    return policy, policy_stable

# 策略迭代主程式
for i in range(max_iterations):
    # 策略評估
    V = policy_evaluation(policy, V, gamma, epsilon)
    
    # 策略改進
    policy, policy_stable = policy_improvement(policy, V, gamma)
    
    # 檢查是否收斂
    if policy_stable:
        print(f"在第 {i+1} 次迭代時策略收斂")
        break

# 輸出最終的策略與價值函數
print("最優策略:")
for state in states:
    print(f"π({state}) = {policy[state]}")

print("\n最優價值函數:")
for state in states:
    print(f"V({state}) = {V[state]:.2f}")
