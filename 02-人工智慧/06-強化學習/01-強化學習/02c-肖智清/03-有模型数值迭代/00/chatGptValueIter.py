# https://chatgpt.com/c/67282e0f-18f8-8012-b785-97caa58ed78c

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

# 初始化狀態價值
V = {state: 0 for state in states}

# 設定收斂條件
epsilon = 1e-4
max_iterations = 1000

for i in range(max_iterations):
    delta = 0  # 追蹤價值變化
    V_new = V.copy()  # 新一輪迭代的價值
    for state in states:
        # 對於每個狀態，計算行動的最大價值
        action_values = []
        for action in actions:
            # 計算當前行動的價值
            value = sum(prob * (reward + gamma * V[next_state]) 
                        for next_state, prob, reward in P[state][action])
            action_values.append(value)
        # 更新狀態的價值
        V_new[state] = max(action_values)
        # 計算最大變化量以檢查收斂
        delta = max(delta, abs(V_new[state] - V[state]))

    # 更新狀態價值
    V = V_new
    
    # 檢查是否收斂
    if delta < epsilon:
        print(f"在第 {i+1} 次迭代時收斂")
        break

# 輸出最終的價值函數
print("最優價值函數:")
for state in states:
    print(f"V({state}) = {V[state]:.2f}")
