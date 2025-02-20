import numpy as np

# 生成馬可夫鏈數據
def generate_markov_chain(num_samples, transition_matrix):
    states = np.arange(len(transition_matrix))
    current_state = np.random.choice(states)
    observations = []
    
    for _ in range(num_samples):
        observations.append(current_state)
        current_state = np.random.choice(states, p=transition_matrix[current_state])
    
    return observations

# EM 演算法實現
def em_algorithm(observations, num_states, num_iterations):
    # 隨機初始化轉移矩陣
    transition_matrix = np.random.rand(num_states, num_states)
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
    
    for _ in range(num_iterations):
        # E 步驟：計算每一對狀態的出現次數
        xi = np.zeros((num_states, num_states))  # 狀態轉移計數
        for t in range(len(observations) - 1):
            i = observations[t]
            j = observations[t + 1]
            xi[i, j] += 1
        
        # M 步驟：更新轉移矩陣
        transition_matrix = xi / xi.sum(axis=1, keepdims=True)

    return transition_matrix

# 設定參數
num_states = 3
num_samples = 100
true_transition_matrix = np.array([[0.7, 0.2, 0.1],
                                    [0.3, 0.4, 0.3],
                                    [0.2, 0.3, 0.5]])

# 生成數據
observations = generate_markov_chain(num_samples, true_transition_matrix)

# 執行 EM 演算法
num_iterations = 100
estimated_transition_matrix = em_algorithm(observations, num_states, num_iterations)

# 輸出結果
print("Estimated Transition Matrix:\n", estimated_transition_matrix)
print("True Transition Matrix:\n", true_transition_matrix)
