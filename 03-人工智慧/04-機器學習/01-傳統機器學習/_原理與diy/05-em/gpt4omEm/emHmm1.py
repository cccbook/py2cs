import numpy as np
import matplotlib.pyplot as plt

# 生成隱馬爾可夫模型數據
def generate_hmm_data(num_samples):
    states = [0, 1]  # 兩個隱藏狀態
    observations = [0, 1]  # 兩個觀察值
    transition_matrix = np.array([[0.7, 0.3],  # 從狀態0到0、1的轉移概率
                                   [0.4, 0.6]]) # 從狀態1到0、1的轉移概率
    emission_matrix = np.array([[0.9, 0.1],  # 狀態0觀察0、1的概率
                                 [0.2, 0.8]]) # 狀態1觀察0、1的概率

    # 初始化狀態和觀察序列
    hidden_states = []
    observations_sequence = []
    
    # 隨機選擇初始狀態
    current_state = np.random.choice(states)
    
    for _ in range(num_samples):
        hidden_states.append(current_state)
        observation = np.random.choice(observations, p=emission_matrix[current_state])
        observations_sequence.append(observation)
        current_state = np.random.choice(states, p=transition_matrix[current_state])
    
    return hidden_states, observations_sequence

# EM 演算法實現
def hmm_em_algorithm(observations, num_states, num_iterations):
    # 隨機初始化參數
    transition_matrix = np.random.rand(num_states, num_states)
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
    
    emission_matrix = np.random.rand(num_states, 2)  # 假設只有兩種觀察值
    emission_matrix /= emission_matrix.sum(axis=1, keepdims=True)
    
    for _ in range(num_iterations):
        # E 步驟：計算前向和後向概率
        alpha = np.zeros((len(observations), num_states))
        beta = np.zeros((len(observations), num_states))
        
        # 前向算法
        alpha[0] = 1 / num_states * emission_matrix[:, observations[0]]
        for t in range(1, len(observations)):
            for j in range(num_states):
                alpha[t, j] = emission_matrix[j, observations[t]] * np.sum(alpha[t - 1] * transition_matrix[:, j])
        
        # 後向算法
        beta[-1] = 1
        for t in range(len(observations) - 2, -1, -1):
            for j in range(num_states):
                beta[t, j] = np.sum(beta[t + 1] * transition_matrix[j] * emission_matrix[:, observations[t + 1]])

        # M 步驟：更新參數
        xi = np.zeros((len(observations) - 1, num_states, num_states))
        for t in range(len(observations) - 1):
            denom = np.sum(alpha[t] * transition_matrix * emission_matrix[:, observations[t + 1]] * beta[t + 1])
            if denom > 0:  # 確保分母不為零
                for i in range(num_states):
                    for j in range(num_states):
                        xi[t, i, j] = alpha[t, i] * transition_matrix[i, j] * emission_matrix[j, observations[t + 1]] * beta[t + 1, j] / denom

        transition_matrix = xi.sum(axis=0) / xi.sum(axis=(0, 1))

        # 更新觀察矩陣
        for j in range(num_states):
            for observation in range(2):  # 假設只有兩種觀察值
                numerator = np.sum((observations == observation) * (alpha[:, j] * beta[:, j]))
                denominator = np.sum(alpha[:, j] * beta[:, j])
                if denominator > 0:  # 確保分母不為零
                    emission_matrix[j, observation] = numerator / denominator

    return transition_matrix, emission_matrix

# 生成數據
num_samples = 100
hidden_states, observations = generate_hmm_data(num_samples)

# 執行 EM 演算法
num_states = 2
num_iterations = 100
estimated_transition_matrix, estimated_emission_matrix = hmm_em_algorithm(observations, num_states, num_iterations)

# 輸出結果
print("Estimated Transition Matrix:\n", estimated_transition_matrix)
print("Estimated Emission Matrix:\n", estimated_emission_matrix)

# 可視化隱藏狀態
plt.figure(figsize=(12, 6))
plt.plot(observations, marker='o', label='Observations', linestyle='None')
plt.title('Hidden Markov Model - Observations')
plt.xlabel('Time Step')
plt.ylabel('Observation Value')
plt.legend()
plt.show()
