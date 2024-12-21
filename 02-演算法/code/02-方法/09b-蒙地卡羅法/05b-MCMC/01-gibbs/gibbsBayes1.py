import numpy as np
import pandas as pd

# 假設有一個簡單的貝葉斯網絡，其中有兩個變數 A 和 B
# P(A) 和 P(B|A) 的參數
p_A = [0.6, 0.4]  # P(A=1) = 0.6, P(A=0) = 0.4
p_B_given_A = [[0.9, 0.1],  # P(B=1|A=1)
                [0.2, 0.8]]  # P(B=1|A=0)

# Gibbs Sampling 函數
def gibbs_sampling(num_samples, initial_state):
    samples = []
    current_state = initial_state

    for _ in range(num_samples):
        # 更新 A
        prob_A = np.random.rand()
        if prob_A < p_A[current_state[1]]:  # A的機率
            current_state[0] = 1  # A=1
        else:
            current_state[0] = 0  # A=0
        
        # 更新 B
        prob_B = np.random.rand()
        if current_state[0] == 1:  # A=1
            if prob_B < p_B_given_A[0][current_state[0]]:
                current_state[1] = 1  # B=1
            else:
                current_state[1] = 0  # B=0
        else:  # A=0
            if prob_B < p_B_given_A[1][current_state[0]]:
                current_state[1] = 1  # B=1
            else:
                current_state[1] = 0  # B=0

        samples.append(tuple(current_state))

    return samples

# 進行抽樣
initial_state = [0, 0]  # A=0, B=0
num_samples = 10000
samples = gibbs_sampling(num_samples, initial_state)

# 統計結果
samples_df = pd.DataFrame(samples, columns=['A', 'B'])
print(samples_df.groupby(['A', 'B']).size() / num_samples)  # 計算每種組合的概率
