import numpy as np

# 模型參數
pi = np.array([0.6, 0.4])  # 初始概率
A = np.array([[0.7, 0.3],   # 轉移概率
              [0.4, 0.6]])
B = np.array([[0.9, 0.1],   # 發射概率
              [0.2, 0.8]])

# 觀察序列
O = [0, 1, 0, 1, 0]  # 對應於觀察狀態 O1, O2, O1, O2, O1

# Gibbs Sampling 函數
def gibbs_sampling(O, pi, A, B, num_samples):
    num_states = len(pi)
    samples = []

    # 初始隱藏狀態隨機選擇
    hidden_states = np.random.choice(num_states, size=len(O), p=pi)

    for _ in range(num_samples):
        # 隱藏狀態抽樣
        for t in range(len(O)):
            # 計算條件概率
            if t == 0:
                prob = pi * B[:, O[t]]
            else:
                prob = (A[:, hidden_states[t-1]] * B[:, O[t]])

            prob /= np.sum(prob)  # 正規化
            hidden_states[t] = np.random.choice(num_states, p=prob)

        samples.append(hidden_states.copy())

    return np.array(samples)

# 生成樣本
num_samples = 1000
samples = gibbs_sampling(O, pi, A, B, num_samples)

# 顯示結果
print("生成的隱藏狀態樣本：")
print(samples)
