import numpy as np

# 定義先驗概率 P(Cold)
P_Cold = [0.7, 0.3]  # [P(Cold=0), P(Cold=1)]

# 定義條件概率 P(Cough | Cold)
P_Cough_given_Cold = [[0.8, 0.1],  # P(Cough=0 | Cold=0), P(Cough=0 | Cold=1)
                      [0.2, 0.9]]  # P(Cough=1 | Cold=0), P(Cough=1 | Cold=1)

# 定義條件概率 P(Fever | Cold)
P_Fever_given_Cold = [[0.9, 0.3],  # P(Fever=0 | Cold=0), P(Fever=0 | Cold=1)
                      [0.1, 0.7]]  # P(Fever=1 | Cold=0), P(Fever=1 | Cold=1)

# 拒絕抽樣過程
def sample_once():
    # 按照先驗概率采樣 Cold
    cold = np.random.choice([0, 1], p=P_Cold)
    
    # 按照條件概率采樣 Cough 和 Fever
    # P(Cough | Cold=cold)
    cough = np.random.choice([0, 1], p=[P_Cough_given_Cold[0][cold], P_Cough_given_Cold[1][cold]])
    
    # P(Fever | Cold=cold)
    fever = np.random.choice([0, 1], p=[P_Fever_given_Cold[0][cold], P_Fever_given_Cold[1][cold]])
    
    return cold, cough, fever

def reject_sampling(num_samples, evidence):
    samples = []
    
    for _ in range(num_samples):
        cold, cough, fever = sample_once()
        
        # 檢查樣本是否符合證據（例如 Cough=1）
        if evidence['Cough'] == cough:
            samples.append(cold)
    
    return samples

# 推斷 P(Cold | Cough=1)
num_samples = 10000
evidence = {'Cough': 1}

samples = reject_sampling(num_samples, evidence)

# 計算 P(Cold=1 | Cough=1) 和 P(Cold=0 | Cough=1)
P_Cold_given_Cough_1 = sum(samples) / len(samples)
P_not_Cold_given_Cough_1 = 1 - P_Cold_given_Cough_1

print(f"P(Cold=1 | Cough=1) = {P_Cold_given_Cough_1:.4f}")
print(f"P(Cold=0 | Cough=1) = {P_not_Cold_given_Cough_1:.4f}")
