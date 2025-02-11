import numpy as np

# 定義先驗概率 P(Cold)
P_Cold = np.array([0.7, 0.3])  # [P(Cold=0), P(Cold=1)]

# 定義條件概率 P(Cough | Cold)
P_Cough_given_Cold = np.array([[0.8, 0.1],  # P(Cough=0 | Cold=0), P(Cough=0 | Cold=1)
                               [0.2, 0.9]]) # P(Cough=1 | Cold=0), P(Cough=1 | Cold=1)

# 定義條件概率 P(Fever | Cold)
P_Fever_given_Cold = np.array([[0.9, 0.3],  # P(Fever=0 | Cold=0), P(Fever=0 | Cold=1)
                               [0.1, 0.7]]) # P(Fever=1 | Cold=0), P(Fever=1 | Cold=1)

# 推斷過程：給定病人咳嗽 (Cough=1)，推斷感冒 (Cold) 的概率
# 使用貝氏定理計算 P(Cold | Cough=1)

# 計算 P(Cough=1)
P_Cough_1 = (P_Cold[0] * P_Cough_given_Cold[1, 0] + 
             P_Cold[1] * P_Cough_given_Cold[1, 1])

# 計算 P(Cold=1 | Cough=1) = P(Cough=1 | Cold=1) * P(Cold=1) / P(Cough=1)
P_Cold_1_given_Cough_1 = (P_Cough_given_Cold[1, 1] * P_Cold[1]) / P_Cough_1

# 計算 P(Cold=0 | Cough=1)
P_Cold_0_given_Cough_1 = (P_Cough_given_Cold[1, 0] * P_Cold[0]) / P_Cough_1

# 打印結果
print(f"P(Cold=1 | Cough=1) = {P_Cold_1_given_Cough_1:.4f}")
print(f"P(Cold=0 | Cough=1) = {P_Cold_0_given_Cough_1:.4f}")
