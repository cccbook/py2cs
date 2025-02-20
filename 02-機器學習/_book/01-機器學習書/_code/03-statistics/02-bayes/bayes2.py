# 定義先驗、似然和邊際概率
P_A = 0.3   # P(A)
P_B_given_A = 0.7  # P(B|A)
P_B = 0.5   # P(B)

# 計算後驗概率 P(A|B)
P_A_given_B = (P_B_given_A * P_A) / P_B
print("P(A|B) =", P_A_given_B)
