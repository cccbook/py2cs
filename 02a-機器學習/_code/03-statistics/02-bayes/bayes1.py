# 定義聯合概率 P(A ∩ B) 和邊際概率 P(B)
P_A_and_B = 0.2  # P(A ∩ B)
P_B = 0.5        # P(B)

# 計算條件概率 P(A|B)
P_A_given_B = P_A_and_B / P_B
print("P(A|B) =", P_A_given_B)
