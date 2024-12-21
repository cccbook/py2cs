from scipy.stats import chi2_contingency

# 建立一個觀察到的頻率表（示例資料）
observed_data = [[25, 30, 15], [10, 20, 25], [5, 10, 15]]

# 執行卡方檢定
chi2_stat, p_value, dof, expected = chi2_contingency(observed_data)

# 顯示結果
print("Chi-square statistic:", chi2_stat)
print("p-value:", p_value)
print("Degrees of freedom:", dof)
print("Expected frequencies table:")
print(expected)

# 判斷是否拒絕虛無假設（通常 p-value 低於 0.05 表示統計上顯著）
if p_value < 0.05:
    print("拒絕虛無假設")
else:
    print("未拒絕虛無假設")
