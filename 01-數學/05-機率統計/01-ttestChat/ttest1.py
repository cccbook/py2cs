from scipy import stats

# 建立一組樣本資料（示例資料）
sample_data = [28, 30, 25, 32, 29, 31, 27, 30, 28, 26]

# 假設的平均值（虛無假設的平均值）
null_hypothesis_mean = 30

# 執行單樣本t檢定
t_statistic, p_value = stats.ttest_1samp(sample_data, null_hypothesis_mean)

# 顯示結果
print("t-statistic:", t_statistic)
print("p-value:", p_value)

# 判斷是否拒絕虛無假設（通常 p-value 低於 0.05 表示統計上顯著）
if p_value < 0.05:
    print("拒絕虛無假設")
else:
    print("未拒絕虛無假設")
