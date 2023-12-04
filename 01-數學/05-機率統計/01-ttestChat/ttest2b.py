from scipy import stats

# 建立兩組配對樣本資料（示例資料）
before_treatment = [28, 30, 25, 32, 29]
after_treatment = [25, 28, 24, 30, 27]

# 執行配對樣本t檢定
t_statistic, p_value = stats.ttest_rel(before_treatment, after_treatment)

# 顯示結果
print("t-statistic:", t_statistic)
print("p-value:", p_value)

# 判斷是否拒絕虛無假設（通常 p-value 低於 0.05 表示統計上顯著）
if p_value < 0.05:
    print("拒絕虛無假設")
else:
    print("未拒絕虛無假設")
