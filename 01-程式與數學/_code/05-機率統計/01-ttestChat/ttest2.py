from scipy import stats

# 建立兩組樣本資料（示例資料）
group1 = [25, 30, 35, 28, 22, 29, 34, 30, 35, 28]
group2 = [31, 28, 27, 32, 35, 30, 28, 34, 32, 30]

# 執行獨立樣本t檢定
t_statistic, p_value = stats.ttest_ind(group1, group2)

# 顯示結果
print("t-statistic:", t_statistic)
print("p-value:", p_value)

# 判斷是否拒絕虛無假設（通常 p-value 低於 0.05 表示統計上顯著）
if p_value < 0.05:
    print("拒絕虛無假設")
else:
    print("未拒絕虛無假設")
