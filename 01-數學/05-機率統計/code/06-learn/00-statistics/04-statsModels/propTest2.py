import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# 資料來源 https://statistics-using-python.blogspot.com/2019/08/comparing-two-proportions.html
x = [18,10]
y = [24,25]

# 使用 statsmodels 的 proportions_ztest() 函數來進行比例檢定
# 其中，count 參數指定兩組樣本的成功次數，nobs 參數指定每組樣本的總數
z, p = proportions_ztest(x, y, alternative='two-sided')

# 打印檢驗統計量和 p 值
print("z-statistic: ", z)
print("p-value: ", p)

# 根據 p 值，決定是否拒絕預期的比例值假設
if p < 0.05:
    print("拒絕預期的比例值假設，數據不來自相同的比例值")
else:
    print("不能拒絕預期的比例值假設，數據可能來自相同的比例值")

