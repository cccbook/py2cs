# https://statistics-using-python.blogspot.com/2019/08/comparing-multiple-proportions.html
import numpy as np
import statsmodels.stats.proportion as proportion

x = [32,43,16,9]
y = [87,108,80,25]

f, p, _ = proportion.proportions_chisquare(x, y)
# 打印檢驗統計量和 p 值
print("f-statistic: ", f)
print("p-value: ", p)

# 根據 p 值，決定是否拒絕預期的比例值假設
if p < 0.05:
    print("拒絕預期的比例值假設，數據不來自相同的比例值")
else:
    print("不能拒絕預期的比例值假設，數據可能來自相同的比例值")
