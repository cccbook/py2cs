# https://statistics-using-python.blogspot.com/2019/08/one-proportion-test.html
import numpy as np
import statsmodels.stats.proportion as proportion

# 12及30是觀測數據，亦即觀測到的比例是12/30。
f, p, _ = proportion.proportions_chisquare(12, 30, value = 0.5)
# 打印檢驗統計量和 p 值
print("f-statistic: ", f)
print("p-value: ", p)

# 根據 p 值，決定是否拒絕預期的比例值假設
if p < 0.05:
    print("拒絕預期的比例值假設，數據不來自相同的比例值")
else:
    print("不能拒絕預期的比例值假設，數據可能來自相同的比例值")
