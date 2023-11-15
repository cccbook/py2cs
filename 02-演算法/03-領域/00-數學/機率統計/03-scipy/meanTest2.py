# https://statistics-using-python.blogspot.com/2019/08/t-two-sample-t-test-with-equal.html
import numpy as np
from scipy import stats

# 假設我們有兩組數據，並想要檢驗它們是否具有相同的平均值
# data1 = [1, 2, 3, 4, 5]
data1 = np.random.normal(10, 2, 10)

# data2 = [5, 6, 7, 8, 9]
data2 = np.random.normal(10, 2, 10)

# 使用 scipy 的 ttest_ind() 函數來進行雙樣本平均值檢定
# 其中，equal_var 參數指定是否假設兩組數據具有相同的變異數
t, p = stats.ttest_ind(data1, data2, equal_var=True)

# 打印檢驗統計量和 p 值
print("t-statistic: ", t)
print("p-value: ", p)

# 根據 p 值，決定是否拒絕預期的平均值假設
if p < 0.05:
    print("拒絕預期的平均值假設，數據不來自相同的平均值")
else:
    print("不能拒絕預期的平均值假設，數據可能來自相同的平均值")
