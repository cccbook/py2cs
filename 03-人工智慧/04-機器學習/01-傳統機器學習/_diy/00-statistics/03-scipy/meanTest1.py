# 參考 -- https://statistics-using-python.blogspot.com/2019/08/t-one-sample-t-testparametric.html
import numpy as np
from scipy import stats

# 假設我們有一組數據，並想要檢驗它是否來自平均值為 100 的分佈
# data = [101, 102, 103, 104, 105]
# 設定平均值為 100，標準差為 2
mu, sigma = 100, 2

# 使用 numpy 的 random.normal() 函數產生 10 個常態分佈亂數
data = np.random.normal(mu, sigma, 10)

# 計算數據的平均值
mean = np.mean(data)

# 使用 scipy 的 ttest_1samp() 函數來進行平均值檢定
# 其中，popmean 參數指的是預期的平均值，也就是 100
t, p = stats.ttest_1samp(data, popmean=100)

# 打印檢驗統計量和 p 值
print("t-statistic: ", t)
print("p-value: ", p)

# 根據 p 值，決定是否拒絕預期的平均值假設
if p < 0.05:
    print("拒絕預期的平均值假設，數據不來自平均值為 100 的分佈")
else:
    print("不能拒絕預期的平均值假設，數據可能來自平均值為 100 的分佈")
