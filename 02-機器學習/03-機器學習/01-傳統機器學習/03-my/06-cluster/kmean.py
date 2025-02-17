# 來源 -- https://dotblogs.com.tw/kevinya/2018/06/15/105548
# 進行數據分析之前常要引用的函式庫
import numpy as np
import matplotlib.pyplot as plt

# 產生100筆資料，每筆資料都是2個數字
X = np.random.rand(100,2)

#畫出來看看，想當然是平均的佈滿整個畫面
#然後我們會用KMeans硬把他分類(明明沒意義的100個點……但他就是分的出來)
plt.scatter(X[:,0],X[:,1],s=50)
plt.show()

#接下來匯入KMeans函式庫
from sklearn.cluster import KMeans

#請KMeans分成三類
clf = KMeans(n_clusters=3)

#開始訓練！
clf.fit(X)

# 這樣就可以取得預測結果了！
clf.labels_

# 最後畫出來看看
# 真的分成三類！太神奇了………無意義的資料也能分～
plt.scatter(X[:,0],X[:,1], c=clf.labels_)
# KMeans的使用時機就在於～你根本不知道測試的資料有什麼特性的時候
# 就是用他的時候了，我稱KMeans為盲劍客 XD
plt.show()
