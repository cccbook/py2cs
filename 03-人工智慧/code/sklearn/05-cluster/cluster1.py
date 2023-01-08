# https://ithelp.ithome.com.tw/articles/10207518
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
import sklearn.cluster as skcl
import numpy as np

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=3,
                       cluster_std=0.60, random_state=0)
# y 代表該點屬於哪一群
# plt.scatter(X[:, 0], X[:, 1], c=y) # , s=50
plt.scatter(X[:, 0], X[:, 1]) # , s=50
plt.show()

# 假設我們希望將資料分成 3 個 cluster
# model = skcl.KMeans(n_clusters=3)
# model = skcl.AgglomerativeClustering(n_clusters=3)
# model = skcl.DBSCAN(eps=0.5, min_samples=3)
# model = skcl.SpectralClustering(n_clusters=3)
model = skcl.AffinityPropagation(damping=0.5, max_iter=200)

# 將資料集放入模型中進行訓練
model.fit(X)

# 透過模型對每個樣本進行預測，得到每個樣本所屬的 cluster
# predictions = model.predict(X)
predictions = model.fit_predict(X)
# 繪製散佈圖
plt.scatter(X[:, 0], X[:, 1], c=predictions)
plt.show()
