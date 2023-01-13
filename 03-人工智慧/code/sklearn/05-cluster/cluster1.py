# https://ithelp.ithome.com.tw/articles/10207518
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, cluster

n=300
# X, y = datasets.make_blobs(n_samples=n, centers=3, cluster_std=0.60, random_state=0)
# X, y = datasets.make_moons(n_samples=n, noise=0.1)
# X, y = datasets.make_circles(n_samples=n, noise=0.1, factor=0.5)
X, y = np.random.rand(n, 2), None
# y 代表該點屬於哪一群，分群時沒用到

# plt.scatter(X[:, 0], X[:, 1], c=y) # , s=50
plt.scatter(X[:, 0], X[:, 1]) # , s=50
plt.show()

# 假設我們希望將資料分成 3 個 cluster
model = cluster.KMeans(n_clusters=3)
# model = cluster.AgglomerativeClustering(n_clusters=3)
# model = cluster.DBSCAN(eps=0.5, min_samples=3)
# model = cluster.SpectralClustering(n_clusters=3)
# model = cluster.AffinityPropagation(damping=0.5, max_iter=200)

# 將資料集放入模型中進行訓練
model.fit(X)

# 透過模型對每個樣本進行預測，得到每個樣本所屬的 cluster
# predictions = model.predict(X)
predictions = model.fit_predict(X)
# 繪製散佈圖
plt.scatter(X[:, 0], X[:, 1], c=predictions)
plt.show()
