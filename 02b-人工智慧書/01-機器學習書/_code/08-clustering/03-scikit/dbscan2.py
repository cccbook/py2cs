from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# 創建一個示例數據集
X = np.random.randn(100, 2)  # 100 個隨機生成的 2D 點

# 使用 DBSCAN 進行聚類
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)

# 可視化結果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("DBSCAN Clustering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
