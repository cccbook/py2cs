from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 創建一個示例數據集
X = np.random.randn(300, 2)  # 300 個隨機生成的 2D 點

# 設置 K 值，這裡選擇 3 個簇
kmeans = KMeans(n_clusters=3, random_state=42)

# 擬合模型並預測簇標籤
labels = kmeans.fit_predict(X)

# 可視化結果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.title("K-Means Clustering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()
