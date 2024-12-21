from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt

# 創建一個示例數據集
X = np.random.randn(300, 2)  # 300 個隨機生成的 2D 點

# 設置 K 值，這裡選擇 3 個簇
spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)

# 擬合模型並預測簇標籤
labels = spectral.fit_predict(X)

# 可視化結果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("Spectral Clustering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
