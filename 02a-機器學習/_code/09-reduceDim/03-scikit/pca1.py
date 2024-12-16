from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# 創建示例數據集
X = np.random.randn(100, 10)  # 100 個樣本，10 個特徵

# 創建 PCA 模型
pca = PCA(n_components=2)

# 擬合並轉換數據
X_pca = pca.fit_transform(X)

# 可視化結果
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA - 2D Projection")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
