import umap
import numpy as np
import matplotlib.pyplot as plt

# 創建示例數據集
X = np.random.randn(100, 10)  # 100 個樣本，10 個特徵

# 創建 UMAP 模型
umap_model = umap.UMAP(n_components=2)

# 擬合並轉換數據
X_umap = umap_model.fit_transform(X)

# 可視化結果
plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.title("UMAP - 2D Projection")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
