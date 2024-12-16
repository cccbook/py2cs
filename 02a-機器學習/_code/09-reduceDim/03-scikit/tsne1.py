from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# 創建示例數據集
X = np.random.randn(100, 10)  # 100 個樣本，10 個特徵

# 創建 t-SNE 模型
tsne = TSNE(n_components=2)

# 擬合並轉換數據
X_tsne = tsne.fit_transform(X)

# 可視化結果
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title("t-SNE - 2D Projection")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
