from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 模擬數據
X = np.random.rand(100, 5)  # 100個樣本，5個特徵

# PCA 降維
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# 顯示降維後的數據
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.title("PCA降維結果")
plt.show()
