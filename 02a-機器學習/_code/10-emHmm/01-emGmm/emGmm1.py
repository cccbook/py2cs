from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# 生成樣本數據
np.random.seed(0)
X = np.concatenate([
    np.random.normal(loc=-5, scale=1, size=(100, 2)),
    np.random.normal(loc=5, scale=1, size=(100, 2))
])

# 創建 GMM 模型並擬合數據
gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(X)

# 預測每個數據點的類別
labels = gmm.predict(X)

# 可視化結果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='.')
plt.title("GMM Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 查看 GMM 模型的參數
print("Means:\n", gmm.means_)
print("Covariances:\n", gmm.covariances_)
print("Weights:\n", gmm.weights_)
