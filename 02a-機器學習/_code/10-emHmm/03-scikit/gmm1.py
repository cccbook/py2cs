from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# 生成數據
np.random.seed(0)
n_samples = 1000
# 兩個高斯分佈的數據
X = np.concatenate([np.random.normal(0, 1, (n_samples // 2, 2)),
                   np.random.normal(5, 1, (n_samples // 2, 2))])

# 創建 GMM 模型
gmm = GaussianMixture(n_components=2, covariance_type='full')

# 擬合模型
gmm.fit(X)

# 預測每個點屬於哪個組別
labels = gmm.predict(X)

# 顯示結果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("GMM Clustering")
plt.show()

# 查看高斯模型的參數
print("Means:\n", gmm.means_)
print("Covariances:\n", gmm.covariances_)
print("Weights:\n", gmm.weights_)
