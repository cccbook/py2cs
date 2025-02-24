import numpy as np
import matplotlib.pyplot as plt


from sklearn.mixture import GaussianMixture

# 創建一個 GMM 模型，並擬合兩個高斯分佈
gmm = GaussianMixture(n_components=2, covariance_type='full')
X = gmm.sample(1000)[0]  # 生成1000個樣本

# 可視化結果
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, color='purple')
plt.title('高斯混合模型 (Gaussian Mixture Model)')
plt.show()
