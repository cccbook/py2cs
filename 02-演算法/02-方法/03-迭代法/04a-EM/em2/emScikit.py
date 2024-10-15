import numpy as np

# 生成三群具有不同常態分布的随机数
data1 = np.random.normal(0, 1, (1000, 2))
data2 = np.random.normal(10, 2, (1000, 2))
data3 = np.random.normal(-5, 3, (1000, 2))

# 将三群随机数合并成一个数组
X = np.concatenate((data1, data2, data3), axis=0)

from sklearn.mixture import GaussianMixture

# 定义模型，并指定模型中混合的高斯分布的数量
model = GaussianMixture(n_components=3)

# 拟合模型
model.fit(X)

# 输出模型的参数
print(model.weights_)
print(model.means_)
print(model.covariances_)
