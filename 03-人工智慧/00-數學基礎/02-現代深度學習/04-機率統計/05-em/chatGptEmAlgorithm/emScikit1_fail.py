
from sklearn.mixture import GaussianMixture

# 定义模型，并指定模型中混合的高斯分布的数量
model = GaussianMixture(n_components=3)

# 拟合模型
model.fit(X)

# 输出模型的参数
print(model.weights_)
print(model.means_)
print(model.covariances_)
