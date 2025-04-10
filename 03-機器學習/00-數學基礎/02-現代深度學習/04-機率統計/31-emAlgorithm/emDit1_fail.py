import numpy as np

# 定义模型中混合的高斯分布的数量
n_components = 3

# 初始化模型的参数
weights = np.random.uniform(0, 1, n_components)
weights /= weights.sum()
means = np.random.uniform(-10, 10, size=(n_components, 2))
covariances = np.array([np.eye(2)] * n_components)

# 迭代更新模型的参数
for i in range(1000):
    # E-step：计算每个样本属于每个高斯分布的概率
    probs = []
    for j in range(n_components):
        prob = weights[j] * multivariate_normal(means[j], covariances[j]).pdf(X)
        probs.append(prob)
    probs = np.array(probs)
    probs /= probs.sum(axis=0)

    # M-step：更新模型的参数
    weights = probs.sum(axis=1) / X.shape[0]
    means = np.dot(probs, X) / probs.sum(axis=1)[:, None]
    covariances = np.zeros((n_components, 2, 2))
    for j in range(n_components):
        covariances[j] = np.cov(X.T, aweights=probs[j])

# 输出模型的参数
print(weights)
print(means)
print(covariances)
