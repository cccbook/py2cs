import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 生成數據
np.random.seed(42)
data = np.concatenate([np.random.normal(loc=-2, scale=0.5, size=300),
                       np.random.normal(loc=3, scale=1.0, size=700)])

# EM 演算法實現
def em_algorithm(data, num_components, num_iterations):
    # 隨機初始化參數
    weights = np.ones(num_components) / num_components
    means = np.random.choice(data, num_components)
    variances = np.random.random(num_components)

    for _ in range(num_iterations):
        # E 步驟：計算每個數據點的責任度
        responsibilities = np.zeros((len(data), num_components))
        for k in range(num_components):
            responsibilities[:, k] = weights[k] * norm.pdf(data, means[k], np.sqrt(variances[k]))
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # M 步驟：更新參數
        for k in range(num_components):
            N_k = responsibilities[:, k].sum()
            weights[k] = N_k / len(data)
            means[k] = (responsibilities[:, k] @ data) / N_k
            variances[k] = (responsibilities[:, k] @ (data - means[k])**2) / N_k

    return weights, means, variances

# 執行 EM 演算法
num_components = 2
num_iterations = 100
weights, means, variances = em_algorithm(data, num_components, num_iterations)
print('weights=', weights)
print('means=', means)
print('variances=', variances)

# 繪製結果
x = np.linspace(-5, 6, 1000)
pdf = sum(w * norm.pdf(x, m, np.sqrt(v)) for w, m, v in zip(weights, means, variances))

plt.hist(data, bins=30, density=True, alpha=0.5, color='gray', label='Data Histogram')
plt.plot(x, pdf, label='GMM PDF', color='red')
for k in range(num_components):
    plt.plot(x, weights[k] * norm.pdf(x, means[k], np.sqrt(variances[k])), label=f'Component {k+1}')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('EM Algorithm for Gaussian Mixture Model')
plt.legend()
plt.show()
