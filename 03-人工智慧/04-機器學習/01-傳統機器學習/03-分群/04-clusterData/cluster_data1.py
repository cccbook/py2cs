# https://ithelp.ithome.com.tw/articles/10207518
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
n = 300
X, y = datasets.make_blobs(n_samples=n, centers=4, cluster_std=0.60, random_state=0)
#X, y = datasets.make_moons(n_samples=n, noise=0.1)
#X, y = datasets.make_circles(n_samples=n, noise=0.1, factor=0.5)
#X, y = datasets.make_circles(n_samples=n, noise=0.01, factor=0.5)
#X, y = np.random.rand(n, 2), None
plt.scatter(X[:, 0], X[:, 1]) # , s=50
plt.show()

# X, y = skdt.make_classification(n_samples=n, n_features=10, n_informative=5, n_redundant=5, n_classes=2)
# X, y = datasets.make_regression(n_samples=n, n_features=10)
