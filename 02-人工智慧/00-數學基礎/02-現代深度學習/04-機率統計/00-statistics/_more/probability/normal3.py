import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

n = 10000
r = norm.rvs(size=n)
min, max, len = -5, 5, 40
bins = np.linspace(min, max, len)
count, bins, ignored = plt.hist(r, bins=bins, rwidth=0.9) # 統計長條圖
plt.plot(bins, norm.pdf(bins)*n*(max-min)/len) # 曲線圖
plt.show()