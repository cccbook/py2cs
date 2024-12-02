import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
x = np.linspace(-5, 5, 50)

ax.plot(x, norm.pdf(x)) # normal 曲線

r = norm.rvs(size=10000) # 統計長條圖
ax.hist(r, bins=np.linspace(-5, 5, 20), rwidth=0.9)
# ax.hist(r, bins=20, rwidth=0.9)

plt.show()