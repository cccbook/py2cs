import numpy as np
import matplotlib.pyplot as plt

# 模擬1000個從均勻分佈中生成的隨機數
samples = np.random.uniform(low=0, high=10, size=1000)

# 可視化結果
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g')
plt.title('均勻分佈 (Uniform Distribution)')
plt.show()
