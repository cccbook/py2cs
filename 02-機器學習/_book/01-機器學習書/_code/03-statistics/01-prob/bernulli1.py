import numpy as np
import matplotlib.pyplot as plt

# 模擬1000個來自伯努利分佈的隨機數（成功的概率為0.5）
p = 0.5  # 成功的概率
samples = np.random.binomial(1, p, 1000)

# 可視化結果
plt.hist(samples, bins=2, density=True, alpha=0.6, color='r')
plt.title('伯努利分佈 (Bernoulli Distribution)')
plt.show()
