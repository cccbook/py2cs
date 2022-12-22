import numpy as np

# 設定模型參數
mu1 = 0
mu2 = 5
sigma1 = 1
sigma2 = 3

# 定義高斯混合模型的概率密度函數
def p(x, mu1, mu2, sigma1, sigma2):
  return 0.3 * np.exp(-0.5 * (x - mu1)**2 / sigma1**2) + 0.7 * np.exp(-0.5 * (x - mu2)**2 / sigma2**2)

# 初始化抽樣序列
x = [0]

# 進行抽樣
for i in range(1000):
  # 根據當前抽樣值更新 mu1
  mu1 = np.random.normal(mu1, sigma1)
  # 根據當前抽樣值更新 mu2
  mu2 = np.random.normal(mu2, sigma2)
  # 根據當前抽樣值和新的 mu1, mu2 計算接受機率
  acceptance_probability = p(x[-1], mu1, mu2, sigma1, sigma2) / p(x[-1], x[-1], mu2, sigma1, sigma2)
  # 根據接受機率決定是否接受新的抽樣值
  if np.random.uniform() < acceptance_probability:
    x.append(mu1)
  else:
    x.append(x[-1])

# 顯示抽樣序列
print(x)
