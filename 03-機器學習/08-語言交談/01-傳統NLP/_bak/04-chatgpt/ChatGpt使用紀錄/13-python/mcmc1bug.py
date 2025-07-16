import random

# 產生一個包含隨機樣本的函數
def random_sample(mean, std):
  sample = random.gauss(mean, std)
  return sample

# 定義 MCMC 演算法的迭代函數
def mcmc_iteration(mean, std, samples):
  # 根據樣本產生一個新的值
  new_mean = random_sample(mean, std)
  new_std = random_sample(std, std)

  # 計算新的樣本與原樣本的差異
  diff = 0
  for sample in samples:
    diff += (sample - new_mean) ** 2
  diff /= len(samples)

  # 計算新的樣本與原樣本的比值
  ratio = diff / (mean ** 2 + std ** 2)

  # 如果新的樣本比原樣本好，就接受新樣本
  # 否則以一定概率接受新樣本
  if ratio > 1 or random.random() < ratio:
    mean = new_mean
    std = new_std

  return mean, std

# 主程式
if __name__ == "__main__":
  # 產生樣本
  samples = [random.gauss(5, 2) for i in range(1000)]

  # 設定初始的樣本值
  mean = 0
  std = 1

  # 迭代 100 次
  for i in range(100):
    mean, std = mcmc_iteration(mean, std, samples)

  # 輸出結果
  print("Mean:", mean)
  print("Std:", std)
