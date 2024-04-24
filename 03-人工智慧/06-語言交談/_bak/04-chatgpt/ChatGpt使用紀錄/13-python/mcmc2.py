import random

# 假設我們想要計算一個隨機樣本的平均值
# 我們先建立一個隨機樣本
sample = [random.random() for i in range(1000)]

# 初始化模擬參數
# 假設我們的初始估計值為 0，模擬次數為 10000 次
n_simulations = 10000
current_estimate = 0

# 開始模擬
for i in range(n_simulations):
  # 隨機抽樣一個數字
  sample_index = random.randint(0, len(sample) - 1)
  sample_value = sample[sample_index]

  # 計算新的估計值
  current_estimate = (current_estimate * i + sample_value) / (i + 1)

# 輸出最終的估計值
print(current_estimate)
