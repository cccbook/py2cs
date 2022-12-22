
## ccc: 那改給我你記憶中的 python 範例好了，隨便一個都行

下面是一個簡單的 python 程式，可以印出 "Hello, World!"：

```py
# 定義一個函數，可以印出 "Hello, World!"
def hello_world():
  print("Hello, World!")

# 呼叫函數
hello_world()

```

這個程式的運作流程是：首先，我們在程式的第一行定義了一個函數 hello_world()，這個函數會印出 "Hello, World!"。然後，在程式的第五行，我們呼叫了這個函數，讓它執行。在執行完這個函數後，程式就結束了。

## ccc: 那給我 python 的雙層迴圈範例

```py
for i in range(0, 5):
    for j in range(0, 5):
        print(i, j)

```

輸出結果為:

```
(0, 0)
(0, 1)
(0, 2)
(0, 3)
(0, 4)
(1, 0)
(1, 1)
(1, 2)
(1, 3)
(1, 4)
(2, 0)
(2, 1)
(2, 2)
(2, 3)
(2, 4)
(3, 0)
(3, 1)
(3, 2)
(3, 3)
(3, 4)
(4, 0)
(4, 1)
(4, 2)
(4, 3)
(4, 4)

```

## ccc: 用 python 做查表的範例呢?

這是一個使用 python 實現查表的範例：

```py
# 建立一個名為 "table" 的字典，這個字典就是我們的查表表格
table = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

# 接下來，我們就可以透過 key 值 (例如 "A"、"B"、"C" 等) 來查詢 value 值
print(table["A"])  # 輸出 1
print(table["B"])  # 輸出 2

# 我們也可以利用迴圈來輸出所有 key-value 對
for key, value in table.items():
  print(key, value)

# 輸出結果：
# A 1
# B 2
# C 3
# D 4
# E 5

```

此範例中，我們透過建立一個字典來實現查表功能。我們可以透過 key 值 (例如 "A"、"B"、"C" 等) 來查詢 value 值，並且也可以使用迴圈來輸出所有的 key-value 對。

## ccc: 用 python 蒙地卡羅法計算圓周率

可以使用下列程式碼來計算圓周率：

```py
import random

n = 1000000
inside = 0

for i in range(n):
    x = random.uniform(0, 1)
    y = random.uniform(0, 1)
    if x**2 + y**2 < 1:
        inside += 1

pi = 4 * inside / n
print(pi)

```

該程式會隨機產生一定數量的點（這裡是 1000000 個），並判斷它們是否在圓的內部。若在圓的內部，則加一到計數器。最後，通過計算點的數量求出圓周率。

## ccc: 那怎麼用 python 實作 MCMC (Monte Carlo Markov Chain) 呢?

以下是用 python 實作 MCMC 的一個範例：

```py
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

```

在這個範例中，我們用一組樣本來估計平均值和標準差。我們假設初始的平均值為 0，標

## ccc: 有更複雜的 MCMC 範例嗎?

用 python 實作 MCMC 的範例很多，但可能會因為您的問題和需求而有所不同。我可以為您提供一個基礎的範例，您可以按照自己的需求進行擴展和修改。

以下是一個簡單的 MCMC 範例，計算一個隨機樣本的平均值：

```py
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

```

在這個範例中，我們每次模擬都隨機抽樣一個數字，然後根據當前的估計值和抽到的數字來計算新的估計值。這樣的模擬過程最終

## ccc: 那用 python 做 gibbs sampling 怎麼做?

Gibbs sampling 是一種 MCMC 演算法，可以用來從一個含有多個隨機變數的機率分佈中抽樣。以下是一個使用 Gibbs sampling 求解一個簡單的高斯混合模型的範例：

```py
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

```
