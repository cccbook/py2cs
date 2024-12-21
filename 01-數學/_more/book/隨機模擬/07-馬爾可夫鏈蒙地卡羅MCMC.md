### 第七章：馬爾可夫鏈蒙地卡羅（MCMC）

馬爾可夫鏈蒙地卡羅（MCMC）是一種強大的隨機模擬技術，廣泛應用於統計學、機器學習和物理等領域。MCMC 的主要思想是通過構建一個馬爾可夫鏈，來生成符合目標概率分佈的樣本。這一章將介紹 MCMC 的基本概念以及一些常見的 MCMC 演算法。

---

##### MCMC 的基本概念

1. **馬爾可夫鏈的概念**：
   - 馬爾可夫鏈是一種隨機過程，其中當前狀態的轉移僅依賴於前一狀態，而與過去的狀態無關。這種性質被稱為「馬爾可夫性」。
   - 在 MCMC 中，我們希望通過馬爾可夫鏈的穩態分佈來近似某一目標分佈 \(P(x)\)。

2. **MCMC 的過程**：
   - **初始化**：從某個初始狀態開始。
   - **迭代**：在每一步中，根據當前狀態生成一個新的候選狀態，然後根據一定的接受機率決定是否接受這個新狀態。
   - **收斂**：當馬爾可夫鏈達到穩態時，生成的樣本將近似於目標分佈。

3. **接受-拒絕機制**：
   - 在每次迭代中，我們使用接受-拒絕機制來決定是否接受新狀態 \(x'\)：
   - 計算接受機率：

   \[
   \alpha = \min\left(1, \frac{P(x')Q(x|x')}{P(x)Q(x'|x)}\right)
   \]

   - 若一個隨機數 \(u \sim U(0, 1)\) 小於 \(\alpha\)，則接受新狀態，否則保持在舊狀態。

---

##### 常見的 MCMC 演算法

1. **隨機行走馬爾可夫鏈（Random Walk MCMC）**：
   - 隨機行走是最基本的 MCMC 方法。它通過在當前狀態附近隨機選擇候選狀態來生成樣本。
   - 儘管簡單，但在高維空間中，隨機行走可能會導致低效的收斂。

2. **吉布斯取樣（Gibbs Sampling）**：
   - 吉布斯取樣是一種特殊的 MCMC 方法，適用於多變量分佈。該方法通過依次取樣每個變量的條件分佈來生成樣本。
   - 對於變量 \(x_1, x_2, \ldots, x_n\)，吉布斯取樣的步驟如下：
     1. 從條件分佈 \(P(x_1 | x_2, \ldots, x_n)\) 中取樣。
     2. 從條件分佈 \(P(x_2 | x_1, \ldots, x_n)\) 中取樣。
     3. 重複上述步驟直到收斂。

3. **Metropolis-Hastings 演算法**：
   - Metropolis-Hastings 演算法是一種通用的 MCMC 方法，可以適用於任意目標分佈。其基本步驟包括：
     1. 從當前狀態 \(x\) 生成候選狀態 \(x'\)。
     2. 計算接受機率 \(\alpha\)。
     3. 根據接受機率決定是否接受新狀態。

4. **Hamiltonian Monte Carlo (HMC)**：
   - HMC 是一種利用物理學中的哈密頓動力學的 MCMC 方法，通過引入動量來生成樣本。這種方法在高維空間中的表現優於隨機行走。
   - 它的步驟通常包括：
     1. 隨機生成動量。
     2. 使用哈密頓動力學推進狀態。
     3. 使用 Metropolis-Hastings 機制進行接受-拒絕。

---

### MCMC 的 Python 實作範例

接下來，我們將展示如何使用 Python 實現簡單的 Metropolis-Hastings 演算法。假設我們想要從一個一維的標準正態分佈中生成樣本。

```python
import numpy as np
import matplotlib.pyplot as plt

# 目標分佈的概率密度函數（標準正態分佈）
def target_distribution(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

# Metropolis-Hastings 演算法
def metropolis_hastings(num_samples, proposal_width):
    samples = []
    current_sample = 0  # 初始化當前樣本

    for _ in range(num_samples):
        # 生成候選樣本
        proposed_sample = np.random.normal(current_sample, proposal_width)
        
        # 計算接受機率
        acceptance_ratio = target_distribution(proposed_sample) / target_distribution(current_sample)
        
        # 隨機生成 u
        u = np.random.uniform(0, 1)
        
        # 根據接受機率決定是否接受候選樣本
        if u < acceptance_ratio:
            current_sample = proposed_sample  # 接受候選樣本
        
        samples.append(current_sample)

    return np.array(samples)

# 模擬設定
num_samples = 10000
proposal_width = 1.0

# 獲取樣本
samples = metropolis_hastings(num_samples, proposal_width)

# 繪製結果
plt.hist(samples, bins=30, density=True, alpha=0.5, color='blue', label='MCMC 樣本')
x = np.linspace(-4, 4, 1000)
plt.plot(x, target_distribution(x), color='red', label='目標分佈（標準正態）')
plt.title('Metropolis-Hastings MCMC 模擬')
plt.xlabel('樣本值')
plt.ylabel('密度')
plt.legend()
plt.show()
```

在這段代碼中，我們使用 Metropolis-Hastings 演算法從標準正態分佈生成樣本，並可視化模擬結果。透過這種方式，我們可以看到生成的樣本與目標分佈的對比。

---

這一章介紹了馬爾可夫鏈蒙地卡羅（MCMC）的基本概念及其常見演算法，並展示了如何使用 Python 實作 MCMC 方法。接下來的章節將深入探討其他進階的隨機模擬技術及其應用。