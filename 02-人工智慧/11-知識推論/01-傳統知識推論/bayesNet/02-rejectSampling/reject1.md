https://chatgpt.com/c/67076859-667c-8012-b79d-0e255ebd2758

## ccc: 什麼是 Reject Sampling 

**Reject Sampling**（拒絕抽樣）是一種在統計學和機器學習中的蒙地卡羅方法，用來從難以直接抽樣的目標分佈（target distribution）中生成樣本。這種方法通過使用一個容易抽樣的分佈來近似目標分佈，然後拒絕某些不符合條件的樣本，直到得到來自目標分佈的有效樣本為止。

具體步驟如下：

1. **選擇一個提議分佈**（proposal distribution）\( q(x) \)，這個分佈應該容易從中生成樣本。
2. **設置一個常數** \( M \)，這個常數需要滿足 \( p(x) \leq M \cdot q(x) \) 對於所有的 \( x \) 成立，其中 \( p(x) \) 是目標分佈。
3. 從 \( q(x) \) 中抽樣 \( x \)。
4. 根據一個概率 \( \frac{p(x)}{M \cdot q(x)} \) 決定是否接受這個樣本。這個概率是目標分佈和提議分佈之間的比率。
   - 如果接受，則將 \( x \) 作為目標分佈 \( p(x) \) 的樣本。
   - 如果拒絕，則重複抽樣。

Reject Sampling 的核心思想是使用一個容易抽樣的分佈來近似目標分佈，並且只保留那些與目標分佈一致的樣本。

### 優缺點
- **優點**：在某些情況下，它能夠有效地生成目標分佈的樣本，尤其當目標分佈較為複雜且難以直接抽樣時。
- **缺點**：當提議分佈與目標分佈的差異較大時，可能會拒絕大量樣本，導致效率低下。

Reject Sampling 是生成樣本的一種通用技術，經常用於貝葉斯推斷和蒙地卡羅方法中。

## ccc: 請給一個 Reject Sampling 的範例並用 python 實作

這裡我們將以一個簡單的範例來展示如何使用 Reject Sampling 從目標分佈（例如，標準正態分佈）生成樣本。提議分佈我們可以選擇為均勻分佈，因為它的樣本容易生成。

### 範例說明：
我們想從標準正態分佈 \( p(x) \sim \mathcal{N}(0, 1) \) 中生成樣本，但假設我們無法直接生成樣本。取而代之的是使用一個簡單的提議分佈 \( q(x) \)，這裡選擇均勻分佈 \( U(-3, 3) \)，並使用 Reject Sampling 來生成正態分佈的樣本。

### Python 實作

```python
import numpy as np
import matplotlib.pyplot as plt

# 目標分佈：標準正態分佈
def target_distribution(x):
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

# 提議分佈：均勻分佈
def proposal_distribution(size=1):
    return np.random.uniform(-3, 3, size)

# 常數 M：用於縮放提議分佈，這裡選擇 M 大於或等於目標分佈與提議分佈比值的最大值
M = 2.5

# Reject Sampling
def reject_sampling(num_samples):
    samples = []
    while len(samples) < num_samples:
        # 從提議分佈中抽樣
        x = proposal_distribution()
        
        # 計算接受概率
        acceptance_ratio = target_distribution(x) / (M * 1/6)  # 提議分佈均勻分佈 U(-3, 3) 的概率密度是 1/6
        
        # 擲骰子決定是否接受樣本
        if np.random.uniform(0, 1) < acceptance_ratio:
            samples.append(x[0])
    
    return np.array(samples)

# 生成 10000 個樣本
num_samples = 10000
samples = reject_sampling(num_samples)

# 畫出結果
x = np.linspace(-4, 4, 1000)
plt.figure(figsize=(10, 6))

# 標準正態分佈的理論曲線
plt.plot(x, target_distribution(x), label="Target Distribution (Normal)", color='red', lw=2)

# 使用 Reject Sampling 生成的樣本的直方圖
plt.hist(samples, bins=50, density=True, alpha=0.6, color='blue', label="Samples from Reject Sampling")

plt.title("Reject Sampling from Standard Normal Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()
```

### 代碼說明：
1. **目標分佈**：定義標準正態分佈的概率密度函數 `target_distribution(x)`。
2. **提議分佈**：使用均勻分佈 \( U(-3, 3) \) 作為提議分佈，這裡的概率密度是常數 \( 1/6 \)。
3. **Reject Sampling 過程**：我們重複從提議分佈中抽樣，計算接受樣本的概率，並決定是否接受。如果接受，將該樣本存入樣本集中。
4. **結果可視化**：我們生成的樣本應該接近標準正態分佈，並將其與理論上的標準正態分佈進行比較。

### 輸出：
此代碼會生成一個圖形，顯示使用 Reject Sampling 生成的樣本的分佈（直方圖）和理論上的標準正態分佈（紅色曲線）。

## ccc: reject samping 背後的數學理論是什麼？為何根據 步驟 4 就能產生 q 分佈的樣本

**Reject Sampling** 背後的數學理論主要基於概率論的「接受-拒絕」策略。其核心原理是通過從提議分佈中生成樣本，然後根據特定的概率決定是否接受這些樣本，以達到目標分佈的樣本生成效果。下面來分步解釋 Reject Sampling 背後的數學推導。

### 1. **分佈與比例關係**：
我們有兩個分佈：
- **目標分佈** \( p(x) \)：我們想從這個分佈中抽樣，但假設很難直接抽樣。
- **提議分佈** \( q(x) \)：這個分佈相對簡單，易於抽樣。

我們希望從 \( p(x) \) 中生成樣本，於是我們使用 \( q(x) \) 來近似 \( p(x) \)，並且引入一個常數 \( M \)，使得：
\[
p(x) \leq M \cdot q(x)
\]
這意味著，對於所有 \( x \)，\( M \cdot q(x) \) 是目標分佈 \( p(x) \) 的上界。

### 2. **接受概率的設計**：
當從 \( q(x) \) 中抽取樣本 \( x \) 時，為了確保最終樣本遵循 \( p(x) \)，我們設計一個接受概率：
\[
\text{接受概率} = \frac{p(x)}{M \cdot q(x)}
\]
這個概率的設計基於將目標分佈 \( p(x) \) 和提議分佈 \( q(x) \) 之間的比例進行縮放。換句話說，如果 \( p(x) \) 與 \( q(x) \) 非常接近，則該樣本更有可能被接受；如果 \( p(x) \) 與 \( q(x) \) 差異很大，則樣本被拒絕的可能性更高。

### 3. **抽樣過程的合法性**：
Reject Sampling 的合法性來自於這樣一個觀察：通過這樣的接受-拒絕過程，我們保證在足夠長的運行時間內，從 \( p(x) \) 和 \( q(x) \) 中被接受的樣本比例會趨近於它們的理論概率比值。

具體來說，我們從 \( q(x) \) 中抽取樣本 \( x \)，然後根據 \( \frac{p(x)}{M \cdot q(x)} \) 來決定是否接受樣本。這樣做的結果是，我們有效地篩選出符合目標分佈 \( p(x) \) 的樣本，並排除那些不符合的樣本。

### 4. **數學推導過程**：
假設 \( x \) 來自 \( q(x) \) 並且被接受的概率是 \( \frac{p(x)}{M \cdot q(x)} \)，那麼被接受的樣本的總體分佈可以用下面的期望來描述：

令 \( A(x) \) 表示樣本 \( x \) 被接受的概率，
\[
A(x) = \frac{p(x)}{M \cdot q(x)}
\]
則 \( x \) 被接受並遵循分佈的密度是：
\[
q(x) \cdot A(x) = q(x) \cdot \frac{p(x)}{M \cdot q(x)} = \frac{p(x)}{M}
\]
這意味著，經過這個接受過程後，樣本的密度是 \( \frac{1}{M} \cdot p(x) \)，它與目標分佈 \( p(x) \) 成比例。當我們最終只保留那些被接受的樣本時，這些樣本將遵循目標分佈 \( p(x) \)。

### 5. **為何步驟 4 能生成 \( p(x) \) 分佈的樣本**？
步驟 4 中的接受概率 \( \frac{p(x)}{M \cdot q(x)} \) 保證了生成的樣本遵循目標分佈 \( p(x) \)。具體原因在於：
- 每次抽取樣本時，我們使用 \( q(x) \) 來生成樣本，然後根據 \( p(x) \) 和 \( q(x) \) 的相對大小來接受或拒絕該樣本。
- 被接受的樣本的分佈密度正比於 \( p(x) \)，而被拒絕的樣本則不會影響最終結果。

當我們重複這個過程足夠多次時，剩下來的樣本就是從 \( p(x) \) 分佈中生成的。

### 總結：
Reject Sampling 背後的數學原理是通過提議分佈 \( q(x) \) 生成樣本，然後根據一個比例概率決定是否接受該樣本。這個概率確保生成的樣本最終服從目標分佈 \( p(x) \)。換句話說，這個方法利用了提議分佈來進行樣本生成，並透過接受或拒絕來修正樣本，使其逼近目標分佈。