### **高斯混合模型（GMM）與期望最大化算法（EM）**

#### **高斯混合模型（GMM）**

高斯混合模型（Gaussian Mixture Model，GMM）是一種生成模型，它假設數據集是由若干個高斯分佈組成的，每個高斯分佈對應數據中的一個簇或類別。GMM 是一種概率模型，它能夠根據數據的分佈來估計各個高斯成分的參數，包括均值、協方差和權重。

GMM 的數學表達式為：

\[
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)
\]

其中：
- \( p(x) \) 是數據點 \( x \) 的概率密度。
- \( K \) 是高斯分佈的數量。
- \( \pi_k \) 是第 \( k \) 個高斯成分的權重，滿足 \( \sum_{k=1}^{K} \pi_k = 1 \)。
- \( \mathcal{N}(x | \mu_k, \Sigma_k) \) 是第 \( k \) 個高斯分佈，其均值為 \( \mu_k \) ，協方差為 \( \Sigma_k \)。

#### **期望最大化算法（EM）**

期望最大化（Expectation-Maximization，EM）算法是一種迭代方法，通常用於含有潛在變量的概率模型中。GMM 就是一個典型的應用場景，其中潛在變量是數據點所屬的高斯成分。EM 算法的目標是最大化觀察數據的對數似然函數，並不斷迭代直到收斂。

EM 算法包含兩個步驟：

1. **E 步驟（期望步驟）：** 根據當前參數估計潛在變量的期望值。對於 GMM，這意味著計算每個數據點屬於每個高斯成分的概率（即責任度或後驗概率）。
   
   \[
   \gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}
   \]
   
   其中， \( \gamma_{ik} \) 是第 \( i \) 個數據點屬於第 \( k \) 個高斯成分的概率。

2. **M 步驟（最大化步驟）：** 根據 E 步驟中的期望計算結果，最大化參數的對數似然，從而更新模型參數（均值、協方差、權重）。
   
   \[
   \mu_k = \frac{\sum_{i=1}^{N} \gamma_{ik} x_i}{\sum_{i=1}^{N} \gamma_{ik}}
   \]
   \[
   \Sigma_k = \frac{\sum_{i=1}^{N} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i=1}^{N} \gamma_{ik}}
   \]
   \[
   \pi_k = \frac{1}{N} \sum_{i=1}^{N} \gamma_{ik}
   \]

這樣，EM 算法會在每次迭代中更新 GMM 的參數，直到達到收斂，通常是當參數變化小於某個閾值時。

### **scikit-learn 中的高斯混合模型（GMM）實現**

`scikit-learn` 提供了 `GaussianMixture` 類來實現高斯混合模型。這個類可以自動處理 EM 算法的執行，並支持多種分佈的高斯混合模型。

#### **範例程式碼：**

```python
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# 生成樣本數據
np.random.seed(0)
X = np.concatenate([
    np.random.normal(loc=-5, scale=1, size=(100, 2)),
    np.random.normal(loc=5, scale=1, size=(100, 2))
])

# 創建 GMM 模型並擬合數據
gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(X)

# 預測每個數據點的類別
labels = gmm.predict(X)

# 可視化結果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='.')
plt.title("GMM Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 查看 GMM 模型的參數
print("Means:\n", gmm.means_)
print("Covariances:\n", gmm.covariances_)
print("Weights:\n", gmm.weights_)
```

#### **代碼解析：**
- 我們首先生成了兩個不同的高斯分佈樣本。
- 然後使用 `GaussianMixture` 來擬合這些數據，並指定 `n_components=2`，即我們假設有兩個高斯成分。
- 接著，使用 `predict` 方法來預測每個數據點屬於哪個高斯成分。
- 最後，將結果可視化，顯示數據點在各自的簇中。

### **GMM 的優點與缺點**

#### **優點：**
- **靈活性：** GMM 能夠捕捉數據的多模態分佈，適合於複雜的數據集。
- **概率性：** GMM 能夠給出每個數據點屬於不同簇的概率，而不僅僅是確定的分類結果，這對不確定性較高的數據處理很有用。
- **生成模型：** GMM 是一個生成模型，它可以用來生成新數據點，這在一些生成任務中非常有用。

#### **缺點：**
- **高計算開銷：** 由於需要迭代運行 EM 算法來估計模型參數，因此計算開銷較大，尤其是當數據量非常大時。
- **對初始值敏感：** GMM 可能會陷入局部最優解，因此其結果依賴於初始參數的選擇。通常需要多次初始化來確保結果的穩定性。

### **總結**
高斯混合模型（GMM）是一個強大的生成模型，能夠用於處理具有多個潛在分佈的數據集。它的核心是期望最大化（EM）算法，這使得它在實際應用中能夠有效地估計混合模型的參數。`scikit-learn` 提供了 GMM 的高效實現，並可以與其他機器學習算法進行集成。