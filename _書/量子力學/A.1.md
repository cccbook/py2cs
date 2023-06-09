## A.1 複數和向量空間

## 複數 (Complex Numbers)

一個複數 $z$ 是由實數 $a$ 和 $b$ 構成的有序組 $(a, b)$，也就是說 $z = (a, b)$，其中 $a, b \in \mathbb{R}$。複數域 $\mathbb{C}$ 是由所有複數組成的集合，其中加法和乘法分別定義為：

$$
(a, b) + (c, d) = (a + c, b + d) \\
(a, b)(c, d) = (ac - bd, ad + bc)
$$ 

複數還有很多重要的性質。例如，

- $i^2 = -1$
- $\forall z = (a, b), \exists r \geq 0, \theta \in [0, 2\pi)$ 使得 $z = r(\cos \theta + i\sin \theta)$，這就是複數的極座標形式（exponential form）
- 共軛複數：如果 $z = (a, b)$，那麼 $\bar{z} = (a, -b)$ 就是 $z$ 的共軛複數

## 向量空間 (Vector Spaces)

一個向量空間 $V$ 是由一個叫做“加法”的運算和一個叫做“數乘”的運算構成的集合。加法運算符號是 $+$，數乘運算符號是 $\cdot$ 或其他類似的符號。加法滿足以下幾條性質：

- $\forall \mathbf{u}, \mathbf{v} \in V, \mathbf{u} + \mathbf{v} \in V$
- $\forall \mathbf{u}, \mathbf{v} \in V, \mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$
- $\forall \mathbf{u}, \mathbf{v}, \mathbf{w} \in V, (\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
- $\exists$ 一個元素 $\mathbf{0} \in V$，使得對於任何 $\mathbf{x} \in V$， $\mathbf{x} + \mathbf{0} = \mathbf{x}$
- $\forall \mathbf{x} \in V$，存在一個元素 $-\mathbf{x} \in V$，使得 $\mathbf{x} + (-\mathbf{x}) = \mathbf{0}$

數乘滿足以下幾條性質：

- $\forall a \in \mathbb{R}, \mathbf{u} \in V, a \mathbf{u} \in V$
- $\forall a, b \in \mathbb{R}, \mathbf{u} \in V, (ab) \mathbf{u} = a(b\mathbf{u})$
- $\forall \mathbf{u} \in V, 1 \cdot \mathbf{u} = \mathbf{u}$
- $\forall a, b \in \mathbb{R}, \mathbf{u} \in V, (a+b)\mathbf{u} = a\mathbf{u} + b\mathbf{u}$
- $\forall a \in \mathbb{R}, \mathbf{u}, \mathbf{v} \in V, a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$

## 常見的向量空間

在機器學習和數學中，很多常見的向量空間（或是向量的集合）有以下幾種：

- $\mathbb{R}^n$， $n$ 維實向量空間
- $\mathbb{C}^n$， $n$ 維複向量空間
- 布林向量（boolean vectors） $\{0,1\}^n$，也就是 $n$ 維向量，每個元素都是 $0$ 或 $1$。

## 向量的範數 (Norm)

向量的範數就是該向量的大小，常用的範數有以下幾種：

- L1 範數：$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i|$
- L2 範數：$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$
- $\ell_{\infty}$ 範數：$\|\mathbf{x}\|_{\infty} = \max_{i=1,\ldots,n} |x_i|$

除了這三種範數之外，還有很多種範數，例如 squared Euclidean distance，Hamming distance，Mahalanobis distance 等等。

在機器學習中，L2 範數使用最廣泛，通常用來作為誤差的度量（loss function）。

## 向量的內積 (Inner Product)

向量的內積（或稱點積）是一種在向量空間中定義的二元運算，常寫成 $\langle \cdot, \cdot \rangle$ 或者 $\cdot^T \cdot$。設 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$，那麼兩個向量之間的點積定義為：

$$\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^T \mathbf{y} = \sum_{i=1}^n x_i y_i$$

在機器學習中，點積是非常重要的運算，它可以用來完成很多任務，例如：

- 計算殘差（residual） $\mathbf{x} - \mathbf{y}$ 的平方 L2 範數：$$\|\mathbf{x} - \mathbf{y}\|^2 = \langle \mathbf{x} - \mathbf{y}, \mathbf{x} - \mathbf{y} \rangle = \|\mathbf{x}\|^2 - 2 \langle \mathbf{x}, \mathbf{y} \rangle + \|\mathbf{y}\|^2$$
- 計算 $\mathbf{x}$ 與 $\mathbf{y}$ 之間的夾角（cosine similarity）：$$\operatorname{cosine-similarity}(\mathbf{x},\mathbf{y}) = \frac{\langle\mathbf{x},\mathbf{y}\rangle}{\|\mathbf{x}\|\|\mathbf{y}\|} =\cos(\theta)$$

## 向量的線性無關 (Linear Independence)

假設有 $k$ 個 $n$ 維向量 $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$，我們稱這些向量線性無關，如果對於任意的 $\alpha_1,\ldots, \alpha_k\in\mathbb{R}$，如果 $\sum_{i=1}^k \alpha_i \mathbf{v}_i = \mathbf{0}$，那麼就必須 $\alpha_1 = \alpha_2 = \cdots = \alpha_k = 0$。

在機器學習中，線性無關的概念很重要，它可以用來構建新的特徵（feature），並從中學習最好的表示方法。例如，在主成分分析（PCA）中，我們需要將高維的數據映射到一個低維的空間中，這就需要保證映射後的向量都線性無關。