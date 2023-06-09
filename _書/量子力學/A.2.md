## A.2 矩陣代數和線性變換

## 矩陣代數

在線性代數中，"矩陣"是由數字排成的矩形陣列。在數學和科學中，矩陣被用來表示線性變換的系統性方法。矩陣代數是專門研究矩陣及其性質的數學分支。

### 矩陣運算

定義假設 $\boldsymbol{A}$， $\boldsymbol{B}$ 是 $m \times n$ 矩陣， $\boldsymbol{C}$ 是 $n \times p$ 矩陣， $k$ 是標量，則矩陣的基本運算如下：

1. 矩陣加法：

$$
\boldsymbol{A} + \boldsymbol{B} = 
\begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \dots & a_{mn} \\
\end{bmatrix}
+
\begin{bmatrix}
b_{11} & b_{12} & \dots & b_{1n} \\
b_{21} & b_{22} & \dots & b_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
b_{m1} & b_{m2} & \dots & b_{mn} \\
\end{bmatrix}
=
\begin{bmatrix}
a_{11} + b_{11} & a_{12} + b_{12} & \dots & a_{1n} + b_{1n} \\
a_{21} + b_{21} & a_{22} + b_{22} & \dots & a_{2n} + b_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} + b_{m1} & a_{m2} + b_{m2} & \dots & a_{mn} + b_{mn} \\
\end{bmatrix}
$$

2. 矩陣乘法：

$$
\boldsymbol{A} \boldsymbol{C} =
\begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \dots & a_{mn} \\
\end{bmatrix}
\begin{bmatrix}
c_{11} & c_{12} & \dots & c_{1p} \\
c_{21} & c_{22} & \dots & c_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
c_{n1} & c_{n2} & \dots & c_{np} \\
\end{bmatrix}
=
\begin{bmatrix}
a_{11} c_{11} + a_{12} c_{21} + \dots + a_{1n} c_{n1} & \dots & a_{11} c_{1p} + a_{12} c_{2p} + \dots + a_{1n} c_{np} \\
a_{21} c_{11} + a_{22} c_{21} + \dots + a_{2n} c_{n1} & \dots & a_{21} c_{1p} + a_{22} c_{2p} + \dots + a_{2n} c_{np} \\
\vdots & \ddots & \vdots \\
a_{m1} c_{11} + a_{m2} c_{21} + \dots + a_{mn} c_{n1} & \dots & a_{m1} c_{1p} + a_{m2} c_{2p} + \dots + a_{mn} c_{np} \\
\end{bmatrix}
$$

3. 矩陣標量乘法：

$$
k \boldsymbol{A} =
k \begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \dots & a_{mn} \\
\end{bmatrix}
=
\begin{bmatrix}
k a_{11} & k a_{12} & \dots & k a_{1n} \\
k a_{21} & k a_{22} & \dots & k a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
k a_{m1} & k a_{m2} & \dots & k a_{mn} \\
\end{bmatrix}
$$

矩陣乘法並不滿足交換律，即 $\boldsymbol{A} \boldsymbol{C} \neq \boldsymbol{C} \boldsymbol{A}$，也不一定滿足結合律，即 $\boldsymbol{A}(\boldsymbol{B} \boldsymbol{C}) \neq (\boldsymbol{A} \boldsymbol{B}) \boldsymbol{C}$。

### 矩陣特殊類型

1. 單位矩陣：元素在對角線上為 1，其它元素均為 0 的矩陣，用 $\boldsymbol{I}$ 表示。

$$
\boldsymbol{I}_3 =
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
\end{bmatrix}
$$

2. 零矩陣：所有元素均為 0 的矩陣，用 $\boldsymbol{O}$ 表示。

$$
\boldsymbol{O}_{2 \times 3} =
\begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
\end{bmatrix}
$$

3. 對稱矩陣：矩陣 $\boldsymbol{A}$ 的對稱矩陣定義為 $\boldsymbol{A} + \boldsymbol{A}^T$。

$$
\boldsymbol{A} =
\begin{bmatrix}
1 & 2 \\
2 & 3 \\
\end{bmatrix}
\Rightarrow
\boldsymbol{A}^T =
\begin{bmatrix}
1 & 2 \\
2 & 3 \\
\end{bmatrix}^T
=
\begin{bmatrix}
1 & 2 \\
2 & 3 \\
\end{bmatrix}
\Rightarrow
\boldsymbol{A} +
\boldsymbol{A}^T =
\begin{bmatrix}
2 & 4 \\
4 & 6 \\
\end{bmatrix}
$$

4. 正交矩陣：矩陣 $\boldsymbol{A}$ 的正交矩陣定義為 $\boldsymbol{A}^T \boldsymbol{A} = \boldsymbol{A} \boldsymbol{A}^T = \boldsymbol{I}$，即 $\boldsymbol{A}$ 的轉置等於其逆矩陣。

$$
\boldsymbol{Q} =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta \\
\end{bmatrix}
\Rightarrow
\boldsymbol{Q}^{-1} =
\begin{bmatrix}
\cos\theta & \sin\theta \\
-\sin\theta & \cos\theta \\
\end{bmatrix}
\Rightarrow
\boldsymbol{Q}^T =
\begin{bmatrix}
\cos\theta & \sin\theta \\
-\sin\theta & \cos\theta \\
\end{bmatrix}
$$

其中， $\boldsymbol{Q}$ 是一個 $2 \times 2$ 的旋轉矩陣，將二維點逆時針旋轉 $\theta$ 度。

5. 對角矩陣：其它元素均為 0，對角線元素可以不相等的矩陣。

$$
\boldsymbol{D} =
\begin{bmatrix}
\lambda_1 & 0 & 0 & \dots & 0 \\
0 & \lambda_2 & 0 & \dots & 0 \\
0 & 0 & \lambda_3 & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \dots & \lambda_n \\
\end{bmatrix}
$$

### 矩陣求逆

如果一個矩陣存在逆矩陣，則該矩陣為可逆矩陣或非奇異矩陣。對於一個 $n \times n$ 的可逆矩陣 $\boldsymbol{A}$，其逆矩陣 $\boldsymbol{A}^{-1}$ 滿足以下條件：

$$
\begin{aligned}
\boldsymbol{A}^{-1} \boldsymbol{A} &= \boldsymbol{I} \\
\boldsymbol{A} \boldsymbol{A}^{-1} &= \boldsymbol{I}
\end{aligned}
$$

求逆矩陣的方法有許多種，其中一種是高斯-喬丹消元法。以 $3 \times 3$ 矩陣為例，我們需要求解下列矩陣的逆矩陣：

$$
\boldsymbol{A} = 
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33} \\
\end{bmatrix}
$$

我們可以在 $\boldsymbol{A}$ 的右側附加一個 $3 \times 3$ 的單位矩陣 $\boldsymbol{I}_3$，得到擴展矩陣 $\boldsymbol{B} = [\boldsymbol{A} | \boldsymbol{I}_3]$：

$$
\boldsymbol{B} =
\begin{bmatrix}
a_{11} & a_{12} & a_{13} & 1 & 0 & 0 \\
a_{21} & a_{22} & a_{23} & 0 & 1 & 0 \\
a_{31} & a_{32} & a_{33} & 0 & 0 & 1 \\
\end{bmatrix}
$$

接下來，我們使用高斯-喬丹消元法，通過矩陣消元將 $\boldsymbol{B}$ 化簡為 $\boldsymbol{I}_{3}$，此時 $\boldsymbol{B}$ 的左側 $3 \times 3$ 矩陣就是 $\boldsymbol{A}^{-1}$。更新的步驟如下：

1. 將 $\boldsymbol{B}$ 第一行的第一個非零元素變為 1，即固定矩陣 $\boldsymbol{A}$ 的上左角位置。

$$
\boldsymbol{B}^{(1)} =
\begin{bmatrix}
1 & \frac{a_{12}}{a_{11}} & \frac{a_{13}}{a_{11}} & \frac{1}{a_{11}} & 0 & 0 \\
a_{21} & a_{22} & a_{23} & 0 & 1 & 0 \\
a_{31} & a_{32} & a_{33} & 0 & 0 & 1 \\
\end{bmatrix}
$$

2. 將 $\boldsymbol{B}^{(1)}$ 第二行的第一個元素變為 0，即使用第一行的倍數消去第二行的首元。

$$
\boldsymbol{B}^{(2)} =
\begin{bmatrix}
1 & \frac{a_{12}}{a_{11}} & \frac{a_{13}}{a_{11}} & \frac{1}{a_{11}} & 0 & 0 \\
0 & a_{22}-\frac{a_{12}a_{21}}{a_{11}} & a_{23}-\frac{a_{12}a_{23}}{a_{11}} & -\frac{a_{21}}{a_{11}} & 1 & 0 \\
a_{31} & a_{32} & a_{33} & 0 & 0 & 1 \\
\end{bmatrix}
$$

3. 重複步驟 1 和 2，將矩陣消成階梯形式，此時，左側 $3 \times 3$ 的矩陣就是 $\boldsymbol{A}^{-1}$。

### 特徵值與特徵向量

如果一個 $n \times n$ 矩陣 $\boldsymbol{A}$ 可以寫成 $\boldsymbol{A} = \boldsymbol{Q} \boldsymbol{\Lambda} \boldsymbol{Q}^{-1}$ 的形式，其中 $\boldsymbol{\Lambda}$ 是一個對角矩陣， $\boldsymbol{Q}$ 是一個正交矩陣，那麼 $\boldsymbol{\Lambda}$ 的對角線上的元素就是矩陣 $\boldsymbol{A}$ 的特徵值，對應的 $\boldsymbol{Q}$ 的列向量就是特徵向量。

特徵向量的形式：$$
\boldsymbol{A} \boldsymbol{v} = \lambda \boldsymbol{v}
$$

其中， $\boldsymbol{v}$ 為特徵向量， $\lambda$ 為特徵值。

特徵向量 $\boldsymbol{v}$ 滿足以下性質：

1. 特徵向量與它對應的特徵值是成一對的。
2. 矩陣的每個特徵向量都不相等，但是矩陣可以有重複的特徵值。
3. 如果特徵向量 $\boldsymbol{v}$ 關於 $\boldsymbol{A}$ 的某一小空間或子空間是不變的，則 $\boldsymbol{v}$ 為該空間的特徵向量。

### 奇異值分解

對於一個矩陣 $\boldsymbol{A}$，它的奇異值分解表示為 $\boldsymbol{A} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^T$，其中 $\boldsymbol{U}$ 和 $\boldsymbol{V}$ 分別是 $\boldsymbol{AA}^T$ 和 $\boldsymbol{A}^T \boldsymbol{A}$ 的正交矩陣， $\boldsymbol{\Sigma}$ 是對角矩陣，對角線上元素為矩陣的奇異值：

$$
\boldsymbol{\Sigma} =
\begin{bmatrix}
\sigma_1 & 0 & \dots & 0 \\
0 & \sigma_2 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & \sigma_r \\
0 & 0 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & 0 \\
\end{bmatrix}
$$

其中 $r$ 為矩陣 $\boldsymbol{A}$ 的秩。

根據奇異值分解定理，任意矩陣都可以分解為三個矩陣的乘積。矩陣的奇異值包含了其重要程度，可以用於數據壓縮、矩陣近似等問題。如果矩陣的奇異值分解中只保留了前 $k$ 個奇