## Softmax 的反傳遞數學公式

* [Softmax with cross-entropy](https://mattpetersen.github.io/softmax-with-cross-entropy) (超讚)
* [Backpropagation with softmax outputs and cross-entropy cost](https://sootlasten.github.io/2016/softmax-backprop-derivation/) (讚)
* [softmax 输出层数学原理推导](https://segmentfault.com/a/1190000024457412)
* [Backpropagation with Softmax / Cross Entropy](https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy)


ChatGPT:

Softmax 函數是一種常用於多分類任務的激活函數，在深度學習中被廣泛使用。它可以將一個長度為 $n$ 的向量 $\textbf{x}$ 轉換成一個長度仍然為 $n$ 的向量 $\textbf{y}$，其中每個元素 $y_i$ 表示 $\textbf{x}$ 屬於第 $i$ 類的概率。Softmax 函數的數學公式如下：

$$
y_i=\frac{\exp(x_i)}{\sum_{j=1}^n\exp(x_j)},\text{ for }i=1,2,\dots,n.
$$

其中，$x_i$ 是向量 $\textbf{x}$ 的第 $i$ 個元素，$y_i$ 則是向量 $\textbf{y}$ 的第 $i$ 個元素。這個公式包含兩個步驟：首先，對向量 $\textbf{x}$ 中的每個元素 $x_i$ 做指數運算，得到一個新的向量 $\textbf{e}=\big(\exp(x_1),\exp(x_2),\dots,\exp(x_n)\big)$；其次，將向量 $\textbf{e}$ 中的每個元素除以向量 $\textbf{e}$ 的元素總和，得到一個長度為 $n$ 的概率分佈向量 $\textbf{y}$。

為了進行反向傳播計算，需要計算 Softmax 函數對每個輸入 $x_i$ 的梯度，即 $\frac{\partial y_i}{\partial x_j}$，其中 $i,j=1,2,\dots,n$。我們可以首先對 $y_i$ 做求導：

$$
\begin{aligned}
\frac{\partial y_i}{\partial x_j}&=\frac{\partial}{\partial x_j}\frac{\exp(x_i)}{\sum_{k=1}^n\exp(x_k)} \\
&=\frac{\exp(x_i)\frac{\partial}{\partial x_j}\sum_{k=1}^n\exp(x_k)-\frac{\partial}{\partial x_j}\exp(x_i)\sum_{k=1}^n\exp(x_k)}{\left(\sum_{k=1}^n\exp(x_k)\right)^2}.
\end{aligned}
$$

當 $i\neq j$ 時，$\frac{\partial}{\partial x_j}\exp(x_i)=0$；當 $i=j$ 時，$\frac{\partial}{\partial x_j}\exp(x_i)=\exp(x_i)$。因此，上式可以簡化為：

$$
\frac{\partial y_i}{\partial x_j}=\begin{cases}\displaystyle
y_i(1-y_j),&\text{if }i=j,\\
-y_i y_j,&\text{if }i\neq j.
\end{cases}
$$

這個結果可以表示成矩陣形式，即 $\frac{\partial \textbf{y}}{\partial \textbf{x}}=\text{diag}(\textbf{y})-\textbf{y}\textbf{y}^\top$，其中 $\text{diag}(\textbf{y})$ 表示將 $\textbf{y}$ 中的元素放在主對角線上，其他位置填充為零的對角矩陣。

總結一下，Softmax 函數的反向傳播計算可以分為兩個步驟：首先，計算 Softmax 函數的輸出向量 $\textbf{y}$；其次，根據上面的公式計算 Softmax 函數對輸入向量 $\textbf{x}$ 的梯度。這個過程可以用矩陣形式表示，並且可以通過求解矩陣乘法和向量加法得到。