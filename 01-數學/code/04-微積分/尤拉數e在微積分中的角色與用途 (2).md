# 尤拉數e在微積分中的角色與用途

## 簡介

尤拉數 e 是數學中，與圓周率幾乎同樣重要的一個數字，然而、尤拉數卻並沒有像圓周率這樣清楚的直覺意義，而且其用途與表現非常多樣化，這使得一般學生無法掌握到尤拉數的意義，本文將列出尤拉數 e 的幾種不同表示法，並說明其這些表示法之間的關係，以使讀者了解尤拉數 e 的數學價值。

## 尤拉數 e 的定義方式

尤拉數 e 是一個很難掌握的數字，但卻應用很廣，其表現出來的形式大致有下列三種：

```math
e = \lim_{n \rightarrow \infty} 1+\frac{1}{1!}+\frac{1}{2!}+...+\frac{1}{n!}
```

```math
e = \lim_{n \rightarrow \infty} (1+\frac{1}{n})^n
```

```math
\int_1^e \frac{1}{x} dx = 1
```

## 尤拉數 e 的特性

1 - 尤拉函數 $`e^x `$ 是微分運算的特徵值.

```math
\frac{d}{dx} e^x = e^x
```

證明：

根據以上定義 ([[eref eq2])，只要將 n 改寫為 $`\frac{1}{\Delta x}`$，就可以得到下列代換式

```math
\begin{eqnarray} 
e &=& \lim_{n \to \infty} (1+\frac{1}{n})^n &=& \lim_{\Delta x \to 0} (1+\Delta x)^{1/\Delta x}
\end{eqnarray} 
```

所以 
```math
e^{\Delta x} = 1+\Delta x ;
```

故

```math
\lim_{\Delta x \to 0} \frac{e^{\Delta x}-1}{\Delta x} = 1
```

因此

```math
\begin{eqnarray} 
\frac{d}{dx} e^x &=& \lim_{\Delta x \to 0} \frac{e^{(x+\Delta x)} - e^x}{\Delta x} \\
&=& \lim_{\Delta x \to 0} \frac{e^x(e^{\Delta x} - 1)}{\Delta x} \\
&=& e^x \lim_{\Delta x \to 0} \frac{e^{\Delta x} - 1}{\Delta x} \\
&=& e^x * 1 \\
&=& e^x
\end{eqnarray} 
```


2 - 尤拉函數 $`e^x `$ 的泰勒級數有無窮多項，但卻很簡單.

根據前一個特性，也就是公式 ([[eref eq4])，我們可以用泰勒展開始將 $`e^x `$ 進行微分，會發現其泰勒級數如下：

```math
e^x = 1+\frac{1}{1!} x + \frac{2}{2!} x^2 + ... \frac{n}{n!} x^n+ ...
```

3 - 尤拉複函數 $`e^{i x} `$ 可分解為三角函數

$`e^{i x} `$ 的泰勒級數，是傅立葉轉換的基礎，因為它可以被分解為 sin() 與 cos() 的組合。

```math
e^{i x} = cos(x) + i*sin(x)
```
