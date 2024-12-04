# Tex數學式寫法

## 簡易算式

符號 | 說明 | LaTex 寫法 
------|-------|----------------------
 $` \simeq `$ | approximately equal to (趨近) | \simeq 
 $` \equiv `$ | equivlent to (全等、定義) | \equiv 
 $` \propto `$ | proportional to (正比) | \propto 
 $` \infty `$ | infinity (無限大) | \infty |
 $` x \mapsto a `$ | x maps to a |  x \mapsto a 
 $` x \to a `$ | x approaches a |  x \to a 
 $` \lim_{x \to a} f(x) `$ | f(x) -- 當 x 趨近於 a 時， |  \lim_{x \to a} f(x) |
 $` \arg \max_x f(x) `$ | 最大化 f(x) | \arg \max_x f(x) 
 $` \arg \min_x f(x) `$ | 最小化 f(x) | \arg \min_x f(x) 
 $` \lceil x \rceil `$ | ceil 函數 (天花板) | \lceil x \rceil 
 $` \lfloor x \rfloor `$ | floor 函數 (地板) | \lfloor x \rfloor
 $` \hat{\theta} `$ | 最大似然估計 | \hat{\theta}


## 空白的寫法

no space | 34 | no space | 34
---------|----|----------|-------------
\, \thinspace | 3 4 |  \! \negthinspace | 34
\: \medspace | 3 4 |  \negmedspace | 34
\; \thickspace | 3 4 | \negthickspace | 34
\quad | 3 4|  | 
\qquad | 3 4| | 

## 常用符號

寫法    | 顯示             |寫法    | 顯示             |寫法    | 顯示             |寫法    | 顯示
--------|-----------------|----------|-----------------|--------|----------------|--------|------------------
\forall | $` \forall `$ |  \exists | $` \exists `$ | \infty | $` \infty `$ | \pm | $` \pm `$
\div | $` \div `$  | \times  | $` \times `$ |  \cap  | $` \cap `$  | \cup  | $` \cup `$  
\le   | $` \le `$   | \ge   | $` \ge `$   | \ll   | $` \ll `$   | \gg   | $` \gg `$   
\neq  | $` \neq `$  | \nleq | $` \nleq `$ | \ngeq | $` \ngeq `$ | \not< | $` \not< `$ 
\not> | $` \not> `$ | \not= | $` \not= `$ | \not\le | $` \not\le `$ | \not\ge | $` \not\ge `$ 
\equiv | $` \equiv `$ | \supset | $` \supset `$ | \subset | $` \subset `$ | \supseteq| $` \supseteq `$
\subseteq| $` \subseteq `$| \in  | $` \in `$  | \ni  | $` \ni `$  | \approx  | $` \approx `$ 
\vdash | $` \vdash `$ | \rightarrow | $` \rightarrow `$ | \leftarrow | $` \leftarrow `$| \Rightarrow | $` \Rightarrow `$ 
\Leftarrow  | $` \Leftarrow `$  | \Leftrightarrow | $` \Leftrightarrow `$ 

## 運算符號

寫法    | 顯示            
--------|----------------------
`\sum_{i=0}^n f(x)`	| $`\sum_{i=0}^n f(x)`$
`\frac{a}{x^2}`	| $`\frac{a}{x^2}`$
`(\frac{a}{x} )^2`	| $`(\frac{a}{x} )^2`$
`\left(\frac{a}{x} \right)^2`	| $` \left(\frac{a}{x} \right)^2`$
`\sum_{i=0}^n f(x)`	| $`\sum_{i=0}^n f(x)`$
`\sum_{i=0}^n f(x)`	| $`\sum_{i=0}^n f(x)`$

## 大型括號 (Bracketing Symbols)

寫法    | 顯示     | 寫法    | 顯示           | 寫法    | 顯示  
--------|----------|-------|-----------------|--------|-------
\{      | $` \{ `$ | \}    | $` \} `$        | \rangle| $` \rangle `$   
\backslash| $` \backslash `$ | \lfloor| $` \lfloor `$ |\rfloor| $` \rfloor `$
\lceil| $` \lceil `$ |\rceil| $` \rceil `$ |\langle| $` \langle `$


* () 大型括號使用：\left( 與 \right)，例如：\left(\frac{a}{x} \right)^2 會呈現 $` \left(\frac{a}{x} \right)^2 `$

* [] 大型括號使用：\left[ 與 \right]，例如：\left[\frac{a}{x} \right]^2 會呈現 $` \left[\frac{a}{x} \right]^2 `$

## 上標

寫法    | 顯示             |寫法    | 顯示             |寫法    | 顯示             |寫法    | 顯示
--------|-----------------|----------|-----------------|--------|----------------|--------|------------------
| \hat{x}| $` \hat{x} `$  | \dot{x}| $` \dot{x} `$ | \bar{x}| $` \bar{x} `$ | \vec{x}| $` \vec{x} `$

## 數學大寫粗體

參考：LaTeX 教材：黑板粗體字 -- http://libai.math.ncu.edu.tw/bcc16/7/latex/16.shtml

* 範例：\Bbb{ Y = X \beta + E } 會顯示成 $` \Bbb{ Y = X \beta + E } `$ 

* 範例：\mathbb{ Y = X \beta + E } 會顯示成 $` \mathbb{ Y = X \beta + E } `$

* 範例：\mathcal{ Y = X \beta + E } 會顯示成 $` \mathcal{ Y = X \beta + E } `$

## 單行範例 

```
f(n) = \sum^{N-1}_{k=0} F(k) e^{i 2 \pi k} \frac{n}{N}
```

```math
f(n) = \sum^{N-1}_{k=0} F(k) e^{i 2 \pi k} \frac{n}{N}
```

## 單行範例

```
e = lim_{n \rightarrow \infty}\; 1+\frac{1}{1!}+\frac{1}{2!}+...+\frac{1}{n!}
```

```math
e = lim_{n \rightarrow \infty}\; 1+\frac{1}{1!}+\frac{1}{2!}+...+\frac{1}{n!}
```

## 單行範例

```
e = lim_{n \rightarrow \infty} (1+\frac{1}{x})^x
```

```math
e = lim_{n \rightarrow \infty} (1+\frac{1}{x})^x
```

## 複雜範例

```
 \begin{split}
    \mathbf{T n} &= \left[T_{ij} \mathbf{e}_i \otimes \mathbf{e}_j \right] n_k \mathbf{e}_k \\
                        & = T_{ij} n_k \left(\mathbf{e}_i \otimes \mathbf{e}_j\right) \mathbf{e}_k \\
                        & = T_{ij} n_j \mathbf{e}_i
 \end{split}
```

```math
 \begin{split}
    \mathbf{T n} &= \left[T_{ij} \mathbf{e}_i \otimes \mathbf{e}_j \right] n_k \mathbf{e}_k \\
                        & = T_{ij} n_k \left(\mathbf{e}_i \otimes \mathbf{e}_j\right) \mathbf{e}_k \\
                        & = T_{ij} n_j \mathbf{e}_i
 \end{split}
```

## 多行陳述

```math
\begin{align*}
& (1)\;  S \in F； \\
& (2)\;  if\; A \in F,\;then\; \bar{A} \in F;  \\
& (3)\;  if\; A \in F, B \in F,\;then\; A \cup B \in F;  \\
\end{align*}
```

```
\begin{align*}
& (1)\;  S \in F； \\
& (2)\;  if\; A \in F,\;then\; \bar{A} \in F;  \\
& (3)\;  if\; A \in F, B \in F,\;then\; A \cup B \in F;  \\
\end{align*}
```


## 多行範例

```
\begin{eqnarray} 
f'(x) & = & \frac{d f(x)}{dx} = c_1+c_2*2*x+c_3*3*x^2+c_4*4*x^3+... \\
f''(x) & = & \frac{d f'(x)}{dx}  = c_2*2*1+c_3*3*2*x+c_4*4*3*x^2+... \\
f'''(x) & = & \frac{d f''(x)}{dx} = c_3*3*2*1+c_4*4*3*2*x+... \\
... \\
f^k(x) & = & \frac{d f^{k-1}(x)}{dx} = c_k k!+c_{k+1} (k+1)! x+...
\end{eqnarray}
```

```math
\begin{eqnarray} 
f'(x) & = & \frac{d f(x)}{dx} = c_1+c_2*2*x+c_3*3*x^2+c_4*4*x^3+... \\
f''(x) & = & \frac{d f'(x)}{dx}  = c_2*2*1+c_3*3*2*x+c_4*4*3*x^2+... \\
f'''(x) & = & \frac{d f''(x)}{dx} = c_3*3*2*1+c_4*4*3*2*x+... \\
... \\
f^k(x) & = & \frac{d f^{k-1}(x)}{dx} = c_k k!+c_{k+1} (k+1)! x+...
\end{eqnarray}
```


## 多行算式

多行等式 (eqnarray) 

```
\begin{eqnarray} 
y &=& x^4 + 4      \nonumber \\
&=& (x^2+2)^2 -4x^2 \nonumber \\
&\le&(x^2+2)^2    \nonumber
\end{eqnarray}

```

顯示

```math
\begin{eqnarray} 
y &=& x^4 + 4      \nonumber \\
&=& (x^2+2)^2 -4x^2 \nonumber \\
&\le&(x^2+2)^2    \nonumber
\end{eqnarray}
```


## 矩陣範例

```
\left[
  \begin{array}{ccc}
    T_{11} & T_{12} & T_{13} \\
    T_{21} & T_{22} & T_{23} \\
    T_{31} & T_{32} & T_{33} 
  \end{array}
\right]
```

```math
\left[
  \begin{array}{ccc}
    T_{11} & T_{12} & T_{13} \\
    T_{21} & T_{22} & T_{23} \\
    T_{31} & T_{32} & T_{33} 
  \end{array}
\right]
```

## 矩陣範例

```
 \left\{
 \begin{array}{c}
    t_1 \\ t_2 \\ t_3
 \end{array}
 \right\} =
 \left[ 
 \begin{array}{ccc}
    T_{11} & T_{12} & T_{13} \\
    T_{21} & T_{22} & T_{23} \\
    T_{31} & T_{32} & T_{33} 
 \end{array} 
 \right]
 \left\{
 \begin{array}{c}
    n_1 \\ n_2 \\ n_3
 \end{array}
 \right\}
```

```math
\left\{
 \begin{array}{c}
    t_1 \\ t_2 \\ t_3
 \end{array}
 \right\} =
 \left[ 
 \begin{array}{ccc}
    T_{11} & T_{12} & T_{13} \\
    T_{21} & T_{22} & T_{23} \\
    T_{31} & T_{32} & T_{33} 
 \end{array} 
 \right]
 \left\{
 \begin{array}{c}
    n_1 \\ n_2 \\ n_3
 \end{array}
 \right\}
```

## 範例 

```math
\int_1^x \frac{1}{x} dx = 1
```

## 範例 

```math
\frac{d}{dx} e^x = e^x
```

## 範例

```math
e^x = 1+\frac{1}{1!} x + \frac{2}{2!} x^2 + ... \frac{n}{n!} x^n+ ...
```

## 範例 

```math
e^{i x} = cos(x) + i*sin(x)
```

## 範例 

```math
f(x) = c_0 + c_1 x + c_2 x^2 + ...+ c_k x^k+...=\sum_{k=0}^\infty c_k x^k
```

## 範例 

```math
c_k = \frac{f^k(0)}{k!}
```

## 範例 

```math
f(x) = f(0) + \frac{f'(0)}{1!} x +...+ \frac{f^k (0)}{k!} x^k+...=\sum^{\infty}_{k=0} 

\frac{f^k(0)}{k!} x^k
```

## 範例

```math
f(x) = f(a) + \frac{f'(a)}{1!} x +...+ \frac{f^{k(a)}}{k!} x^k+...= \sum^\infty_{k=0} 

\frac{f^k(a)}{k!} x^k
```

## 範例 

```math
e^{i x} = 1 + i \frac{x}{1!} - \frac{x^2}{2!} - i \frac{x^3}{3!} + ... 
```

## 範例 

```math
cos(x) = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} + ... 
```

## 範例 

```math
sin(x) = \frac{x}{1!} - \frac{x^3}{3!} + \frac{x^5}{5!} + ...
```

## 範例 

```math
e^{i x} = cos(x) + i * sin(x) 
```

## 範例 

```math
f(x) = \frac{a_0}{2} + \sum^{\infty}_{n=- \infty} a_n cos(n x)+ b_n sin(n x)
```

## 範例 

```math
cos(n x) + i * sin(n x) = e^{i n x}
```

## 範例 

```math
f(x) = \sum^{\infty}_{n=-\infty} F_n e^{i n x}
```

## 範例 

```math
F_t = \frac{1}{2\pi} \int^{\pi}_{-\pi} f(x) e^{i t x} dx
```


## 範例 

x0    | xn             | x-n    
--------|-----------------|---------
$` F_0 = \frac{1}{2} `$ | $` F_n= \frac{1}{2} (a_n- i b_n) `$ | $` F_{- n}= \frac{1}{2} (a_n+i b_n) `$
$` a_0 = 2 c_0 `$ | $` a_n=F_n+F_{- n} `$ | $`  b_n=i (F_n-F_{-n}) `$


## 範例 

```math
f(t) = \int^\infty_{- \infty} F(x) e^{i 2 \pi x t} dt
```

## 範例

```
\begin{eqnarray}
f(x) = \left\{
 \begin{array}{c}
   1 \qquad x \in Z \\
   0 \qquad x \notin Z
 \end{array}
\right.
\end{eqnarray}
```

```math
\begin{eqnarray}
f(x) = \left\{
 \begin{array}{c}
   1 \qquad x \in Z \\
   0 \qquad x \notin Z
 \end{array}
\right.
\end{eqnarray}
```


## 參考文獻

* 更多的數學符號請參考 LaTeX:Symbols -- http://www.artofproblemsolving.com/Wiki/index.php/LaTeX:Symbols
* [ A brief tutorial on using math symbols and equations in wikidot](http://community.wikidot.com/howto:more-on-math-syntax)