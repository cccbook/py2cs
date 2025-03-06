
## run

```
% python diffEx1.py
Eq(y(x), C1 + x**3/3 + x)
```

## ChatGPT

ccc: 請用一個 python 示範如何求解微分方程

ChatGpt

這裡提供一個簡單的 Python 示例來求解微分方程，使用 `SymPy` 庫，它是一個強大的符號計算工具。以下範例將解一個簡單的一階常微分方程（ODE）：

### 問題
求解微分方程：

\[
\frac{dy}{dx} = x^2 + 1
\]

### 步驟
1. 匯入 `sympy` 庫
2. 定義符號變數
3. 設定微分方程
4. 使用 `dsolve` 函數解方程

### 代碼示例

```python
import sympy as sp

# 定義變數
x = sp.symbols('x')
y = sp.Function('y')

# 定義微分方程 dy/dx = x^2 + 1
ode = sp.Eq(y(x).diff(x), x**2 + 1)

# 求解微分方程
solution = sp.dsolve(ode)

# 顯示結果
solution
```

### 結果
這段代碼將會輸出微分方程的通解，形式如下：

\[
y(x) = \frac{x^3}{3} + x + C
\]

其中 `C` 是積分常數。

這是微分方程求解的基本過程，`SymPy` 提供了強大的符號微積分和方程求解功能，適合處理多種數學問題。如果你有更複雜的微分方程，也可以使用這種方法。

