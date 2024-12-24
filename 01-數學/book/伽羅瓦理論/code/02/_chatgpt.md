在這一章中，我們將探索多項式與代數擴張的基本概念。具體來說，我們將探討多項式的根、代數擴張以及如何在 Python 中實現這些概念來驗證一些基本理論。

### 1. **多項式的定義與基本操作**

多項式通常表達為係數與變數的線性組合，例如：
\[
P(x) = a_n x^n + a_{n-1} x^{n-1} + \cdots + a_1 x + a_0
\]
其中，\( a_n, a_{n-1}, \dots, a_0 \) 是多項式的係數。

#### Python 程式：定義多項式並進行基本操作

```python
import numpy as np

# 定義多項式類
class Polynomial:
    def __init__(self, coefficients):
        # coefficients 是多項式的係數，從最高次項開始
        self.coefficients = np.array(coefficients)

    def evaluate(self, x):
        # 計算多項式的值
        return sum(c * x**i for i, c in enumerate(reversed(self.coefficients)))
    
    def degree(self):
        # 返回多項式的次數
        return len(self.coefficients) - 1
    
    def __add__(self, other):
        # 多項式加法
        degree = max(self.degree(), other.degree())
        new_coeffs = [0] * (degree + 1)
        
        for i in range(self.degree() + 1):
            new_coeffs[degree - i] += self.coefficients[-(i + 1)]
        
        for i in range(other.degree() + 1):
            new_coeffs[degree - i] += other.coefficients[-(i + 1)]
        
        return Polynomial(new_coeffs)

    def __mul__(self, other):
        # 多項式乘法
        degree = self.degree() + other.degree()
        new_coeffs = [0] * (degree + 1)
        
        for i, a in enumerate(reversed(self.coefficients)):
            for j, b in enumerate(reversed(other.coefficients)):
                new_coeffs[degree - (i + j)] += a * b
        
        return Polynomial(new_coeffs)

# 測試多項式的基本操作
P = Polynomial([1, -3, 2])  # P(x) = x^2 - 3x + 2
Q = Polynomial([2, 1])       # Q(x) = 2x + 1

print("P(x) =", P.coefficients)
print("Q(x) =", Q.coefficients)

# 計算 P(3) 和 Q(3)
print("P(3) =", P.evaluate(3))
print("Q(3) =", Q.evaluate(3))

# 多項式加法
R = P + Q
print("P(x) + Q(x) =", R.coefficients)

# 多項式乘法
S = P * Q
print("P(x) * Q(x) =", S.coefficients)
```

### 2. **代數擴張的基本概念**

代數擴張是指在一個域（或環）的基礎上，構造一個包含更多元素的域（或環）。對於多項式方程的解，代數擴張包括了根的引入，形成新的代數結構。

例如，考慮一個代數擴張 \(\mathbb{Q}(\alpha)\)，其中 \(\alpha\) 是某個多項式的根。

#### Python 程式：實現代數擴張

在這裡，我們將探索如何在 Python 中實現簡單的代數擴張，並解決代數方程。具體來說，我們將考慮解方程 \(x^2 - 2 = 0\)，即 \(\alpha = \sqrt{2}\) 的情況。

```python
import sympy as sp

# 定義符號變數
x = sp.symbols('x')

# 定義多項式
polynomial = x**2 - 2

# 解多項式方程 x^2 - 2 = 0
roots = sp.solve(polynomial, x)

# 顯示解
print("Roots of x^2 - 2 = 0:", roots)

# 計算代數擴張：在這個例子中我們加入根 sqrt(2)
alpha = roots[0]  # sqrt(2)

# 將代數擴張進行數值計算
print("Value of alpha (sqrt(2)):", alpha.evalf())
```

### 3. **有限域上的代數擴張**

對於有限域 \( \mathbb{F}_q \)，代數擴張的結構也是十分重要的。在這裡，我們會構造一個有限域上的代數擴張。

例如，考慮域 \(\mathbb{F}_5\) 上的代數擴張，其中多項式 \(x^2 + 1\) 在該域上是不可約的。

#### Python 程式：有限域上的代數擴張

```python
from sympy import FiniteField

# 定義有限域 F_5
F5 = FiniteField(5)

# 在 F_5 上構造代數擴張：x^2 + 1
x = F5.gen()  # 獲取 F_5 的生成元素
polynomial = x**2 + 1

# 解方程 x^2 + 1 = 0 在 F_5 上的根
roots_F5 = sp.solve(polynomial, x)

# 顯示根
print("Roots of x^2 + 1 = 0 in F_5:", roots_F5)
```

### 4. **代數擴張與最小多項式**

最小多項式是最小的代數方程，其有根的元素可以被視為代數擴張中的一個元素。對於一個代數數字，最小多項式是最小的多項式，它有這個數字作為根。

#### Python 程式：計算最小多項式

```python
from sympy import Poly

# 假設代數元素是根 sqrt(2)
alpha = sp.sqrt(2)

# 計算最小多項式
min_poly = sp.poly(alpha)

print("Minimal polynomial of sqrt(2):", min_poly)
```

### 小結

在這一部分中，我們用 Python 展示了多項式的基本操作，代數擴張的概念，並探討了如何解代數方程以及如何在有限域上構造代數擴張。這些程式不僅能幫助您理解多項式與代數擴張的基本理論，還能提供驗證這些概念的實際工具。