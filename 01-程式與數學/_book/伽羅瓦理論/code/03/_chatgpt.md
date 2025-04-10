在這一章中，我們將探討**分裂域**（splitting field）與**可分性**（separability）的概念，這是代數擴張和加羅瓦理論中的重要話題。我們會從代數方程的解的結構開始，了解什麼是分裂域以及如何處理多項式的可分性問題。這些概念對於理解加羅瓦理論的深層結構非常關鍵。

### 1. **分裂域的定義與構造**

**分裂域**是指一個代數擴張，其中某個多項式的所有根都位於該擴張中。給定一個多項式 \( f(x) \)，其分裂域是包含所有 \( f(x) \) 根的最小域擴張。

#### Python 程式：計算多項式的分裂域

我們以 \(x^2 - 2\) 為例，計算其分裂域。對於這個例子，分裂域是包含 \( \sqrt{2} \) 和 \( -\sqrt{2} \) 的最小域。

```python
import sympy as sp

# 定義符號變數
x = sp.symbols('x')

# 定義多項式 x^2 - 2
polynomial = x**2 - 2

# 解方程 x^2 - 2 = 0
roots = sp.solve(polynomial, x)

# 顯示解（即分裂域中的元素）
print("Roots of x^2 - 2 = 0:", roots)

# 構造分裂域
split_field = sp.FiniteField(roots[0])

print("Split field containing the roots:", split_field)
```

### 2. **可分性與不可分性**

在代數擴張中，一個多項式如果其所有的根都是簡單的，即沒有重根，則稱為**可分的**。反之，若多項式有重根，則稱為**不可分的**。

例如，對於多項式 \(f(x) = x^2 - 2\)，其根為 \( \sqrt{2} \) 和 \( -\sqrt{2} \)，因此它是可分的。然而，對於 \( f(x) = x^2 - 2x + 1 \) 這樣的多項式，其根是重根，這樣的多項式是不可分的。

#### Python 程式：檢查多項式的可分性

我們可以使用 `sympy` 來檢查多項式是否可分，即是否有重根。對於一個多項式，若其導數與原多項式有公因數，則該多項式是不可分的。

```python
# 定義多項式 x^2 - 2x + 1（不可分多項式）
polynomial_inseparable = x**2 - 2*x + 1

# 計算多項式的導數
derivative = sp.diff(polynomial_inseparable, x)

# 檢查多項式與導數的最大公因數（GCD）
gcd = sp.gcd(polynomial_inseparable, derivative)

# 判斷多項式是否可分
if gcd == 1:
    print("The polynomial is separable.")
else:
    print("The polynomial is inseparable.")
```

### 3. **分裂域與可分性**

**分裂域**與**可分性**的關係在於，對於一個可分多項式，它的分裂域是一個域擴張，其中包含了該多項式的所有簡單根（即無重根）。對於不可分多項式，其分裂域可能需要包含一些不可分的元素（如重根的連續擴張）。

例如，對於多項式 \( f(x) = x^3 - 2x \)，其根是 \(0\), \(\sqrt{2}\), 和 \(-\sqrt{2}\)，我們可以檢查它的分裂域和可分性。

#### Python 程式：計算分裂域並檢查可分性

```python
# 定義多項式 x^3 - 2x
polynomial_split = x**3 - 2*x

# 解多項式
roots_split = sp.solve(polynomial_split, x)

# 顯示根
print("Roots of x^3 - 2x = 0:", roots_split)

# 計算多項式的導數
derivative_split = sp.diff(polynomial_split, x)

# 檢查多項式是否可分
gcd_split = sp.gcd(polynomial_split, derivative_split)

if gcd_split == 1:
    print("The polynomial x^3 - 2x is separable.")
else:
    print("The polynomial x^3 - 2x is inseparable.")
```

### 4. **分裂域的結構**

分裂域的結構可以用加羅瓦群來描述，該群表示了多項式的根之間的對稱性。對於一個可分多項式，分裂域可以被視為一個由該多項式的根所生成的域擴張。

#### Python 程式：計算分裂域的加羅瓦群

```python
# 導入加羅瓦群計算庫
from sympy import GF

# 定義有限域 F_5（可以用於更簡單的例子）
F5 = sp.FiniteField(5)

# 定義多項式 x^2 - 2 在 F_5 上的分裂
polynomial_f5 = x**2 - 2

# 解多項式在 F_5 上的根
roots_f5 = sp.solve(polynomial_f5, x)

# 顯示根
print("Roots of x^2 - 2 in F_5:", roots_f5)

# 計算分裂域的加羅瓦群
# 這裡我們需要將根視為生成元，並觀察它們的對稱性
# 可以進一步使用群論工具來分析加羅瓦群
```

### 小結

這一章中，我們介紹了**分裂域**與**可分性**的基本概念，並使用 Python 程式演示了如何計算多項式的分裂域和檢查其可分性。我們還討論了分裂域的結構，並簡單介紹了如何用加羅瓦群描述這些結構。這些工具和方法在數學與代數擴張的研究中是不可或缺的，有助於我們更深入地理解加羅瓦理論的應用。