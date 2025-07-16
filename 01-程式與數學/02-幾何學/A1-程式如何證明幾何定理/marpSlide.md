---
marp: true
theme: default
class: lead
paginate: true
---

# 三角形中線交於一點的證明  
### 用 Gröbner Basis 證明重心存在

---

## 幾何命題：中線交於一點

- 任意三角形 ABC 的三條中線  
  （從頂點連到對邊中點的線段）  
  會交於一點，稱為**重心**。

---

## 幾何轉換為代數

設三點如下，方便計算：

\[
A = (0, 0), \quad B = (2a, 0), \quad C = (0, 2b)
\]

- 對邊中點：
  - D（BC中點）：\( (a, b) \)
  - E（AC中點）：\( (0, b) \)
  - F（AB中點）：\( (a, 0) \)

---

## 三條中線的直線方程

- AD：\( y \cdot a - x \cdot b = 0 \)
- BE：\( y(-2a) - (x - 2a)b = 0 \)
- CF：\( (y - 2b)a + 2bx = 0 \)

---

## 證明中線 AD 與 BE 交於一點

```python
from sympy import symbols, Eq, groebner, solve

x, y, a, b = symbols('x y a b')

poly1 = y * a - x * b
poly2 = (y)*(-2*a) - (x - 2*a)*b

G = groebner([poly1, poly2], x, y, order='lex')
sol = solve(G, x, y)
print(sol)  # {x: a/3, y: b/3}
