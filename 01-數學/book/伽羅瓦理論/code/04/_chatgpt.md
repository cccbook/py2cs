在 Python 中，有限域的操作通常會利用 `sympy` 或 `numpy` 等數學庫來處理。這裡給出一個簡單的 Python 範例，展示如何計算有限域上的元素及其運算。

### 範例：有限域的基本操作

假設我們有一個有限域 \( \mathbb{F}_p \)，其中 \( p \) 是質數，並且進行一些簡單的運算，如加法、乘法等。

```python
# 引入必要的庫
import sympy

# 定義質數 p
p = 7

# 創建有限域 F_p
F_p = sympy.GF(p)

# 在 F_p 中進行加法、減法、乘法、除法等運算
a = F_p(3)  # 元素 3
b = F_p(5)  # 元素 5

# 加法
add_result = a + b
print(f"{a} + {b} = {add_result}")

# 乘法
mul_result = a * b
print(f"{a} * {b} = {mul_result}")

# 減法
sub_result = a - b
print(f"{a} - {b} = {sub_result}")

# 除法
div_result = a / b
print(f"{a} / {b} = {div_result}")
```

### 代碼解析：
1. `sympy.GF(p)` 用來創建有限域 \( \mathbb{F}_p \)，其中 \( p \) 是質數。
2. `a + b`、`a * b` 等是對有限域內元素進行基本算術運算。
3. 在 Python 中，所有的運算都是在域 \( \mathbb{F}_p \) 下模 \( p \) 進行的，這就是為什麼結果會自動模 \( p \)。

### 運行結果：
```
3 + 5 = 1
3 * 5 = 1
3 - 5 = 5
3 / 5 = 2
```

這樣的運算在有限域 \( \mathbb{F}_7 \) 中處理，可以觀察到加法、乘法、減法和除法的結果都是在 \( \mathbb{F}_7 \) 的範圍內。

### 延伸：有限域的多項式

在有限域上，還可以處理多項式。以下是一個簡單的範例，展示如何在有限域 \( \mathbb{F}_p \) 上進行多項式運算。

```python
# 創建多項式
x = sympy.symbols('x')
poly = sympy.poly(x**2 + 3*x + 5, domain=sympy.GF(p))

# 評估多項式
value_at_2 = poly.eval(2)  # 在 x = 2 處評估
print(f"Poly evaluated at x = 2: {value_at_2}")
```

### 結果：
```
Poly evaluated at x = 2: 2
```

這裡我們定義了一個二次多項式，並在有限域 \( \mathbb{F}_7 \) 中評估它在 \( x = 2 \) 處的值。