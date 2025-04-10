
### 1. **K-Combinator** (常數組合子)
K-Combinator 也叫做常數組合子，返回其第一個參數，並忽略第二個參數。

**定義：**

\[
K = \lambda x . \lambda y . x
\]

**Python 實現：**

```python
def K(x):
    return lambda y: x

# 測試 K
k = K(10)
print(k(20))  # 輸出 10
```

### 2. **I-Combinator** (恒等組合子)
I-Combinator 是最簡單的組合子，它將輸入返回不做任何修改。簡單來說，它就是恒等函數。

**定義：**

\[
I = \lambda x . x
\]

**Python 實現：**

```python
def I(x):
    return x

# 測試 I
print(I(5))  # 輸出 5
```

### 3. **S-Combinator** (組合子 S)
S-Combinator 是一個高階函數，可以將兩個函數應用到某個參數上。它的特點是：它接收三個參數，然後返回一個由這些參數組合而來的函數。

**定義：**

\[
S = \lambda f . \lambda g . \lambda x . f x (g x)
\]

這裡，`S` 組合子會將兩個函數 `f` 和 `g` 應用到相同的參數 `x`，然後將它們的結果傳遞給 `f`。

**Python 實現：**

```python
def S(f):
    return lambda g: lambda x: f(x)(g(x))

# 測試 S
def add1(x):
    return x + 1

def multiply2(x):
    return x * 2

s = S(add1)(multiply2)
print(s(5))  # 輸出 11，因為 (5 * 2) + 1 = 11
```

### 5. **W-Combinator** (增強組合子)
W-Combinator 是一個增強組合子，能夠接受一個函數並返回一個新的函數。它通常用來將一個函數變得更有表達力。

**定義：**

\[
W = \lambda f . \lambda x . f (f x)
\]

**Python 實現：**

```python
def W(f):
    return lambda x: f(f(x))

# 測試 W
def double(x):
    return x * 2

w = W(double)
print(w(5))  # 輸出 10，因為 double(double(5)) = 10
```

### 6. **B-Combinator** (組合子 B)
B-Combinator 是另一個重要的組合子，它可以用來將一個函數應用到多個參數上。它通常用於將參數重排列以適應所需的函數簽名。

**定義：**

\[
B = \lambda f . \lambda g . \lambda x . f (g x)
\]

**Python 實現：**

```python
def B(f):
    return lambda g: lambda x: f(g(x))

# 測試 B
def add5(x):
    return x + 5

def multiply3(x):
    return x * 3

b = B(add5)(multiply3)
print(b(4))  # 輸出 17，因為 (4 * 3) + 5 = 17
```

### 7. **C-Combinator** (交替組合子)
C-Combinator 是一種函數組合技術，允許將兩個函數應用於相同的參數並交替使用。

**定義：**

\[
C = \lambda f . \lambda g . \lambda x . g (f x)
\]

**Python 實現：**

```python
def C(f):
    return lambda g: lambda x: g(f(x))

# 測試 C
def subtract3(x):
    return x - 3

def add2(x):
    return x + 2

c = C(subtract3)(add2)
print(c(10))  # 輸出 9，因為 (10 + 2) - 3 = 9
```

### 8. **T-Combinator** (切換組合子)
T-Combinator 用來將輸入作為函數的輸入並返回結果。它特別適合用來測試函數的行為。

**定義：**

\[
T = \lambda x . \lambda f . f x
\]

**Python 實現：**

```python
def T(x):
    return lambda f: f(x)

# 測試 T
def square(x):
    return x * x

t = T(4)(square)
print(t)  # 輸出 16，因為 square(4) = 16
```

---

### 9. **P-Combinator** (投影組合子)
P-Combinator 可用於操作函數參數的組合，特別是當有多個輸入但只需要一部分時。

**定義：**

\[
P = \lambda x . \lambda y . x
\]

**Python 實現：**

```python
def P(x):
    return lambda y: x

# 測試 P
p = P(10)(20)
print(p)  # 輸出 10
```

---

### 10. **M-Combinator** (自映射組合子)
M-Combinator 將輸入應用到自身，常用於模擬遞歸結構的行為。

**定義：**

\[
M = \lambda f . f f
\]

**Python 實現：**

```python
def M(f):
    return f(f)

# 測試 M
def twice(f):
    return lambda x: f(f(x))

m = M(twice)(lambda x: x + 2)
print(m(3))  # 輸出 7，因為 (3 + 2) + 2 = 7
```

### 關於組合子的應用

#### 1. **Lambda 演算與組合子邏輯的關聯**
組合子是 Lambda 演算的具體應用形式，在構建計算模型、優化函數式程式設計、構造不變性函數等方面非常重要。可以將這些概念應用到：
- 無狀態函數式程式設計中，用於構造高效的純函數。
- 確保函數重組時的結構完整性。

#### 2. **Y-Combinator 的特別應用**
你提到的 Y-Combinator 是不動點的經典例子，可以用於：
- 實現不依賴顯式遞歸的遞歸函數。
- 構建無狀態的計算環境（例如函數式編程語言中的閉包）。

#### 3. **程式設計中的具體應用**
- S 和 B 組合子可以用於管道式資料處理。
- C 組合子用於實現函數的參數交換，特別是在柯里化函數中有實用價值。
- Y 組合子用於實現記憶化或尾遞歸優化。
