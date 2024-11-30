# 第 8 章：Combinator (組合子)

## Y-Combinator

```py
Y = lambda f:\
  (lambda x:f(lambda y:x(x)(y)))\
  (lambda x:f(lambda y:x(x)(y)))

# 階層 FACTORIAL(n) = n!
FACTORIAL = Y(lambda f:lambda n:\
  1 if n==0\
  else n*f(n-1)\
)

print(f'FACTORIAL(3)={FACTORIAL(3)}')
print(f'FACTORIAL(5)={FACTORIAL(5)}')

```

## ccc: 還有哪些 Combinator

在函數式程式設計中，有許多不同類型的組合子（combinator），它們提供了函數組合的能力，使得函數可以被更靈活和更高效地組合。以下是一些常見的組合子及其用途：

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

### 4. **Y-Combinator** (遞歸組合子)
Y-Combinator 是一個固定點組合子，可以用來實現遞歸函數。它是許多遞歸函數式程式語言的基礎。

**定義：**

\[
Y = \lambda f . (\lambda x . f (x x)) (\lambda x . f (x x))
\]

Y-Combinator 的作用是生成一個自我引用的函數，從而實現遞歸。

**Python 實現：**

```python
def Y(f):
    return (lambda x: f(lambda y: x(x)(y)))(lambda x: f(lambda y: x(x)(y)))

# 測試 Y
def factorial(f):
    return lambda x: 1 if x == 0 else x * f(x - 1)

fact = Y(factorial)
print(fact(5))  # 輸出 120
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

---

這些組合子提供了強大的功能，可以用來進行函數的組合、遞歸、運算符重組等操作。它們在理論計算機科學和函數式程式設計中具有深遠的影響，並且在許多高階語言的設計中得到了廣泛的應用。