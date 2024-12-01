# 第 8 章：Combinator (組合子)

## Y-Combinator

在 Church 的 LambdaCalculus 當中，使用了 Y-Combinator ，這是一種『遞迴不動點組合子』

以下是一個使用 Y-Combinator 計算階層 n! 的範例。

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

執行結果

```sh
$ python ycombinator.py 
FACTORIAL(3)=6
FACTORIAL(5)=120
```

但是以上版本的 Y-Combinator 比較不容易理解其遞迴原理，因此我們改用另一版本來分析

檔案: ycombinator2.py

```py
# js 版來源 -- https://bmsdave.github.io/blog/y-combinator-eng/

# Y(f) = f(Y(f)) = f(f(Y(f))) ..., 但是要 lazy ，所以最後加上 x
Y = lambda f: lambda x:f(Y(f))(x)

factorial = lambda f: lambda n: 1 if n == 0 else n * f(n - 1)

print(f'Y(factorial)(5)={Y(factorial)(5)}')
```

執行結果

```sh
$ python ycombinator2.py
Y(factorial)(5)=120
```

其原理就如同註解中所說的，

    Y(f) = f(Y(f)) = f(f(Y(f))) ...
    
但是要 lazy，否則會無窮遞迴當掉，所以最後加上 lambda x.

## Z-Combinator

並非只有 Y-Combinator 才能完成遞迴呼叫，以下的 Z-Combinator 也具有類似的效果

```py
def Z(f):
    return (lambda x: f(lambda y: x(x)(y)))(lambda x: f(lambda y: x(x)(y)))

# 測試 Z-Combinator 實現
# 定義一個簡單的遞歸函數，例如計算階乘
def factorial(f):
    return lambda x: 1 if x == 0 else x * f(x - 1)

# 計算 5 的階乘
print(f'factorial(5)={Z(factorial)(5)}')  # 輸出 120

```

## 更多組合子範例

在函數式程式設計中，有許多不同類型的組合子（combinator），它們提供了函數組合的能力，使得函數可以被更靈活和更高效地組合。以下是一些常見的組合子及其用途：

```py
I  = lambda x: x                   # Identity
K  = lambda x: lambda y:  x        # constant x, 丟棄 y
A  = lambda f: lambda x: f (x)     # apply f(x)
T  = lambda x: lambda f: f (x)     # thrush x(f) = f(x)
W  = lambda f: lambda x: f (x) (x) # duplication 重複
C  = lambda f: lambda y: lambda x: f (x) (y)  # flip 交換
B  = lambda f: lambda g: lambda x: f (g (x))  # Compose = f*g
S  = lambda f: lambda g: lambda x: f (x) (g(x)) # substitution
S_ = lambda f: lambda g: lambda x: f (g (x)) (x) # chain
S2 = lambda f: lambda g: lambda h: lambda x: f (g (x)) (h (x)) # converge
P  = lambda f: lambda g: lambda x: lambda y: f (g (x)) (g (y)) # psi
Y  = lambda f: (lambda g: g (g)) (lambda g: f (lambda x: g (g) (x))) # fix-point Y-Combinator 的另一種實現
```

這些組合子（combinators）在函數式程式設計中扮演著非常重要的角色，主要用來組合和操控函數。以下是每個組合子的解釋和 Python 示例：

### 1. **I-Combinator**
**作用**：`I` 是恒等函數，它會將輸入原封不動地返回。

**定義**：
```python
I = lambda x: x
```

**Python 範例**：
```python
print(I(5))  # 輸出 5
print(I("Hello"))  # 輸出 "Hello"
```

**解釋**：`I` 組合子可以作為恒等函數，將任何輸入值返回，不做任何改變。

---

### 2. **K-Combinator**
**作用**：`K` 是常數組合子，返回第一個參數，忽略第二個參數。

**定義**：
```python
K = lambda x: lambda y: x
```

**Python 範例**：
```python
print(K(10)(20))  # 輸出 10
print(K("Hello")("World"))  # 輸出 "Hello"
```

**解釋**：`K` 組合子用來返回第一個參數，而忽略第二個參數。

---

### 3. **A-Combinator**
**作用**：`A` 組合子可以將一個函數 `f` 應用到 `x` 上，然後返回結果。

**定義**：
```python
A = lambda f: lambda x: f(x)
```

**Python 範例**：
```python
def double(x):
    return x * 2

print(A(double)(5))  # 輸出 10
```

**解釋**：`A` 組合子實際上是將函數 `f` 應用到參數 `x` 上，等同於直接寫 `f(x)`。

---

### 4. **T-Combinator**
**作用**：`T` 組合子將 `x` 應用到函數 `f` 上。

**定義**：
```python
T = lambda x: lambda f: f(x)
```

**Python 範例**：
```python
def add_one(x):
    return x + 1

print(T(5)(add_one))  # 輸出 6
```

**解釋**：`T` 組合子將值 `x` 傳遞給函數 `f`，然後返回 `f(x)`。

---

### 5. **W-Combinator**
**作用**：`W` 組合子會將函數 `f` 應用於自己，實現自我應用。

**定義**：
```python
W = lambda f: lambda x: f(x)(x)
```

**Python 範例**：
```python
def self_apply(f):
    return f(f)

print(W(self_apply)(lambda x: x * 2)(5))  # 輸出 20
```

**解釋**：`W` 組合子實現了函數的自我應用，即將函數 `f` 應用於它自身的結果。

---

### 6. **C-Combinator**
**作用**：`C` 組合子用來交換兩個參數的順序。

**定義**：
```python
C = lambda f: lambda y: lambda x: f(x)(y)
```

**Python 範例**：
```python
def subtract(x, y):
    return x - y

print(C(subtract)(5)(3))  # 輸出 -2，等同於 subtract(3, 5)
```

**解釋**：`C` 組合子交換了傳遞給函數的兩個參數的順序，使得原來的 `(x, y)` 變成 `(y, x)`。

---

### 7. **B-Combinator**
**作用**：`B` 組合子實現了函數的組合（函數組合）。它會先將 `x` 作用於 `g`，然後將結果傳遞給 `f`。

**定義**：
```python
B = lambda f: lambda g: lambda x: f(g(x))
```

**Python 範例**：
```python
def add_5(x):
    return x + 5

def multiply_2(x):
    return x * 2

print(B(add_5)(multiply_2)(3))  # 輸出 11，因為 add_5(multiply_2(3)) = add_5(6) = 11
```

**解釋**：`B` 組合子實現了 `f(g(x))`，即將 `g(x)` 的結果作為參數傳遞給 `f`。

---

### 8. **S-Combinator**
**作用**：`S` 組合子用來同時將參數應用於兩個函數，並將它們的結果傳遞給第一個函數。

**定義**：
```python
S = lambda f: lambda g: lambda x: f(x)(g(x))
```

**Python 範例**：
```python
def add1(x):
    return x + 1

def multiply2(x):
    return x * 2

print(S(add1)(multiply2)(3))  # 輸出 7，因為 add1(multiply2(3)) = add1(6) = 7
```

**解釋**：`S` 組合子會同時將 `x` 應用到 `f` 和 `g` 上，並將 `g(x)` 的結果作為參數傳遞給 `f(x)`。

以下是對這些新的組合子（combinators）的詳細解釋和 Python 範例：

### 9. **S_-Combinator (Chain)**
**作用**：`S_` 是一個鏈接組合子，將兩個函數 `f` 和 `g` 應用到參數 `x` 上，然後將 `g(x)` 作為參數傳遞給 `f(g(x))`，並將結果應用於 `x`。

**定義**：
```python
S_ = lambda f: lambda g: lambda x: f(g(x))(x)
```

**Python 範例**：
```python
def add1(x):
    return x + 1

def multiply2(x):
    return x * 2

print(S_(add1)(multiply2)(3))  # 輸出 7，因為 add1(multiply2(3)) = add1(6) = 7
```

**解釋**：這個組合子首先將 `x` 傳遞給 `g`，得到 `g(x)`，然後再將 `g(x)` 作為參數傳遞給 `f(g(x))`，最後將結果應用於 `x`。

---

### 10. **S2-Combinator (Converge)**
**作用**：`S2` 是一個收斂組合子，將三個函數 `f`、`g` 和 `h` 應用到 `x` 上，並將 `g(x)` 和 `h(x)` 的結果作為參數傳遞給 `f(g(x))`。

**定義**：
```python
S2 = lambda f: lambda g: lambda h: lambda x: f(g(x))(h(x))
```

**Python 範例**：
```python
def add(x, y):
    return x + y

def multiply(x):
    return x * 2

def subtract(x):
    return x - 1

print(S2(add)(multiply)(subtract)(5))  # 輸出 13，因為 add(multiply(5), subtract(5)) = add(10, 4) = 13
```

**解釋**：這個組合子會將 `x` 應用到 `g` 和 `h` 上，然後將結果傳遞給 `f(g(x))(h(x))`。即先將 `x` 應用到 `g` 和 `h`，然後將它們的結果傳遞給 `f`。

---

### 11. **P-Combinator (Psi)**
**作用**：`P` 是一個 `psi` 組合子，它將兩個函數 `f` 和 `g` 應用到兩個參數 `x` 和 `y` 上，並且將 `g(x)` 和 `g(y)` 傳遞給 `f`。

**定義**：
```python
P = lambda f: lambda g: lambda x: lambda y: f(g(x))(g(y))
```

**Python 範例**：
```python
def add(x, y):
    return x + y

def multiply(x):
    return x * 2

print(P(add)(multiply)(3)(5))  # 輸出 16，因為 add(multiply(3), multiply(5)) = add(6, 10) = 16
```

**解釋**：這個組合子會將 `x` 和 `y` 分別傳遞給 `g(x)` 和 `g(y)`，然後將它們作為參數傳遞給 `f(g(x))(g(y))`。


### 小結

這些組合子在高階函數和遞歸操作中非常有用，能夠幫助實現函數的組合和自我應用，從而創建更靈活和高效的函數式程式設計結構。

- **I**：恒等函數，返回輸入的值。
- **K**：常數函數，返回第一個參數，忽略第二個。
- **A**：簡單的應用，直接返回 `f(x)`。
- **T**：將一個值應用到函數上，`T(x)(f)` 等於 `f(x)`。
- **W**：函數的自我應用，將函數應用於自己。
- **C**：交換兩個參數的順序。
- **B**：函數組合，先應用一個函數，再將結果應用到另一個函數。
- **S**：將參數同時應用到兩個函數，並將結果傳遞給第一個函數。
- **S_-Combinator (Chain)**：將函數 `g(x)` 應用到 `f(g(x))(x)`，實現鏈式應用。
- **S2-Combinator (Converge)**：將 `x` 應用到三個函數，並將結果應用於 `f(g(x))(h(x))`。
- **P-Combinator (Psi)**：將兩個參數 `x` 和 `y` 傳遞給 `g(x)` 和 `g(y)`，然後將其傳遞給 `f(g(x))(g(y))`。
- **Y-Combinator (Fix-Point)**：實現固定點（固定遞歸），生成一個自我引用的函數，便於實現遞歸操作。

這些組合子能幫助在函數式程式設計中進行靈活的函數組合和應用，對於構建更複雜的函數式程式結構有很大的幫助。
