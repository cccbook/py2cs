### **第六章：遞迴與 Y-Combinator**

在 Lambda Calculus 中，遞迴是一種強大的工具，用於實現像階乘、費波那契數列等數學函數。然而，由於 Lambda Calculus 本質上是無狀態的，它無法直接定義自我參照的函數，因此需要借助固定點組合子（Fixed Point Combinator），如著名的 **Y-Combinator**，來實現遞迴。

---

#### **6.1 遞迴的挑戰**

在傳統的編程語言中，我們可以直接通過函數名稱來調用自身，比如：

```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

print(f'factorial(5)={factorial(5)}')
```

但在純 Lambda Calculus 中，函數是匿名的，無法直接通過名稱進行自我調用。因此，我們需要一個結構來幫助函數「引用自己」。

---

#### **6.2 固定點組合子（Fixed Point Combinator）**

**固定點組合子**的核心思想是：給定一個函數 \( F \)，我們可以構造一個值 \( Y(F) \)，使得 \( Y(F) \) 恰好滿足以下條件：
\[
Y(F) = F(Y(F))
\]
換句話說，\( Y(F) \) 是 \( F \) 的一個固定點（Fixed Point）。

Y-Combinator 的定義為：
\[
Y = \lambda f. (\lambda x. f (x x)) (\lambda x. f (x x))
\]

這個結構巧妙地實現了函數的自我引用。

**Python 模擬：**
```python
Y_COMBINATOR = lambda f: (lambda x: f(lambda v: x(x)(v)))(lambda x: f(lambda v: x(x)(v)))
```

---

#### **6.3 階乘的遞迴實現**

使用 Y-Combinator，我們可以定義階乘函數。首先，我們需要定義一個「偽階乘函數」 \( F \)，它接收一個函數 \( g \)（用於遞迴調用），並在其基礎上構建階乘邏輯。

\[
F(g) = \lambda n. (\text{IfThenElse} (\text{IsZero}(n)) (1) (n \cdot g(n-1)))
\]

**Python 實現：**
```python
FACTORIAL = Y_COMBINATOR(lambda g: lambda n: 
    IF(IS_ZERO(n))(_one)(MULTIPLICATION(n)(g(PRED(n))))
)
```

**測試：**
```python
assert FACTORIAL(_three)(lambda x: x + 1)(0) == 6  # 3! = 6
assert FACTORIAL(_five)(lambda x: x + 1)(0) == 120  # 5! = 120
```

---

#### **6.4 費波那契數列**

費波那契數列的遞迴定義為：
\[
F(n) = 
\begin{cases} 
0 & \text{if } n = 0 \\
1 & \text{if } n = 1 \\
F(n-1) + F(n-2) & \text{otherwise}
\end{cases}
\]

使用 Y-Combinator 實現費波那契數列：

**Python 實現：**
```python
FIBONACCI = Y_COMBINATOR(lambda g: lambda n: 
    IF(IS_ZERO(n))(_zero)(
        IF(IS_EQUAL(n)(_one))(_one)(
            ADDITION(g(PRED(n)))(g(PRED(PRED(n))))
        )
    )
)
```

**測試：**
```python
assert FIBONACCI(_zero)(lambda x: x + 1)(0) == 0  # Fib(0) = 0
assert FIBONACCI(_one)(lambda x: x + 1)(0) == 1  # Fib(1) = 1
assert FIBONACCI(_five)(lambda x: x + 1)(0) == 5  # Fib(5) = 5
```

---

#### **6.5 Y-Combinator 的直覺解釋**

Y-Combinator 之所以能實現遞迴，是因為它利用了高階函數與延遲計算的特性。具體來說：
1. **自我引用：** \( \lambda x. f(x x) \) 是一個能將自身作為參數傳遞的結構。
2. **固定點求解：** 它通過不斷套用自身來找到函數的固定點，從而實現遞迴。

我們可以將 Y-Combinator 理解為遞迴的一種「展開」形式，每次執行遞迴時，它會動態地構造下一層函數調用。

---

#### **6.6 本章小結**

本章通過引入 **Y-Combinator**，展示了如何在 Lambda Calculus 中實現遞迴。我們以階乘和費波那契數列為例，說明如何使用固定點組合子解決無名稱函數的自我調用問題。

在下一章中，我們將進一步探討 Lambda Calculus 與現代編程的聯繫，並應用這些基礎構建簡單的 Lambda 表達式解釋器。
