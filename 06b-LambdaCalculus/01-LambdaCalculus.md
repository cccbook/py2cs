# 第 1 章：λ-Calculus

如果你學過 Python ，在閱讀本書之前，請先閱讀並執行以下文章與程式，你將會看到一種奇特的 Python 程式。

* 文章：[用 Python 實作 Church 的 Lambda Calculus](LambdaCalculus.md)
* 程式：[lambdaCalculus.py](./_code/lambdaCalculus.py)

本書將帶領你理解上述這個奇妙的 Python 實作，並解說背後的運作機制，以及當初 Alonzo Church 設計 Lambda Calculus 的背景想法。

## λ-Calculus 的語法

λ-Calculus 的語法只有三個基本部分：

1. 變數 (Variable)：表示一個函數的參數或值。  
   範例：`x`、`y`。  

2. 函數抽象 (Abstraction)：用於定義一個函數，語法為 `λx.E`，表示一個變數 `x` 與表達式 `E`。  
   範例：`λx.x+1` 表示一個接收參數 `x` 並返回 `x+1` 的函數。

3. 函數應用 (Application)：將一個函數應用於一個參數，語法為 `(F X)`，表示將函數 `F` 應用到參數 `X` 上。  
   範例：`(λx.x+1 2)` 表示將數字 `2` 傳入函數 `λx.x+1`，結果為 `3`。

## 2.4 λ-Calculus 的核心概念

1. α-轉換 (Alpha Conversion)  
   改變函數參數的名稱，而不影響函數的邏輯。  
   範例：`λx.x` 可以改寫為 `λy.y`。

2. β-簡化 (Beta Reduction)  
   將函數應用於實際參數，並計算出結果。  
   範例：`(λx.x+1 2)` 簡化為 `2+1 = 3`。

3. η-轉換 (Eta Conversion)  
   簡化表達式，使其等價於一個更簡潔的形式。  
   範例：`λx.(F x)` 等價於 `F`，若 `F` 在 `x` 無其他依賴。

## Church 與 λ-Calculus

1930 年代 Church 在設計 Lambda Calculus 時，還沒有電腦，是以純粹數學（代數）的角度在思考這些問題。

Calculus 一詞，現在通常被認為是『微積分』，其實在數學領域，Calculus 其實是指代數系統。

而微積分則是一種研究『微分和積分』的代數系統。

所以 Lambda Calculus 其實是 Lambda 代數系統的意思。

由於 Church 在設計 Lambda Calculus 時，為了讓數學系統非常純粹，因此用函數代表一切，所以 

1. IF 是一個 Lambda 函數
2. TRUE, FALSE 也是 Lambda 函數 (Church Bool)
3. 0,1,2,3 等數值也都用 Lambda 函數定義 (Church Numeral)

## Church 的邏輯系統 (Church Boolean)
   
定義布林值：`TRUE = λx.λy.x`，`FALSE = λx.λy.y`。

定義 IF 條件：`IF = λc.λx.λy.c x y`。

定義邏輯運算 AND, OR, NOT, XOR ...

用 Python 實作如下

```py
IF    = lambda c:lambda x:lambda y:c(x)(y) #  if: lambda c x y. c x y # if c then x else y.
TRUE  = lambda x:lambda y:x # if true then x # 兩個參數執行第一個
FALSE = lambda x:lambda y:y # if false then y # 兩個參數執行第二個
AND   = lambda p:lambda q:p(q)(p) # if p then q else p
OR    = lambda p:lambda q:p(p)(q) # if p then p else q
XOR   = lambda p:lambda q:p(NOT(q))(q) #  if p then not q else q
NOT   = lambda c:c(FALSE)(TRUE) # if c then false else true
```

## Church 的數值系統（Church Numerals)

數字 0：`λf.λx.x`  

數字 1：`λf.λx.f x`  

數字 2：`λf.λx.f f x`  

後繼者 SUCCESSOR(x) = x+1 

    λ n:λ f:λ x:f(n(f)(x))

前一個 PREDECESSOR(x) = x-1:

    λ n:λ f:λ x:n(λ g : λ h : h(g(f)))(λ _ : x)(λ u : u)

加法運算 ADD：

    λm.λn.λf.λx.m f (n f x)`

用 Python 實作如下

```py
# Arithmetics
IDENTITY       = lambda x:x
SUCCESSOR      = lambda n:lambda f:lambda x:f(n(f)(x))
PREDECESSOR    = lambda n:lambda f:lambda x:n(lambda g : lambda h : h(g(f)))(lambda _ : x)(lambda u : u)
ADDITION       = lambda m:lambda n:n(SUCCESSOR)(m)
SUBTRACTION    = lambda m:lambda n:n(PREDECESSOR)(m)
MULTIPLICATION = lambda m:lambda n:lambda f:m(n(f))
POWER          = lambda x:lambda y:y(x)
ABS_DIFFERENCE = lambda x:lambda y:ADDITION(SUBTRACTION(x)(y))(SUBTRACTION(y)(x))

# Church Numerals
_zero  = lambda f:IDENTITY # 0      : 用 lambdaf. lambdax. x 當 0
_one   = SUCCESSOR(_zero)  # 1=S(0) : lambdaf. lambdaf. lambdax. x 當 1
_two   = SUCCESSOR(_one)   # 2=S(1) : lambdaf. lambdaf. lambdaf. lambdax. x 當 2
_three = SUCCESSOR(_two)   # 3=S(2)
_four  = MULTIPLICATION(_two)(_two)  # 4 = 2*2
_five  = SUCCESSOR(_four)            # 5 = S(4)
_eight = MULTIPLICATION(_two)(_four) # 8 = 2*4
_nine  = SUCCESSOR(_eight)           # 9 = S(8)
_ten   = MULTIPLICATION(_two)(_five) # 10 = 2*5
```

## 用 Y-Combinator 實現遞迴

遞迴是一種重複應用函數的技術，Y-Combinator 是 λ-Calculus 中的遞迴工具，形式為：

```
Y = λf.(λx.f (x x)) (λx.f (x x))
```

用 Python 實作如下

```py
Y = lambda f:\
  (lambda x:f(lambda y:x(x)(y)))\
  (lambda x:f(lambda y:x(x)(y)))

# 階層 FACTORIAL(n) = n!
FACTORIAL = Y(lambda f:lambda n:IF(IS_ZERO(n))\
  (lambda _:SUCCESSOR(n))\
  (lambda _:MULTIPLICATION(n)(f(PREDECESSOR(n))))\
(NIL))
```

## 將資料儲存在函數閉包 (Closure) 中

定義列表結構和操作方法，例如 `CONS`、`CAR` 和 `CDR`:

```python
CONS = lambda x: lambda y: lambda f: f(x)(y)
CAR  = lambda p: p(TRUE)
CDR  = lambda p: p(FALSE)
```

## Functional Programming 的常用函數

實現如範圍生成 (`RANGE`) 和映射 (`MAP`) 的函數：

```python
RANGE = lambda m: lambda n: Y(lambda f: lambda m: IF(IS_EQUAL(m)(n))
    (lambda _: CONS(m)(NIL))
    (lambda _: CONS(m)(f(SUCCESSOR(m))))
)(m)

MAP = lambda x:lambda g:Y(lambda f:lambda x:IF(IS_NULL(x))\
  (lambda _: x)\
  (lambda _: CONS(g(CAR(x)))(f(CDR(x))))\
(NIL))(x)
```

## 結語

如果你能輕易理解上述程式碼的運作原理，那麼應該已經是 Lambda Calculus 的專家，所以就不需要閱讀本書了。

但是如果你覺得上述程式碼非常詭異，非常難以理解，或者說，根本就是魔法 ...

那麼，你或許應該閱讀本書！

透過本書，你可以理解那個在電腦還沒發明的 1930 年代，到底 Church 是如何構想出『純粹函數式程式語言』Lambda Calculus 的。

理解了 Lambda Calculus ，也有助於理解『程式的本質』，特別是『函數式編程』，以及『計算理論』領域上一些重要的成果，像是『不可計算問題』。

在圖靈 (Alan Turing) 提出停止問題之前，其實 Alonzo Church 就已經證明了

    Lambda Calculus 當中兩個程式結果是否相同，是無法被判定的

該證明是 Church 發表在 American Journal of Mathematics 的一篇論文

* [Church, A. An unsolvable problem of elementary number theory.](https://www.ics.uci.edu/~lopes/teaching/inf212W12/readings/church.pdf)


