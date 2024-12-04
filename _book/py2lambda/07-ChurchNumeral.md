### 第 7 章：Church Numeral (丘奇的數值系統)

本章介紹了 **Church Numerals**，即使用 λ 演算表達自然數的抽象方式，並基於此構建基本的數學運算（如加法、乘法、冪運算等）和比較運算（如小於、大於、等於等）。

---

#### **1. 定義基礎運算與數字**

在 Church Numerals 中，自然數定義為接受一個函數並將其重複應用的函數。以下是數字和基礎運算的定義：

- **_zero**: 表示 0，定義為 `λf.λx.x`，即不對輸入進行任何應用。
- **SUCCESSOR**: 表示數字加 1，將函數的應用次數增加一。
- **PREDECESSOR**: 表示數字減 1。
- **ADDITION**: 定義為重複應用繼承函數。
- **MULTIPLICATION**: 表示函數應用次數的重複（嵌套）結構。
- **POWER**: 表示次方運算。

##### 代碼示例：

```py
IDENTITY       = lambda x: x
SUCCESSOR      = lambda n: lambda f: lambda x: f(n(f)(x))
PREDECESSOR    = lambda n: lambda f: lambda x: n(lambda g: lambda h: h(g(f)))(lambda _: x)(lambda u: u)
ADDITION       = lambda m: lambda n: n(SUCCESSOR)(m)
SUBTRACTION    = lambda m: lambda n: n(PREDECESSOR)(m)
MULTIPLICATION = lambda m: lambda n: lambda f: m(n(f))
POWER          = lambda x: lambda y: y(x)
```

#### **2. 定義範例數字**

使用 **SUCCESSOR** 等基礎運算構造範例數字：

```py
# Church Numerals
_zero  = lambda f:IDENTITY # 0      : 用 λf. λx. x 當 0
_one   = SUCCESSOR(_zero)  # 1=S(0) : λf. λf. λx. x 當 1
_two   = SUCCESSOR(_one)   # 2=S(1) : λf. λf. λf. λx. x 當 2
_three = SUCCESSOR(_two)   # 3=S(2)
_four  = MULTIPLICATION(_two)(_two)  # 4 = 2*2
_five  = SUCCESSOR(_four)            # 5 = S(4)
_eight = MULTIPLICATION(_two)(_four) # 8 = 2*4
_nine  = SUCCESSOR(_eight)           # 9 = S(8)
_ten   = MULTIPLICATION(_two)(_five) # 10 = 2*5
```

#### **3. 定義比較運算**

- **IS_ZERO**: 判斷是否為 0。
- **IS_LESS_THAN**: 判斷 m 是否小於 n。
- **IS_EQUAL**: 判斷兩數是否相等。
- **IS_GREATER_THAN**: 判斷 m 是否大於 n。

##### 代碼示例：

```py
IS_ZERO               = lambda n: n(lambda _: FALSE)(TRUE)
IS_LESS_THAN          = lambda m: lambda n: NOT(IS_LESS_THAN_EQUAL(n)(m))
IS_LESS_THAN_EQUAL    = lambda m: lambda n: IS_ZERO(SUBTRACTION(m)(n))
IS_EQUAL              = lambda m: lambda n: AND(IS_LESS_THAN_EQUAL(m)(n))(IS_LESS_THAN_EQUAL(n)(m))
IS_GREATER_THAN       = lambda m: lambda n: NOT(IS_LESS_THAN_EQUAL(m)(n))
```

## 完整程式碼

檔案： churchNumerial.py

```py
# Church Booleans : Logic
IF    = lambda c:lambda x:lambda y:c(x)(y) #  if: λ c x y. c x y # if c then x else y.
TRUE  = lambda x:lambda y:x # if true then x # 兩個參數執行第一個
FALSE = lambda x:lambda y:y # if false then y # 兩個參數執行第二個
AND   = lambda p:lambda q:p(q)(p) # if p then q else p
OR    = lambda p:lambda q:p(p)(q) # if p then p else q
XOR   = lambda p:lambda q:p(NOT(q))(q) #  if p then not q else q
NOT   = lambda c:c(FALSE)(TRUE) # if c then false else true

ASSERT = lambda truth: (IF(truth)
    (lambda description:f'[✓] ${description}')
    (lambda description:f'[✗] ${description}')
)

REFUTE = lambda truth:ASSERT(NOT(truth))

TEST   = lambda description:lambda assertion:\
    print(assertion(description))

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
_zero  = lambda f:IDENTITY # 0      : 用 λf. λx. x 當 0
_one   = SUCCESSOR(_zero)  # 1=S(0) : λf. λf. λx. x 當 1
_two   = SUCCESSOR(_one)   # 2=S(1) : λf. λf. λf. λx. x 當 2
_three = SUCCESSOR(_two)   # 3=S(2)
_four  = MULTIPLICATION(_two)(_two)  # 4 = 2*2
_five  = SUCCESSOR(_four)            # 5 = S(4)
_eight = MULTIPLICATION(_two)(_four) # 8 = 2*4
_nine  = SUCCESSOR(_eight)           # 9 = S(8)
_ten   = MULTIPLICATION(_two)(_five) # 10 = 2*5

# Comparison
IS_ZERO               = lambda n:n(lambda _:FALSE)(TRUE)
IS_LESS_THAN          = lambda m:lambda n:NOT(IS_LESS_THAN_EQUAL(n)(m))
IS_LESS_THAN_EQUAL    = lambda m:lambda n:IS_ZERO(SUBTRACTION(m)(n))
IS_EQUAL              = lambda m:lambda n:AND(IS_LESS_THAN_EQUAL(m)(n))(IS_LESS_THAN_EQUAL(n)(m))
IS_NOT_EQUAL          = lambda m:lambda n:OR(NOT(IS_LESS_THAN_EQUAL(m)(n)))(NOT(IS_LESS_THAN_EQUAL(n)(m)))
IS_GREATER_THAN_EQUAL = lambda m:lambda n:IS_LESS_THAN_EQUAL(n)(m)
IS_GREATER_THAN       = lambda m:lambda n:NOT(IS_LESS_THAN_EQUAL(m)(n))
IS_NULL               = lambda p:p(lambda x:lambda y:FALSE)
NIL                   = lambda x:TRUE


TEST('IDENTITY')\
  (ASSERT(IS_EQUAL(IDENTITY)(lambda x:x)))

TEST('SUCCESSOR')\
  (ASSERT(IS_EQUAL(SUCCESSOR(_zero))(_one)))

TEST('PREDECESSOR')\
  (ASSERT(IS_EQUAL(_zero)(PREDECESSOR(_one))))

TEST('ADDITION')\
  (ASSERT(IS_EQUAL(SUCCESSOR(_one))(ADDITION(_one)(_one))))

TEST('SUBTRACTION')\
  (ASSERT(IS_EQUAL(_zero)(SUBTRACTION(_one)(_one))))

TEST('MULTIPLICATION')\
  (ASSERT(IS_EQUAL(_four)(MULTIPLICATION(_two)(_two))))

TEST('POWER')(ASSERT(AND\
  (IS_EQUAL(_nine)(POWER(_three)(_two)))\
  (IS_EQUAL(_eight)(POWER(_two)(_three)))))

TEST('ABS_DIFFERENCE')(ASSERT(AND\
  (IS_EQUAL(_one)(ABS_DIFFERENCE(_three)(_two)))\
  (IS_EQUAL(_one)(ABS_DIFFERENCE(_two)(_three)))))

TEST('IS_ZERO')\
  (ASSERT(IS_ZERO(_zero)))

TEST('IS_LESS_THAN')\
  (ASSERT(IS_LESS_THAN(_zero)(_one)))

TEST('IS_LESS_THAN_EQUAL')(ASSERT(AND\
  (IS_LESS_THAN_EQUAL(_one)(_one))\
  (IS_LESS_THAN_EQUAL(_zero)(_one))))

TEST('IS_EQUAL')(ASSERT(AND\
  (IS_EQUAL(_zero)(_zero))\
  (IS_EQUAL(_one)(_one))))

TEST('IS_NOT_EQUAL')\
  (ASSERT(IS_NOT_EQUAL(_zero)(_one)))

TEST('IS_GREATER_THAN_EQUAL')(ASSERT(AND\
  (IS_GREATER_THAN_EQUAL(_one)(_one))\
  (IS_GREATER_THAN_EQUAL(_one)(_zero))))

TEST('IS_GREATER_THAN')\
  (ASSERT(IS_GREATER_THAN(_one)(_zero)))


執行結果

```sh
$ python churchNumeral.py
[✓] $IDENTITY
[✓] $SUCCESSOR
[✓] $PREDECESSOR
[✓] $ADDITION
[✓] $SUBTRACTION
[✓] $MULTIPLICATION
[✓] $POWER
[✓] $ABS_DIFFERENCE
[✓] $IS_ZERO
[✓] $IS_LESS_THAN
[✓] $IS_LESS_THAN_EQUAL
[✓] $IS_EQUAL
[✓] $IS_NOT_EQUAL
[✓] $IS_GREATER_THAN_EQUAL
[✓] $IS_GREATER_THAN
```