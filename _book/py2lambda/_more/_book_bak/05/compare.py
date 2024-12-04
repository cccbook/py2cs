# Church Booleans : Logic
IF    = lambda c:lambda x:lambda y:c(x)(y) #  if: λ c x y. c x y # if c then x else y.
TRUE  = lambda x:lambda y:x # if true then x # 兩個參數執行第一個
FALSE = lambda x:lambda y:y # if false then y # 兩個參數執行第二個
AND   = lambda p:lambda q:p(q)(p) # if p then q else p
OR    = lambda p:lambda q:p(p)(q) # if p then p else q
XOR   = lambda p:lambda q:p(NOT(q))(q) #  if p then not q else q
NOT   = lambda c:c(FALSE)(TRUE) # if c then false else true

# Church 數字的定義
IDENTITY = lambda x: x
_zero = lambda f: IDENTITY  # 0: 不應用函數 f
_one = lambda f: lambda x: f(x)  # 1: 應用函數 f 一次
_two = lambda f: lambda x: f(f(x))  # 2: 應用函數 f 兩次
_three = lambda f: lambda x: f(f(f(x)))  # 3: 應用函數 f 三次
_four = lambda f: lambda x: f(f(f(f(x))))  # 4: 應用函數 f 4次

# 測試 Church 數字
assert _zero(lambda x: x + 1)(0) == 0  # 0 不改變初始值
assert _one(lambda x: x + 1)(0) == 1  # 1 增加一次
assert _two(lambda x: x + 1)(0) == 2  # 2 增加兩次
assert _three(lambda x: x + 1)(0) == 3  # 3 增加三次

SUCCESSOR = lambda n: lambda f: lambda x: f(n(f)(x))

# 測試繼任者函數
assert _one(lambda x: x + 1)(0) == 1
assert SUCCESSOR(_one)(lambda x: x + 1)(0) == 2  # _one 的繼任者應為 _two
assert SUCCESSOR(_two)(lambda x: x + 1)(0) == 3  # _two 的繼任者應為 _three

ADDITION = lambda m: lambda n: lambda f: lambda x: m(f)(n(f)(x))

# 測試加法
assert ADDITION(_one)(_two)(lambda x: x + 1)(0) == 3  # 1 + 2 = 3
assert ADDITION(_three)(_two)(lambda x: x + 1)(0) == 5  # 3 + 2 = 5

PREDECESSOR = lambda n: lambda f: lambda x: n(
    lambda g: lambda h: h(g(f))
)(lambda _: x)(lambda u: u)

SUBTRACTION = lambda m: lambda n: n(PREDECESSOR)(m)

assert SUBTRACTION(_three)(_one)(lambda x: x + 1)(0) == 2  # 3 - 1 = 2
assert SUBTRACTION(_three)(_three)(lambda x: x + 1)(0) == 0  # 3 - 3 = 0

MULTIPLICATION = lambda m: lambda n: lambda f: m(n(f))

# 測試乘法
assert MULTIPLICATION(_two)(_three)(lambda x: x + 1)(0) == 6  # 2 * 3 = 6
assert MULTIPLICATION(_four)(_two)(lambda x: x + 1)(0) == 8  # 4 * 2 = 8

POWER = lambda x: lambda y: y(x)

# 測試次方
assert POWER(_two)(_three)(lambda x: x + 1)(0) == 8  # 2^3 = 8
assert POWER(_three)(_two)(lambda x: x + 1)(0) == 9  # 3^2 = 9

ABS_DIFFERENCE = lambda x: lambda y: ADDITION(SUBTRACTION(x)(y))(SUBTRACTION(y)(x))

# 測試絕對差
assert ABS_DIFFERENCE(_three)(_two)(lambda x: x + 1)(0) == 1  # |3 - 2| = 1
assert ABS_DIFFERENCE(_two)(_three)(lambda x: x + 1)(0) == 1  # |2 - 3| = 1

# ------------- 比較運算開始 --------------
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

assert IS_EQUAL(_two)(_two) == TRUE  # 2 == 2
assert IS_EQUAL(_three)(_two) == FALSE  # 3 != 2

assert IS_GREATER_THAN(_three)(_two) == TRUE  # 3 > 2
assert IS_GREATER_THAN(_two)(_three) == FALSE  # 2 > 3 不成立

assert IS_LESS_THAN(_two)(_three) == TRUE  # 2 < 3
assert IS_LESS_THAN(_three)(_two) == FALSE  # 3 < 2 不成立

assert IS_GREATER_THAN_EQUAL(_three)(_two) == TRUE  # 3 >= 2
assert IS_GREATER_THAN_EQUAL(_two)(_three) == FALSE  # 2 >= 3 不成立
assert IS_LESS_THAN_EQUAL(_two)(_three) == TRUE  # 2 <= 3
assert IS_LESS_THAN_EQUAL(_three)(_two) == FALSE  # 3 <= 2 不成立

MAX = lambda m: lambda n: IF(IS_GREATER_THAN_EQUAL(m)(n))(m)(n)

assert MAX(_two)(_three)(lambda x: x + 1)(0) == 3  # max(2, 3) = 3
assert MAX(_three)(_two)(lambda x: x + 1)(0) == 3  # max(3, 2) = 3

MIN = lambda m: lambda n: IF(IS_LESS_THAN_EQUAL(m)(n))(m)(n)

assert MIN(_two)(_three)(lambda x: x + 1)(0) == 2  # min(2, 3) = 2
assert MIN(_three)(_two)(lambda x: x + 1)(0) == 2  # min(3, 2) = 2

