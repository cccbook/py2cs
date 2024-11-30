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
