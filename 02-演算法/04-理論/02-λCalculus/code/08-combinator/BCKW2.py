# 定義 B, C, K, W 組合子

# B 組合子：將 f 和 g 應用到 x 上，然後將 f 應用到 g(x) 上
def B(f):
    return lambda g: lambda x: f(g(x))

# C 組合子：交換參數，C f x y = f y x
def C(f):
    return lambda g: lambda x: f(g(x))

# K 組合子：返回常數，K x y = x
def K(x):
    return lambda y: x

# W 組合子：自我應用，W f x = f (x x)
def W(f):
    return lambda x: f(f(x))

# 用 B, C, K, W 組合出 Y-Combinator
def Y(f):
    return W(lambda x: B(f)(B(W)(W(K)(K))))

# 測試 Y-Combinator，使用遞歸計算階層
def factorial(f):
    return lambda x: 1 if x == 0 else x * f(x - 1)

fact = Y(factorial)

# 測試
print(fact(3))  # 輸出 6
print(fact(5))  # 輸出 120
