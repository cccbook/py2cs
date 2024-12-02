# Python 中的高階函數
def apply_twice(f, x):
    return f(f(x))

# 定義一個簡單的函數
square = lambda x: x * x

# 使用高階函數
result = apply_twice(square, 3)  # 先平方 3，再對結果平方
print(result)  # 輸出 81 (即 (3^2)^2)