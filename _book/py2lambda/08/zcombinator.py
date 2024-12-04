def Z(f):
    return (lambda x: f(lambda y: x(x)(y)))(lambda x: f(lambda y: x(x)(y)))

# 測試 Z-Combinator 實現
# 定義一個簡單的遞歸函數，例如計算階乘
def factorial(f):
    return lambda x: 1 if x == 0 else x * f(x - 1)

# 計算 5 的階乘
print(f'factorial(5)={Z(factorial)(5)}')  # 輸出 120
